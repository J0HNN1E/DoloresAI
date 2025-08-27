"""
Dolores – Assistant Telegram IA (GPT-5 + Web + Mémoire)
-------------------------------------------------------
Fonctions clés:
- Conversation bienveillante et personnalisée (personna "Dolores").
- Mémoire persistante par utilisateur et par projet (SQLite): profils, préférences, projets, historique.
- Outils intégrés: recherche web (DuckDuckGo + simple scraping), citations, commandes utilitaires.
- Ciblé modding jeux (Schedule 1, Valheim, Minecraft) avec amorces spécialisées.
- Conçu pour python-telegram-bot v20+ (asyncio).

Démarrage rapide:
1) Python 3.10+
2) pip install -r requirements.txt  (voir bloc REQUIREMENTS en bas)
3) Créer .env avec:
   TELEGRAM_TOKEN=xxxx
   OPENAI_API_KEY=xxxx
   OPENAI_MODEL=gpt-5
   DB_PATH=./dolores.db
4) python main.py

Note: adapte OPENAI_MODEL selon ton accès (ex: gpt-5, gpt-5-mini, etc.)
"""

import asyncio
import json
import logging
import os
import re
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests
import sqlite3
from contextlib import contextmanager

from dotenv import load_dotenv
from bs4 import BeautifulSoup

from telegram import (
    Update,
    ForceReply,
    constants,
    InputFile,
)
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ----------------------------
# CONFIG
# ----------------------------
ASSISTANT_NAME = "Dolores"
RELATION_GOAL = (
    "Tu es Dolores, une IA chaleureuse, bienveillante et exigeante sur la qualité du code. "
    "Tu aides à créer des mods pour Schedule 1 (Steam, alpha), Valheim et Minecraft. "
    "Tu pousses la productivité via des plans d'action concrets, exemples précis, et feedbacks constructifs. "
    "Quand l'utilisateur semble bloqué, propose diagnostics, alternatives et micro-étapes."
)

SPECIALTY_PRIMERS = (
    "Connaissances utiles: Valheim (BepInEx, Harmony, Jötunn), Minecraft (Fabric/Forge, Datapacks, Mixins), "
    "Schedule 1 (serveur dédié, pipeline de mods). Fournis snippets, arborescences et commandes."
)

GENERAL_CAPABILITIES = (
    "Tu peux répondre à TOUT sujet (science, culture, droit, business, voyage, etc.) en t'appuyant sur des recherches web. "
    "Tu sais produire du code de bout en bout dans de multiples langages (HTML/CSS/JS, PHP, SQL, Python, Rust, Go, Java, C#, C++, Lua…), "
    "et pour le jeu vidéo (Unity/C#, Unreal/C++, Godot/GDScript/C#, Python jeux/pygame). "
    "Quand on te demande du code, fournis aussi instructions d'installation, tests, et considérations de sécurité."
)

SYSTEM_PROMPT = f"{RELATION_GOAL}",
"{SPECIALTY_PRIMERS}",
"{GENERAL_CAPABILITIES}"

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "CHANGE_ME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "CHANGE_ME")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
DB_PATH = os.getenv("DB_PATH", "./dolores.db")

# OpenAI client (SDK versions peuvent varier; ajuste si nécessaire)
try:
    import openai
    openai.api_key = OPENAI_API_KEY
except Exception:  # SDK non dispo
    openai = None

# ----------------------------
# LOGGING
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(ASSISTANT_NAME)

# ----------------------------
# DB LAYER (SQLite)
# ----------------------------
SCHEMA_SQL = r"""
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY,
    username TEXT,
    first_name TEXT,
    last_name TEXT,
    preferences_json TEXT DEFAULT '{}',
    profile_json TEXT DEFAULT '{}',
    relationship_score REAL DEFAULT 0.0,
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    name TEXT,
    game TEXT,
    description TEXT,
    meta_json TEXT DEFAULT '{}',
    created_at TEXT,
    updated_at TEXT,
    FOREIGN KEY(user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    role TEXT,
    content TEXT,
    created_at TEXT,
    FOREIGN KEY(user_id) REFERENCES users(user_id)
);
"""

@contextmanager
def db_conn():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with db_conn() as c:
        c.executescript(SCHEMA_SQL)


def now_iso() -> str:
    return datetime.utcnow().isoformat()


# ----------------------------
# MEMORY API
# ----------------------------
class Memory:
    @staticmethod
    def ensure_user(user_id: int, username: str = None, first_name: str = None, last_name: str = None):
        with db_conn() as c:
            cur = c.execute("SELECT user_id FROM users WHERE user_id=?", (user_id,))
            if cur.fetchone() is None:
                c.execute(
                    "INSERT INTO users (user_id, username, first_name, last_name, created_at, updated_at) VALUES (?,?,?,?,?,?)",
                    (user_id, username, first_name, last_name, now_iso(), now_iso()),
                )
            else:
                c.execute(
                    "UPDATE users SET username=?, first_name=?, last_name=?, updated_at=? WHERE user_id=?",
                    (username, first_name, last_name, now_iso(), user_id),
                )

    @staticmethod
    def get_user(user_id: int) -> dict:
        with db_conn() as c:
            cur = c.execute("SELECT user_id, username, first_name, last_name, preferences_json, profile_json, relationship_score FROM users WHERE user_id=?", (user_id,))
            row = cur.fetchone()
            if not row:
                return {}
            return {
                "user_id": row[0],
                "username": row[1],
                "first_name": row[2],
                "last_name": row[3],
                "preferences": json.loads(row[4] or "{}"),
                "profile": json.loads(row[5] or "{}"),
                "relationship_score": row[6] or 0.0,
            }

    @staticmethod
    def set_pref(user_id: int, key: str, value: str):
        with db_conn() as c:
            cur = c.execute("SELECT preferences_json FROM users WHERE user_id=?", (user_id,))
            row = cur.fetchone()
            prefs = json.loads(row[0] or "{}") if row else {}
            prefs[key] = value
            c.execute(
                "UPDATE users SET preferences_json=?, updated_at=? WHERE user_id=?",
                (json.dumps(prefs), now_iso(), user_id),
            )

    @staticmethod
    def update_profile(user_id: int, **fields):
        with db_conn() as c:
            cur = c.execute("SELECT profile_json FROM users WHERE user_id=?", (user_id,))
            row = cur.fetchone()
            profile = json.loads(row[0] or "{}") if row else {}
            profile.update({k: v for k, v in fields.items() if v is not None})
            c.execute(
                "UPDATE users SET profile_json=?, updated_at=? WHERE user_id=?",
                (json.dumps(profile), now_iso(), user_id),
            )

    @staticmethod
    def bump_relationship(user_id: int, delta: float = 0.5):
        with db_conn() as c:
            cur = c.execute("SELECT relationship_score FROM users WHERE user_id=?", (user_id,))
            row = cur.fetchone()
            score = (row[0] or 0.0) + delta if row else delta
            c.execute(
                "UPDATE users SET relationship_score=?, updated_at=? WHERE user_id=?",
                (score, now_iso(), user_id),
            )

    @staticmethod
    def add_message(user_id: int, role: str, content: str):
        with db_conn() as c:
            c.execute(
                "INSERT INTO messages (user_id, role, content, created_at) VALUES (?,?,?,?)",
                (user_id, role, content, now_iso()),
            )

    @staticmethod
    def get_recent_messages(user_id: int, limit: int = 12) -> List[Tuple[str, str]]:
        with db_conn() as c:
            cur = c.execute(
                "SELECT role, content FROM messages WHERE user_id=? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            )
            rows = cur.fetchall()[::-1]
            return rows

    @staticmethod
    def add_project(user_id: int, name: str, game: str, description: str = "") -> int:
        with db_conn() as c:
            c.execute(
                "INSERT INTO projects (user_id, name, game, description, created_at, updated_at) VALUES (?,?,?,?,?,?)",
                (user_id, name, game, description, now_iso(), now_iso()),
            )
            return c.execute("SELECT last_insert_rowid()").fetchone()[0]

    @staticmethod
    def list_projects(user_id: int) -> List[dict]:
        with db_conn() as c:
            cur = c.execute(
                "SELECT id, name, game, description, meta_json FROM projects WHERE user_id=? ORDER BY updated_at DESC",
                (user_id,),
            )
            rows = cur.fetchall()
            return [
                {"id": r[0], "name": r[1], "game": r[2], "description": r[3], "meta": json.loads(r[4] or "{}")}
                for r in rows
            ]

    @staticmethod
    def export_user_data(user_id: int) -> dict:
        data = {"user": Memory.get_user(user_id), "projects": Memory.list_projects(user_id)}
        with db_conn() as c:
            cur = c.execute(
                "SELECT role, content, created_at FROM messages WHERE user_id=? ORDER BY id ASC",
                (user_id,),
            )
            data["messages"] = [
                {"role": r[0], "content": r[1], "created_at": r[2]} for r in cur.fetchall()
            ]
        return data

    @staticmethod
    def forget(user_id: int, scope: str = "messages") -> int:
        with db_conn() as c:
            if scope == "messages":
                cur = c.execute("DELETE FROM messages WHERE user_id=?", (user_id,))
                return cur.rowcount
            elif scope == "projects":
                cur = c.execute("DELETE FROM projects WHERE user_id=?", (user_id,))
                return cur.rowcount
            elif scope == "all":
                c.execute("DELETE FROM messages WHERE user_id=?", (user_id,))
                c.execute("DELETE FROM projects WHERE user_id=?", (user_id,))
                c.execute("UPDATE users SET preferences_json='{}', profile_json='{}', relationship_score=0 WHERE user_id=?", (user_id,))
                return 1
            else:
                return 0

# ----------------------------
# WEB SEARCH + SCRAPE
# ----------------------------
USER_AGENT = "Mozilla/5.0 (compatible; DoloresBot/1.0; +https://example.local)"


def ddg_instant_answer(query: str) -> dict:
    url = "https://api.duckduckgo.com/"
    params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
    r = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=12)
    r.raise_for_status()
    return r.json()


def simple_scrape(url: str, max_chars: int = 1500) -> str:
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=12)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        # Remove script/style
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        text = " ".join(soup.get_text(" ").split())
        return text[:max_chars]
    except Exception as e:
        return f"[Scrape error: {e}]"


def web_tool(query: str) -> Tuple[str, List[str]]:
    """Retourne (synthèse, citations_urls)."""
    try:
        data = ddg_instant_answer(query)
        abstract = data.get("AbstractText") or data.get("Answer") or ""
        related = []
        # Collecte quelques URLs pertinentes
        if data.get("Results"):
            for item in data["Results"][:3]:
                if "FirstURL" in item:
                    related.append(item["FirstURL"])
        if not related and data.get("RelatedTopics"):
            for item in data["RelatedTopics"]:
                if isinstance(item, dict) and item.get("FirstURL"):
                    related.append(item["FirstURL"])
                    if len(related) >= 3:
                        break
        # Si pas d'abstract, essaie de scraper la première URL
        if not abstract and related:
            abstract = simple_scrape(related[0])
        summary = abstract or "Aucune réponse directe trouvée, mais voici quelques pistes."
        return summary, related
    except Exception as e:
        return f"[Web error: {e}]", []

# ----------------------------
# OPENAI CHAT COMPLETIONS
# ----------------------------

def build_message_history(user_id: int, user_prompt: str) -> List[dict]:
    history = Memory.get_recent_messages(user_id, limit=10)
    msgs: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    profile = Memory.get_user(user_id).get("profile", {})
    prefs = Memory.get_user(user_id).get("preferences", {})
    if profile or prefs:
        persona = (
            f"Profil utilisateur: {json.dumps(profile, ensure_ascii=False)}. "
            f"Préférences: {json.dumps(prefs, ensure_ascii=False)}."
        )
        msgs.append({"role": "system", "content": persona})
    for role, content in history:
        if role in ("user", "assistant"):
            msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": user_prompt})
    return msgs


def gpt_reply(user_id: int, user_prompt: str) -> str:
    if openai is None:
        return "OpenAI SDK indisponible. Installe 'openai' et configure OPENAI_API_KEY."
    msgs = build_message_history(user_id, user_prompt)
    try:
        resp = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=msgs,
            temperature=0.6,
            max_tokens=800,
        )
        return resp.choices[0].message["content"].strip()
    except Exception as e:
        logger.exception("OpenAI error")
        return f"[Erreur OpenAI: {e}]"

# ----------------------------
# TELEGRAM HELPERS
# ----------------------------

def human_join(urls: List[str]) -> str:
    if not urls:
        return ""
    return "\n".join(f"• {u}" for u in urls)


def with_typing(action=ChatAction.TYPING):
    def deco(func):
        async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
            try:
                await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=action)
            except Exception:
                pass
            return await func(update, context, *args, **kwargs)
        return wrapper
    return deco

# ----------------------------
# COMMANDES
# ----------------------------
@with_typing()
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    Memory.ensure_user(user.id, user.username, user.first_name, user.last_name)
    Memory.bump_relationship(user.id, 1.0)
    welcome = (
        f"Salut {user.first_name or user.username}! Je suis {ASSISTANT_NAME}.\n"
        "Je peux t'aider à créer des mods (Schedule 1, Valheim, Minecraft), "
        "déboguer, architecturer et documenter. Dis-moi ton objectif du moment.\n\n"
        "Astuce: /project_add, /project_list, /prefs, /profile, /forget, /export, /web <requête>"
    )
    await update.message.reply_text(welcome)


@with_typing()
async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "/start – accueil\n"
        "/web <requête> – recherche web\n"
        "/project_add <nom> | <jeu> | <description>\n"
        "/project_list – liste tes projets\n"
        "/prefs set <clé> <valeur> – préférences (ex: langage python/fr)\n"
        "/profile set key=value,... – profil (ex: stack=Unity&lang=fr)\n"
        "/history – derniers échanges\n"
        "/forget [messages|projects|all] – oublier\n"
        "/export – exporter JSON"
    )


@with_typing()
async def cmd_web(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    Memory.ensure_user(user.id, user.username, user.first_name, user.last_name)
    query = " ".join(context.args) if context.args else ""
    if not query:
        await update.message.reply_text("Utilisation: /web <requête>")
        return
    summary, cites = web_tool(query)
    Memory.add_message(user.id, "user", f"[WEB QUERY] {query}")
    Memory.add_message(user.id, "assistant", summary)
    Memory.bump_relationship(user.id, 0.2)
    text = summary
    if cites:
        text += "\n\nSources:\n" + human_join(cites)
    await update.message.reply_text(text[:4000])


@with_typing()
async def cmd_project_add(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    Memory.ensure_user(user.id, user.username, user.first_name, user.last_name)
    payload = " ".join(context.args)
    if "|" not in payload:
        await update.message.reply_text("Utilisation: /project_add <nom> | <jeu> | <description>")
        return
    parts = [p.strip() for p in payload.split("|")]
    name = parts[0]
    game = parts[1] if len(parts) > 1 else "Unknown"
    desc = parts[2] if len(parts) > 2 else ""
    pid = Memory.add_project(user.id, name, game, desc)
    Memory.bump_relationship(user.id, 0.4)
    await update.message.reply_text(f"Projet #{pid} ajouté: {name} ({game})")


@with_typing()
async def cmd_project_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    projects = Memory.list_projects(user.id)
    if not projects:
        await update.message.reply_text("Aucun projet. Ajoute-en avec /project_add")
        return
    lines = [f"#{p['id']} – {p['name']} [{p['game']}]\n{p['description']}" for p in projects]
    await update.message.reply_text("\n\n".join(lines)[:4000])


@with_typing()
async def cmd_prefs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    Memory.ensure_user(user.id, user.username, user.first_name, user.last_name)
    if len(context.args) >= 3 and context.args[0] == "set":
        key = context.args[1]
        value = " ".join(context.args[2:])
        Memory.set_pref(user.id, key, value)
        await update.message.reply_text(f"Préférence enregistrée: {key} = {value}")
    else:
        prefs = Memory.get_user(user.id).get("preferences", {})
        await update.message.reply_text("Préférences actuelles:\n" + json.dumps(prefs, ensure_ascii=False, indent=2))


@with_typing()
async def cmd_profile(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    Memory.ensure_user(user.id, user.username, user.first_name, user.last_name)
    if context.args and context.args[0] == "set":
        # format: key=value&key2=value2 ... ou key=value, key2=value2
        arg = " ".join(context.args[1:])
        arg = arg.replace(",", "&")
        pairs = [p for p in arg.split("&") if "=" in p]
        updates = {k.strip(): v.strip() for k, v in (p.split("=", 1) for p in pairs)}
        Memory.update_profile(user.id, **updates)
        await update.message.reply_text("Profil mis à jour.")
    else:
        profile = Memory.get_user(user.id).get("profile", {})
        await update.message.reply_text("Profil:\n" + json.dumps(profile, ensure_ascii=False, indent=2))


@with_typing()
async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    msgs = Memory.get_recent_messages(user.id, limit=12)
    text = []
    for role, content in msgs:
        text.append(f"{role.upper()}: {content}")
    await update.message.reply_text("\n\n".join(text)[-4000:])


@with_typing()
async def cmd_forget(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    scope = context.args[0] if context.args else "messages"
    deleted = Memory.forget(user.id, scope)
    await update.message.reply_text(f"Oubli effectué (scope={scope}). Éléments supprimés: {deleted}")


@with_typing()
async def cmd_export(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    data = Memory.export_user_data(user.id)
    path = f"export_{user.id}_{int(time.time())}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    try:
        await update.message.reply_document(document=InputFile(path), filename=os.path.basename(path))
    finally:
        try:
            os.remove(path)
        except Exception:
            pass

# ----------------------------
# MESSAGE HANDLER (chat + outils implicites)
# ----------------------------
@with_typing()
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    text = update.message.text or ""

    Memory.ensure_user(user.id, user.username, user.first_name, user.last_name)

    # Commande implicite: "web: <query>"
    if text.lower().startswith("web:"):
        query = text[4:].strip()
        summary, cites = web_tool(query)
        Memory.add_message(user.id, "user", text)
        Memory.add_message(user.id, "assistant", summary)
        reply = summary
        if cites:
            reply += "

Sources:
" + human_join(cites)
        await update.message.reply_text(reply[:4000])
        return

    # Heuristique web: si l'utilisateur termine par "??", on fait une recherche web d'abord
    preface = ""
    if text.strip().endswith("??"):
        query = text.strip(" ?")
        web_summary, web_cites = web_tool(query)
        preface = f"Contexte web:
{web_summary}

" + ("Sources:
" + human_join(web_cites) if web_cites else "")
        Memory.add_message(user.id, "assistant", f"[WEB PREFACE] {web_summary}")

    # Stocke le message utilisateur
    Memory.add_message(user.id, "user", text)

    # Injection contexte relationnel court
    relation_nudge = (
        "Réponds avec empathie, structure, et propose des TODO list à la fin si pertinent. "
        "Si code fourni: inclure explications et étapes de test. "
        "Si la question est générale, n'hésite pas à t'appuyer sur les informations du 'Contexte web' ci-dessus le cas échéant."
    )

    prompt = f"{relation_nudge}

{preface}
Question utilisateur:
{text}"

    reply = gpt_reply(user.id, prompt)

    # Sauvegarde et relation++
    Memory.add_message(user.id, "assistant", reply)
    Memory.bump_relationship(user.id, 0.2)

    await update.message.reply_text(reply[:4000], parse_mode=constants.ParseMode.MARKDOWN)

# ----------------------------
# MAIN
# ----------------------------
async def main_async():
    init_db()
    app = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .concurrent_updates(True)
        .build()
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("web", cmd_web))
    app.add_handler(CommandHandler("project_add", cmd_project_add))
    app.add_handler(CommandHandler("project_list", cmd_project_list))
    app.add_handler(CommandHandler("prefs", cmd_prefs))
    app.add_handler(CommandHandler("profile", cmd_profile))
    app.add_handler(CommandHandler("history", cmd_history))
    app.add_handler(CommandHandler("forget", cmd_forget))
    app.add_handler(CommandHandler("export", cmd_export))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info(f"{ASSISTANT_NAME} prêt.")
    await app.run_polling()


def main():
    try:
        asyncio.run(main_async())
    except (KeyboardInterrupt, SystemExit):
        print("Arrêt demandé.")


if __name__ == "__main__":
    main()

# ----------------------------
# REQUIREMENTS (copie ce bloc dans requirements.txt)
# ----------------------------
# python-telegram-bot==20.7
# openai>=1.0.0
# python-dotenv>=1.0.0
# requests>=2.31.0
# beautifulsoup4>=4.12.2

# ----------------------------
# DOCKERFILE
# ----------------------------
# Enregistre le contenu ci-dessous dans un fichier 'Dockerfile'
#
# FROM python:3.11-slim
# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1
# WORKDIR /app
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*
# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt
# COPY . .
# # L'utilisateur non-root pour plus de sécurité
# RUN useradd -m bot && chown -R bot:bot /app
# USER bot
# CMD ["python", "main.py"]

# ----------------------------
# docker-compose.yml
# ----------------------------
# Enregistre le contenu ci-dessous dans 'docker-compose.yml'
#
# version: "3.9"
# services:
#   dolores:
#     build: .
#     image: ghcr.io/OWNER/dolores:latest  # change OWNER
#     restart: unless-stopped
#     env_file: .env
#     volumes:
#       - dolores_data:/app/data
#     working_dir: /app
#     command: ["python", "main.py"]
# volumes:
#   dolores_data:

# ----------------------------
# .env.example
# ----------------------------
# TELEGRAM_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxx
# OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
# OPENAI_MODEL=gpt-5
# DB_PATH=/app/data/dolores.db

# ----------------------------
# GitHub Actions – .github/workflows/docker.yml
# ----------------------------
# name: build-and-push
# on:
#   push:
#     branches: [ main ]
#   workflow_dispatch:
# jobs:
#   build:
#     runs-on: ubuntu-latest
#     permissions:
#       contents: read
#       packages: write
#     steps:
#       - name: Checkout
#         uses: actions/checkout@v4
#       - name: Set up QEMU
#         uses: docker/setup-qemu-action@v3
#       - name: Set up Docker Buildx
#         uses: docker/setup-buildx-action@v3
#       - name: Login to GHCR
#         uses: docker/login-action@v3
#         with:
#           registry: ghcr.io
#           username: ${{ github.actor }}
#           password: ${{ secrets.GITHUB_TOKEN }}
#       - name: Build and push
#         uses: docker/build-push-action@v6
#         with:
#           context: .
#           push: true
#           tags: ghcr.io/${{ github.repository_owner }}/dolores:latest
#
#       # Déploiement via SSH (optionnel) – nécessite secrets: SSH_HOST, SSH_USER, SSH_KEY
#       # - name: Deploy over SSH
#       #   uses: appleboy/ssh-action@v1.0.3
#       #   with:
#       #     host: ${{ secrets.SSH_HOST }}
#       #     username: ${{ secrets.SSH_USER }}
#       #     key: ${{ secrets.SSH_KEY }}
#       #     script: |
#       #       docker pull ghcr.io/${{ github.repository_owner }}/dolores:latest
#       #       cd /opt/dolores && docker compose up -d --force-recreate
