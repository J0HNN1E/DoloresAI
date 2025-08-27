#!/bin/bash

echo "🚀 Installation de Dolores (Chatbot IA Telegram + GPT-5 + Web + Voix)"

# Vérification des dépendances
for cmd in git docker docker-compose; do
  if ! command -v $cmd &> /dev/null; then
    echo "❌ $cmd n'est pas installé. Installe-le avant de continuer."
    exit 1
  fi
done

# Clonage du repo (si non déjà présent)
if [ ! -d "dolores" ]; then
  git clone https://github.com/ton-compte/dolores.git
fi

cd dolores

# Création du fichier .env
if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "👉 Entrez votre token Telegram (via BotFather) : "
  read TELEGRAM_BOT_TOKEN
  echo "👉 Entrez votre clé OpenAI API : "
  read OPENAI_API_KEY

  sed -i "s|TELEGRAM_BOT_TOKEN=.*|TELEGRAM_BOT_TOKEN=$TELEGRAM_BOT_TOKEN|" .env
  sed -i "s|OPENAI_API_KEY=.*|OPENAI_API_KEY=$OPENAI_API_KEY|" .env
fi

# Build & start
docker-compose build
docker-compose up -d

echo "✅ Dolores est lancée !"
echo "👉 Ouvre Telegram et envoie /start à ton bot pour commencer."
