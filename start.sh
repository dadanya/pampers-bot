#!/bin/sh
set -e

echo "start.sh: begin setup"

if ! command -v higgsfield >/dev/null 2>&1; then
  echo "start.sh: higgsfield not found, installing..."
  if curl -fsSL https://raw.githubusercontent.com/higgsfield-ai/cli/main/install.sh | sh; then
    echo "start.sh: higgsfield installed"
  else
    echo "start.sh: higgsfield install failed, voice replies will fall back to text"
  fi
else
  echo "start.sh: higgsfield already present"
fi

if [ -n "$HIGGSFIELD_CREDENTIALS_JSON" ]; then
  mkdir -p "$HOME/.config/higgsfield"
  printf '%s' "$HIGGSFIELD_CREDENTIALS_JSON" > "$HOME/.config/higgsfield/credentials.json"
  echo "start.sh: wrote higgsfield credentials.json"
fi

if [ -n "$HIGGSFIELD_CONFIG_JSON" ]; then
  mkdir -p "$HOME/.config/higgsfield"
  printf '%s' "$HIGGSFIELD_CONFIG_JSON" > "$HOME/.config/higgsfield/config.json"
  echo "start.sh: wrote higgsfield config.json"
fi

echo "start.sh: setup done, starting bot"
exec python bot.py
