#!/bin/sh
set -e

if ! command -v higgsfield >/dev/null 2>&1; then
  curl -fsSL https://raw.githubusercontent.com/higgsfield-ai/cli/main/install.sh | sh || \
    echo "higgsfield CLI install failed, voice replies will fall back to text"
fi

if [ -n "$HIGGSFIELD_CREDENTIALS_JSON" ]; then
  mkdir -p "$HOME/.config/higgsfield"
  printf '%s' "$HIGGSFIELD_CREDENTIALS_JSON" > "$HOME/.config/higgsfield/credentials.json"
fi

exec python bot.py
