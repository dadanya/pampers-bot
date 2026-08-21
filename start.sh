#!/bin/sh
set -e

if [ -n "$HIGGSFIELD_CREDENTIALS_JSON" ]; then
  mkdir -p "$HOME/.config/higgsfield"
  printf '%s' "$HIGGSFIELD_CREDENTIALS_JSON" > "$HOME/.config/higgsfield/credentials.json"
fi

exec python bot.py
