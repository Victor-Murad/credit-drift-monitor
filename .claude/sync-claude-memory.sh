#!/bin/bash
PROJECT_ROOT=$(git rev-parse --show-toplevel)
SANITIZED_PATH=$(echo "$PROJECT_ROOT" | sed 's/[^a-zA-Z0-9]/-/g')

CONFIG_HOME="$HOME/.openclaude"
if [ ! -d "$CONFIG_HOME" ] && [ -d "$HOME/.claude" ]; then
  CONFIG_HOME="$HOME/.claude"
fi

GLOBAL_MEM_PATH="$CONFIG_HOME/projects/$SANITIZED_PATH/memory/"
LOCAL_MEM_PATH="$PROJECT_ROOT/.claude/memory/"

# copia os .md de $1 para $2 mantendo as subpastas; sem rsync, que o Git Bash (Windows) não tem
copia_md() {
  [ -d "$1" ] || return 0
  (cd "$1" && find . -type f -name '*.md') | while IFS= read -r f; do
    mkdir -p "$2/$(dirname "$f")" && cp "$1/$f" "$2/$f"
  done
}

if [ "$1" == "pull" ]; then
  mkdir -p "$LOCAL_MEM_PATH"
  if [ -d "$GLOBAL_MEM_PATH" ]; then
    copia_md "$GLOBAL_MEM_PATH" "$LOCAL_MEM_PATH"
    git add "$LOCAL_MEM_PATH"
  fi
elif [ "$1" == "push" ]; then
  mkdir -p "$GLOBAL_MEM_PATH"
  copia_md "$LOCAL_MEM_PATH" "$GLOBAL_MEM_PATH"
else
  echo "Uso: ./.claude/sync-claude-memory.sh [pull|push]"
  exit 1
fi
