#!/bin/bash
# Fonte canônica: templates/harness/protect-secrets.sh (guia-construcao-harness.md §6.1) — instalado em .claude/ pelo /setup
input=$(cat)
file_path=$(echo "$input" | jq -r '.tool_input.file_path // .tool_input.filePath // .tool_input.path // empty' 2>/dev/null)

if [ -n "$file_path" ]; then
  if echo "$file_path" | grep -v "protect-secrets" | grep -q -i -E '\.env|/secrets/|key\.pem$|id_rsa$'; then
    echo '{"continue": false, "stopReason": "Acesso bloqueado em arquivos confidenciais (.env, secrets, chaves)."}'
    exit 0
  fi
fi

if echo "$input" | grep -v "protect-secrets" | grep -q -i -E '"(file_path|filePath|path)":\s*"[^"]*(\\.env|/secrets/|key\.pem|id_rsa)'; then
  echo '{"continue": false, "stopReason": "Acesso bloqueado em arquivos confidenciais (.env, secrets, chaves)."}'
  exit 0
fi

exit 0
