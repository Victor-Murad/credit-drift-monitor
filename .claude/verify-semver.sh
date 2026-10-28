#!/bin/bash
# Fonte canônica: templates/harness/verify-semver.sh (guia-construcao-harness.md §6.1) — instalado em .claude/ pelo /setup
# verify-semver.sh
# Garante de forma automatizada que a versão SemVer seja atualizada antes de realizar merges ou pushes na branch main / develop

COMMAND=$(jq -r '.tool_input.command // empty' 2>/dev/null)

if [ -z "$COMMAND" ]; then
    exit 0
fi

# Verifica se estamos dentro de um repositório git válido
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    exit 0
fi

CURRENT_BRANCH=$(git branch --show-current 2>/dev/null)
ROOT_DIR=$(git rev-parse --show-toplevel 2>/dev/null || pwd)

# Determina se a operação é uma promoção para a branch main / develop
IS_PUSH_PROTECTED=false
IS_MERGE_INTO_PROTECTED=false
TARGET_MERGE_REF=""

# 1. Checa comandos de push
if echo "$COMMAND" | grep -qE "git push.*(\bmain\b|\bdevelop\b|HEAD:main|HEAD:develop)"; then
    IS_PUSH_PROTECTED=true
elif [ "$CURRENT_BRANCH" = "main" ] || [ "$CURRENT_BRANCH" = "develop" ]; then
    if echo "$COMMAND" | grep -qE "git push\b" && ! echo "$COMMAND" | grep -qE "git push.*(origin|upstream)[[:space:]]+(dev|feature|feat|fix|hotfix|chore|docs)"; then
        IS_PUSH_PROTECTED=true
    fi
fi

# 2. Checa comandos de merge quando estamos na branch protegida (main / develop)
if [ "$CURRENT_BRANCH" = "main" ] || [ "$CURRENT_BRANCH" = "develop" ]; then
    if echo "$COMMAND" | grep -qE "git merge\b" && ! echo "$COMMAND" | grep -qE "git merge[[:space:]]+(--abort|--continue|origin/main|origin/develop)"; then
        IS_MERGE_INTO_PROTECTED=true
        # Extrai o nome da branch/ref que está sendo mergeada
        TARGET_MERGE_REF=$(echo "$COMMAND" | grep -oE "git merge[[:space:]]+(\S+)" | awk '{print $3}')
    fi
fi

if [ "$IS_PUSH_PROTECTED" = true ] || [ "$IS_MERGE_INTO_PROTECTED" = true ]; then
    BASE_REF="origin/main"
    if ! git rev-parse --verify "$BASE_REF" >/dev/null 2>&1; then
        BASE_REF="origin/develop"
        if ! git rev-parse --verify "$BASE_REF" >/dev/null 2>&1; then
            BASE_REF="main"
        fi
    fi

    # Se estamos fazendo merge em main/develop, verificamos a versão da branch que está sendo mergeada
    REF_TO_CHECK="HEAD"
    if [ "$IS_MERGE_INTO_PROTECTED" = true ] && [ -n "$TARGET_MERGE_REF" ]; then
        if git rev-parse --verify "$TARGET_MERGE_REF" >/dev/null 2>&1; then
            REF_TO_CHECK="$TARGET_MERGE_REF"
        fi
    fi

    # Verifica se há commits novos entre BASE_REF e a ref sendo promovida
    COMMITS_AHEAD=$(git rev-list --count "${BASE_REF}..${REF_TO_CHECK}" 2>/dev/null || echo "0")
    if [ "$COMMITS_AHEAD" -eq 0 ] && git diff --quiet "${BASE_REF}" "${REF_TO_CHECK}" 2>/dev/null; then
        # Sem mudanças ou sem commits à frente, permite
        exit 0
    fi

    VERSION_LOCAL=""
    VERSION_BASE=""

    # 1. Python (pyproject.toml)
    if [ -f "$ROOT_DIR/pyproject.toml" ]; then
        # Lê a versão da branch que está sendo promovida/mergeada
        VERSION_LOCAL=$(git show "${REF_TO_CHECK}:pyproject.toml" 2>/dev/null | grep -E "^version =" | head -n 1 | cut -d'"' -f2)
        [ -z "$VERSION_LOCAL" ] && VERSION_LOCAL=$(git show "${REF_TO_CHECK}:pyproject.toml" 2>/dev/null | grep -E "^version =" | head -n 1 | cut -d"'" -f2)
        [ -z "$VERSION_LOCAL" ] && VERSION_LOCAL=$(grep -E "^version =" "$ROOT_DIR/pyproject.toml" 2>/dev/null | head -n 1 | cut -d'"' -f2)
        [ -z "$VERSION_LOCAL" ] && VERSION_LOCAL=$(grep -E "^version =" "$ROOT_DIR/pyproject.toml" 2>/dev/null | head -n 1 | cut -d"'" -f2)

        VERSION_BASE=$(git show "${BASE_REF}:pyproject.toml" 2>/dev/null | grep -E "^version =" | head -n 1 | cut -d'"' -f2)
        [ -z "$VERSION_BASE" ] && VERSION_BASE=$(git show "${BASE_REF}:pyproject.toml" 2>/dev/null | grep -E "^version =" | head -n 1 | cut -d"'" -f2)
    # 2. Node / Angular (package.json)
    elif [ -f "$ROOT_DIR/package.json" ]; then
        VERSION_LOCAL=$(git show "${REF_TO_CHECK}:package.json" 2>/dev/null | jq -r '.version // empty' 2>/dev/null)
        [ -z "$VERSION_LOCAL" ] || [ "$VERSION_LOCAL" = "null" ] && VERSION_LOCAL=$(jq -r '.version // empty' "$ROOT_DIR/package.json" 2>/dev/null)

        VERSION_BASE=$(git show "${BASE_REF}:package.json" 2>/dev/null | jq -r '.version // empty' 2>/dev/null)
    fi

    if [ -n "$VERSION_LOCAL" ] && [ -n "$VERSION_BASE" ] && [ "$VERSION_BASE" != "null" ] && [ "$VERSION_LOCAL" = "$VERSION_BASE" ]; then
        MSG="⚠️ [BLOQUEIO DE SEGURANÇA SEMVER]\n\nA versão ($VERSION_LOCAL) na branch '${REF_TO_CHECK}' é idêntica à versão existente na branch base '${BASE_REF}' ($VERSION_BASE)!\n\nDetectamos $COMMITS_AHEAD novo(s) commit(s) sendo integrados.\nAntes de fazer merge ou push na branch protegida, você DEVE atualizar a versão SemVer (Patch, Minor ou Major) em pyproject.toml ou package.json."
        echo "{\"continue\": false, \"stopReason\": \"$MSG\"}"
        exit 0
    fi
fi

exit 0
