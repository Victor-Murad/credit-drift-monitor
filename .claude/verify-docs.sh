#!/bin/bash
# Fonte canônica: templates/harness/verify-docs.sh (guia-construcao-harness.md §6.1) — instalado em .claude/ pelo /setup
COMMAND=$(jq -r '.tool_input.command' 2>/dev/null)

if echo "$COMMAND" | grep -qE "git merge|git push.*main|git push.*develop|gh pr create"; then
    CURRENT_BRANCH=$(git branch --show-current)
    if [ "$CURRENT_BRANCH" != "main" ]; then
        BASE_COMMIT=$(git merge-base main HEAD 2>/dev/null)
        MODIFIED_FILES=$(git diff --name-only "$BASE_COMMIT"...HEAD 2>/dev/null)
        LOCAL_MODIFIED=$(git status --porcelain | awk '{print $2}')
        ALL_FILES=$(echo -e "${MODIFIED_FILES}\n${LOCAL_MODIFIED}" | sort -u)

        HAS_CODE_CHANGES=false
        HAS_DOCS_CHANGES=false
        echo "$ALL_FILES" | grep -qE "^src/" && HAS_CODE_CHANGES=true
        echo "$ALL_FILES" | grep -qE "^docs/|^\.claude/memory/|\.md$" && HAS_DOCS_CHANGES=true

        if [ "$HAS_CODE_CHANGES" = true ] && [ "$HAS_DOCS_CHANGES" = false ]; then
            MSG="⚠️ [BLOQUEIO DE DOCUMENTAÇÃO] Alterações em src/ detectadas sem atualização de docs/. Atualize a documentação antes do merge."
            echo "{\"continue\": false, \"stopReason\": \"$MSG\"}"
            exit 0
        fi
    fi
fi

exit 0
