# Como contribuir

O enunciado exige um **"histórico de commits semântico e organizado"**. Este guia é a convenção combinada pelo grupo.

## Convenção de commits — Conventional Commits em PT-BR

Formato:

```
<tipo>(<escopo opcional>): <descrição no imperativo, minúscula, sem ponto final>

<corpo opcional: o porquê da mudança>
```

| Tipo | Quando usar | Exemplo |
|---|---|---|
| `feat` | nova funcionalidade | `feat(contract): adicionar regra de idade minima` |
| `fix` | correção de bug | `fix(prepare): tratar sentinela de DAYS_EMPLOYED` |
| `docs` | só documentação | `docs(adr): registrar escolha do Pandera` |
| `test` | só testes | `test(gate): cobrir quarentena de lote invalido` |
| `refactor` | mudança sem alterar comportamento | `refactor(train): extrair calculo do KS` |
| `chore` | build, dependências, configuração | `chore(deps): adicionar evidently` |
| `ci` | pipelines de CI | `ci: rodar pytest no GitHub Actions` |
| `perf` | desempenho | `perf(drift): calcular PSI vetorizado` |

Escopos usados no projeto: `data`, `contract`, `model`, `drift`, `monitoring`, `governance`, `deps`, `adr`, `readme`.

Regras:

- Um commit, uma intenção. Não misture refatoração com feature.
- A descrição completa a frase "este commit vai…" (`adicionar`, `corrigir`, `remover`).
- Mudança incompatível no contrato de dados: use `!` e explique no corpo, por exemplo `feat(contract)!: renomear income_total`. Isso também exige subir `CONTRACT_VERSION`.
- **Nunca** commitar dados do Kaggle, token do Kaggle (`access_token`/`kaggle.json`), `.env` ou qualquer dado pessoal.

## Branches

- `main` é protegida e só recebe merge via Pull Request.
- Uma branch por etapa ou tarefa: `etapa-1/validacao-dados`, `etapa-2/drift`, `etapa-3/observabilidade`, `etapa-4/governanca`.

## Antes de abrir o PR

```bash
make lint
make test
```

Se você mudou o contrato (`src/validation/contract.py`), atualize também `contracts/credit_application.yaml`. O teste `tests/test_contract_spec.py` falha se os dois divergirem.
