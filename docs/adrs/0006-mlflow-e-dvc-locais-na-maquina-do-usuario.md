<!--
================================================================================
LOG DE MANUTENÇÃO DE DOCUMENTAÇÃO
--------------------------------------------------------------------------------
Data       | Autor          | Descrição
--------------------------------------------------------------------------------
2026-10-25 | Eduardo Batista | Criação da ADR 0006 detalhando MLflow e DVC, ambos locais, na Etapa 3.
2026-10-27 | Eduardo Batista | Status para Implementada após a execução real do núcleo da Etapa 3 (pipeline_run_id 20261027T103457Z).
2026-10-27 | Eduardo Batista | Nota: o champion servido passou a ser `deps` das demos no dvc.yaml (ADR-0007), o que fecha a limitação do dvc repro com promoção e rollback.
================================================================================
-->

# ADR 0006: MLflow e DVC na Etapa 3, com versionamento apenas local

| Metadado | Detalhe |
| :--- | :--- |
| **Status** | **Implementada** (2026-10-27) |
| **Decisores** | Eduardo Batista |
| **Tags** | `[mlops]`, `[observability]`, `[data-versioning]`, `[licensing]` |
| **Impacto Sistêmico** | Alto (rastreio de experimentos e versionamento de dados/modelos da Etapa 3) |
| **ADRs Relacionadas** | Extends ADR-0001, ADR-0003 e ADR-0005 |

## Contexto

A Etapa 3 precisa centralizar a telemetria do pipeline (hoje em `logs/pipeline.jsonl`), registrar métricas de saúde e alertas de drift, e tornar reprodutível qual dado e qual modelo geraram cada resultado. O architecture review de 2026-10-25 (`docs/ai/reports/2026-10-25-architecture-review-src.md`) apontou a decisão de telemetria como Tipo 1 (custo alto de reversão).

O usuário decidiu adotar **MLflow** (rastreio de execuções, métricas e artefatos) e **DVC** (versionamento de dados e artefatos pesados), com o versionamento **somente na máquina do usuário, para demonstração**.

### Riscos Identificados e Problemas a Resolver:
1. **Licença do dataset**: o Home Credit Default Risk vem do Kaggle, cujas regras da competição não permitem redistribuir os dados. Um remoto de DVC (S3, Drive, SSH), mesmo privado, seria redistribuição.
2. **Credenciais**: remotos de DVC e servidores de MLflow exigem credenciais, e o projeto proíbe segredos em texto puro.
3. **Reprodutibilidade fora da máquina do autor**: quem clonar o repositório não terá o dado nem o histórico do DVC/MLflow.

---

## 2. Decisão de Arquitetura

- **MLflow** com *tracking store* em SQLite (`mlflow.db`) e *artifact store* em diretório local, ambos ignorados pelo git. O banco é necessário para o Model Registry da extensão de retreino e promoção (ver a spec da Etapa 3). Sem servidor remoto.
- **DVC** com remoto **local** (diretório na máquina do usuário, fora do repositório) ou sem remoto. Nada de `dvc push` para serviços externos.
- O `dvc.lock` (apenas hashes e metadados, sem dados) vai para o git; o dataset bruto entra como `deps` do estágio de preparo, sem `dvc add` (um `.dvc` em `data/raw/` cairia no `.gitignore`). Os datasets (`data/`), o cache do DVC, o `mlflow.db` e o `mlartifacts/` ficam fora do git.
- O modelo baseline (`models/`, ~800 KB) e os relatórios (`reports/`) **continuam no git**: o checklist da Etapa 3 (3.4) exige consolidá-los no repositório como evidência, e não contêm dados do Kaggle.

### Motivos do escopo local

| Motivo | Detalhe |
| :--- | :--- |
| **Licença Kaggle** | Evita qualquer redistribuição do dataset (ADR-0001). |
| **Zero segredos** | Sem remoto externo não há credencial a guardar nem vazar. |
| **Demo, não produção** | A Etapa 3 é uma demonstração; um servidor compartilhado e um remoto de nuvem adicionariam custo e operação sem benefício para o objetivo. |
| **Reversível** | O formato do DVC e do MLflow permite trocar para um remoto compartilhado depois, sem refazer o pipeline. |

### Alternativas descartadas por ora

| Critério | Local (eleita) | Remoto compartilhado |
| :--- | :--- | :--- |
| **Licença do dataset** | Sem redistribuição | Redistribui o dado |
| **Segredos e credenciais** | Nenhum | Necessários |
| **Esforço** | Baixo | Alto |
| **Quem reproduz** | Só quem tem o dataset do Kaggle | Qualquer pessoa com acesso ao remoto |

---

## 6. Matriz de Conformidade e Mitigação de Riscos

| Risco mapeado | Impacto | Mitigação |
| :--- | :---: | :--- |
| Dado versionado vazar para o git | Alto | `data/`, `mlruns/` e o cache do DVC no `.gitignore`; teste que falhe se arquivos de `data/` (fora os `.gitkeep`) aparecerem no índice do git |
| Remoto externo configurado por engano | Alto | Documentar o remoto local no README e não commitar `.dvc/config.local` |
| Terceiro não conseguir reproduzir | Médio | README explica que o dataset deve ser baixado do Kaggle e como rodar `dvc repro` |

---

## 7. Consequências e Resultados

### Positivas
- **Rastreabilidade**: cada execução liga dado, modelo, métricas e alertas.
- **Conformidade**: nenhum dado do Kaggle sai da máquina.

### Mitigações e Desafios Gerenciados
- **Reprodução limitada**: aceita, por ser uma demonstração; o caminho para um remoto compartilhado fica documentado.
- **Duas ferramentas novas**: a spec da Etapa 3 define o que cada uma cobre, para evitar sobreposição (por exemplo, métricas só no MLflow).

### Pontos em aberto (a fechar na spec da Etapa 3)
- Quais estágios entram no `dvc.yaml`.
- O que é logado no MLflow (métricas de saúde por lote, PSI, queda de AUC, alertas) e como `run_id` se relaciona com o run do MLflow.
- A política de alerta (`min_psi=PSI_MODERATE` e `performance.degraded`).

### Implementação (2026-10-27)
Os três pontos acima foram fechados na [spec da Etapa 3](../ai/specs/2026-10-25-etapa3-observabilidade-design.md) (§4.1 a §4.5).
- O `dvc.yaml` tem `prepare → train → contract_demo → drift_demo`. O `dvc.lock` é versionado, e o remoto local só existe em `.dvc/config.local`.
- O MLflow usa `mlflow.db` e `mlartifacts/`, fora do git. A tag `pipeline_run_id` liga cada run de estágio ao log JSONL anexado.
- Na execução real (`pipeline_run_id` `20261027T103457Z`), t0 deu `OK`, t1 `ATENCAO` e t2 `CRITICO`.
- Limitação conhecida: o champion servido (`models/champion/`) fica fora dos `outs`. Por isso o `dvc repro` não percebe uma promoção nem um rollback (Fase B, ADR-0007 ainda a criar).

> **Nota (2026-10-27, Fase B):** a limitação acima foi tratada na [ADR-0007](0007-retreino-registry-e-promocao-humana.md). O champion continua fora dos `outs`, mas os dois arquivos de `models/champion/` passaram a ser `deps` de `contract_demo` e `drift_demo`. Depois de uma promoção ou de um rollback, o `dvc repro` reexecuta as duas demos. Com `MONITORING_TRACKING=off` não há champion, e o `dvc repro` para nessas demos; nesse caso, use `make pipeline`. O texto acima fica como histórico.
