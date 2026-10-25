<!--
================================================================================
LOG DE MANUTENÇÃO DE DOCUMENTAÇÃO
--------------------------------------------------------------------------------
Data       | Autor          | Descrição
--------------------------------------------------------------------------------
2026-10-25 | Eduardo Batista | Criação da ADR 0006 detalhando MLflow e DVC, ambos locais, na Etapa 3.
================================================================================
-->

# ADR 0006: MLflow e DVC na Etapa 3, com versionamento apenas local

| Metadado | Detalhe |
| :--- | :--- |
| **Status** | **Proposta** (2026-10-25) |
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

- **MLflow** com *tracking store* e *artifact store* em disco local (diretório ignorado pelo git). Sem servidor remoto.
- **DVC** com remoto **local** (diretório na máquina do usuário, fora do repositório) ou sem remoto. Nada de `dvc push` para serviços externos.
- Os arquivos `.dvc` e `dvc.lock` (apenas hashes e metadados, sem dados) podem ir para o git; os datasets (`data/`), o cache do DVC e o `mlruns/` ficam fora do git.
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
