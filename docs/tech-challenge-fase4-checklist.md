# Tech Challenge — Checklist Operacional

### Fase 4 · Observabilidade Contínua, Detecção de Drift, Qualidade de Dados e Governança em Produção

> **Contexto do desafio:** fintech de crédito com modelo de Credit Scoring em produção; suspeita de degradação silenciosa por mudanças econômicas (inflação, novos perfis de cliente). Objetivo: construir a camada de **Sustentação e Confiabilidade** — data contracts, simulação de Data/Concept Drift, observabilidade e governança/LGPD.
> **Peso da atividade:** 90% da nota de todas as disciplinas da fase.

---

## Estado da validação (2026-10-25)

Validado contra o repositório e reproduzido ponta a ponta com o dataset real (2026-10-25): `pytest` com 74 testes passando, cobertura 95% (`make test-cov`, piso 80%), `ruff` e `lint-imports` limpos, baseline e relatórios Evidently versionados. **Etapas 1 e 2 implementadas e com as pendências de código/documentação fechadas; Etapas 3 e 4 pendentes** (README ainda marca "Em construção"). Situação dos pontos levantados em 2026-10-21:

- [X] **Instalação limpa (1.2) e reprodução ponta a ponta (4.3):** `uv sync --locked` + `ruff` + `lint-imports` + `pytest` verificados numa cópia nova do projeto (sem `.venv`), e `make data && make train && make etapa1 && make etapa2` executados com o `application_train.csv` real (307.511 linhas, 21 rejeitadas, 215.243 / 92.247 no split, idênticos ao README). O retreino nesta máquina deu ROC-AUC 0,7491 (antes 0,7494): diferença numérica entre ambientes, sem mudar conclusões.
- [X] **Concept drift (2.3):** limitação documentada no README, no `docs/etapa-2-...md` e no sumário da demo (o teste sobre `target` mede label shift). A queda de ROC-AUC (`src/drift/performance.py`) entra no sumário JSON/MD (`performance`), pronta para a Etapa 3.
- [X] **PSI:** calculado em toda coluna (`src/drift/psi.py`) e filtro de efeito opcional `min_psi` / `--min-psi` ([ADR-0005](adrs/0005-psi-e-queda-de-auc-como-sinais-de-drift.md)). A demo usa `min_psi=0,10`, validado com dados reais: t0 deixa de ter falso alerta (`annuity_amount`, PSI 0,00) e o teste sobre `target` em t2 deixa de alertar (PSI 0,02), enquanto a queda de AUC (0,181) acusa a degradação. A Etapa 3 deve usar o mesmo critério ao contar alertas.
- [X] **Cobertura:** 95% (antes 67,89%); `fail_under` subiu para 80.
- [X] **Dependência:** `pyproject.toml` com `evidently>=0.7`; `uv.lock` atualizado.
- [X] **CHANGELOG.md** com as entradas das Etapas 1 e 2.
- [X] **`reports/drift/` regenerado** com `make etapa2` (dados reais): artefatos versionados trazem `psi` e `performance`.

---

## Nota sobre a estrutura deste checklist

Este documento está organizado em **4 Etapas sequenciais e auto-contidas**, espelhando as 4 Etapas do próprio PDF (cada uma ligada a uma disciplina da fase). "Auto-contida" aqui significa que cada Etapa carrega tudo que é preciso para executá-la — inclusive as dependências de ambiente específicas dela — sem depender de consultar uma seção genérica de "setup" em outro lugar do documento. "Sequencial" significa que a Etapa N assume que os artefatos da Etapa N−1 já existem (modelo baseline, dataset de referência, relatório de drift etc.); não pule etapas.

Cada Etapa termina com um **checklist de saída** que replica literalmente o entregável exigido pelo PDF para aquela etapa — use-o como critério de "pronto para avançar".

## ⚠️ Ambiguidades identificadas no PDF (sinalizar ao grupo/orientação)

1. **Individual vs. grupo:** o PDF diz "a princípio, deve ser desenvolvida em grupo" — não deixa explícito o que fazer em caso de desenvolvimento individual. Confirmar com a coordenação.
2. **"Análise causal do problema":** citada no Contexto ("garantir que todo o processo esteja documentado... incluindo uma análise causal do problema") e no entregável da Etapa 4 ("Documentação completa de Governança **e Causalidade** no README"), mas **não há tarefa específica** para ela nas 4 Etapas descritas. Neste checklist ela foi alocada na Etapa 4. Atenção: **"mitigação de vieses"** (critério de Governança, 15%) é um requisito **distinto** da análise causal; os dois precisam estar no README.
3. **Prazo de entrega:** o PDF apenas diz "é importante atentar-se ao prazo de entrega", sem informar a data. Confirmar prazo oficial com a coordenação/plataforma da fase.
4. **Formato e hospedagem do vídeo STAR:** não especificados (YouTube não-listado, Drive, Loom etc.). Definir com o grupo antes da gravação.
5. **Great Expectations vs. Pandera** e **MLflow vs. Datadog/Cloud/Open-Source**: o PDF deixa a escolha livre, sem critério de desempate — decisão de arquitetura do grupo, não do enunciado.
6. **Prometheus-client** é opcional apenas "se quiserem integrar com a Fase 3" — pressupõe que o grupo tenha (ou reaproveite) um projeto de Fase 3 existente; se não houver, o item simplesmente não se aplica.

---

## Etapa 1 — Fundação, Ambiente e Validação de Dados

**Disciplina do PDF:** Validação de Dados e Bibliotecas de Qualidade · **Foco:** garantir que lixo não entre no modelo · **Peso relacionado:** Validação de Dados — 25%

### 1.1 Alinhamento e definição do problema

- [X] Ler o enunciado completo do Tech Challenge Fase 4 e revisar os pesos dos critérios de avaliação (Validação 25%, Drift 25%, Observabilidade 20%, Governança 15%, Vídeo 15%)
- [X] Confirmar com a coordenação: prazo de entrega, formato de submissão, e se o trabalho é individual ou em grupo (ambiguidade #1)
- [X] Combinar divisão de responsabilidades entre integrantes do grupo, se aplicável
- [X] Selecionar o dataset: classificação binária financeira (crédito/risco/precificação) com **no mínimo 5.000 amostras**
  - Opções sugeridas no PDF: German Credit Data (Kaggle/UCI), Home Credit Default Risk (Kaggle), Lending Club Loan Data
- [X] Validar que o dataset comporta um cenário plausível de degradação econômica (inflação, mudança de perfil de cliente)
- [X] Definir o algoritmo do modelo base (Scikit-Learn e/ou XGBoost)
- [X] Definir, desde já, **pelo menos 2 variáveis** que serão manipuladas para simular Drift na Etapa 2 (ex.: renda, taxa de desemprego)
- [X] Registrar dúvidas de enunciado no Discord oficial

### 1.2 Setup do projeto e do ambiente

- [X] Criar o repositório GitHub do projeto
- [X] Definir estrutura de pastas (ex.: `data/reference/`, `data/production/`, `src/`, `notebooks/`, `reports/`, `docs/`)
- [X] Configurar `.gitignore` (dados brutos sensíveis, artefatos de modelo, ambientes virtuais, caches)
- [X] Definir e comunicar ao grupo a convenção de commits semânticos (ex.: Conventional Commits) — requisito obrigatório: "Histórico de commits semântico e organizado"
- [X] Criar `README.md` inicial com título, contexto do problema e sumário
- [X] Configurar gerenciador de dependências (`pyproject.toml` ou `requirements.txt`, via Poetry, uv ou pip)
- [X] Adicionar **Scikit-Learn** e/ou **XGBoost** (modelo base de risco de crédito)
- [X] Adicionar **Great Expectations** ou **Pandera** (Data Contracts) — decidir qual das duas (ambiguidade #5) e documentar a escolha
- [X] Gerar e commitar o lock file de dependências
- [X] Validar instalação limpa em ambiente novo (venv/container do zero)
- [X] Fixar seeds globais de aleatoriedade para reprodutibilidade dos experimentos

### 1.3 Dataset de Referência e modelo baseline

- [X] Carregar o dataset escolhido e isolar o **Dataset de Referência** (usado no treino) — manter separação clara e documentada do futuro dataset de produção/análise (Boa Prática Obrigatória)
- [X] Treinar o **classificador base** de risco de crédito com o Dataset de Referência
- [X] Persistir o modelo baseline treinado (ex.: `.pkl`/`.joblib`)

### 1.4 Data Contract e demonstração de bloqueio

- [X] Criar o **contrato de dados** (Great Expectations ou Pandera) com no mínimo **3 regras rígidas de qualidade de dados**, por exemplo:
  - `idade > 18` (o contrato implementado aceita `age_years >= 18`; divergência menor de borda em relação ao exemplo do PDF)
  - renda não pode ser nula
  - sem duplicatas
  - range válido de idade / limites de renda
- [X] Implementar o pipeline de validação de dados que executa o contrato **antes** da ingestão no treino/inferência
- [X] Criar um lote de dados propositalmente inválido (nulos, idade fora do range, duplicatas)
- [X] Rodar o contrato contra esse lote e **documentar o bloqueio da ingestão** (log ou output do contrato falhando)

### ✅ Checklist de saída da Etapa 1

- [X] Script de validação de dados operante, versionado no repositório
- [X] Modelo baseline treinado e persistido
- [X] Contrato de dados com ≥ 3 regras, com evidência de bloqueio de lote inválido

*(Entregável Etapa 1 do PDF: "Script de validação de dados operante e modelo baseline treinado")*

---

## Etapa 2 — Simulação e Detecção de Drift

**Disciplina do PDF:** Data Drift · **Foco:** diagnosticar a degradação do modelo ao longo do tempo · **Peso relacionado:** Detecção de Drift — 25%

> Pré-requisito: Etapa 1 concluída (modelo baseline + Dataset de Referência prontos).

### 2.1 Ambiente adicional necessário

- [X] Adicionar **Evidently AI** às dependências do projeto (a tarefa da Etapa 2 e o critério de Detecção de Drift citam Evidently; Alibi Detect só aparece na lista de bibliotecas como alternativa)
- [X] Atualizar e commitar o lock file

### 2.2 Construção do Dataset de Produção

- [X] Criar o "Dataset de Produção" a partir do Dataset de Referência, simulando a passagem do tempo
- [X] Alterar intencionalmente a distribuição das **pelo menos 2 variáveis** definidas na Etapa 1.1 (ex.: aumentar artificialmente renda e/ou taxa de desemprego dos clientes)
- [X] Gerar as predições do modelo baseline (Etapa 1) para o Dataset de Produção
- [X] Armazenar os dois datasets (Referência e Produção) de forma clara e separada (`data/reference/` e `data/production/`). Os arquivos **não são versionados** (licença do Kaggle); o que vai ao repositório são os scripts de geração (`make data`, `src/drift/simulate.py`), a estrutura de pastas e os relatórios

### 2.3 Detecção estatística e relatório de Drift

- [X] Utilizar **Evidently AI** para comparar Dataset de Referência vs. Dataset de Produção
- [X] Rodar testes estatísticos de mudança de distribuição: **PSI (Population Stability Index)** e/ou **Kolmogorov-Smirnov**
- [X] Identificar explicitamente quais features sofreram Drift
- [X] Gerar o relatório estatístico evidenciando a mudança de distribuição (Boa Prática Obrigatória do PDF)
- [X] Gerar o **Relatório de Drift em HTML** (saída nativa do Evidently) ou painel customizado, apontando claramente as variáveis degradadas
- [X] Salvar o(s) relatório(s) HTML no repositório (ex.: `reports/drift_report.html`)

### ✅ Checklist de saída da Etapa 2

- [X] Dataset de Produção com drift induzido e documentado
- [X] Testes estatísticos (PSI e/ou KS) executados e resultados registrados
- [X] Relatório de Drift em HTML salvo no repositório, apontando claramente as variáveis degradadas

*(Entregável Etapa 2 do PDF: "Relatório de Drift (HTML do Evidently ou painel customizado) apontando claramente as variáveis degradadas")*

---

## Etapa 3 — Observabilidade de Pipelines e Modelos

**Disciplina do PDF:** Monitoramento de Pipelines e Infraestrutura · **Foco:** visibilidade operacional dos pipelines, da infraestrutura e dos modelos em produção · **Peso relacionado:** Observabilidade e Logs — 20%

> Pré-requisito: Etapa 2 concluída (relatório de Drift disponível para ser exposto/monitorado).

### 3.1 Decisão de arquitetura e ambiente adicional

- [ ] Escolher o caminho de rastreamento/monitoramento (ambiguidade #5; o PDF diz "pode optar por" MLflow **ou** dashboard, sem proibir combinar):
  - **Opção A — MLflow:** rastrear ciclo de vida do modelo e experimentos, **ou**
  - **Opção B — Dashboard interativo:** exportar métricas de infraestrutura/pipeline (Datadog, solução Cloud nativa ou alternativa Open-Source)
- [ ] Adicionar a biblioteca/ferramenta escolhida às dependências do projeto
- [ ] Adicionar **Prometheus-client** apenas se houver integração real com um projeto de Fase 3 (opcional — ambiguidade #6)

### 3.2 Centralização de logs e métricas

- [ ] Centralizar logs de execução do pipeline (sucesso/falha na ingestão e transformação de dados)
- [ ] Integrar, no mesmo local, os resultados dos testes de Drift gerados na Etapa 2.3

### 3.3 Definição e documentação de métricas de saúde

- [ ] Definir e documentar as **métricas de saúde operacionais**, no mínimo:
  - status de execução das tarefas do pipeline
  - latência
  - volume de dados processados
  - quantidade de alertas de falha ou drift
  - métricas de saúde do **modelo** (ex.: ROC-AUC e queda de performance), exigidas pelo critério de Observabilidade ("métricas de saúde do modelo e alertas de drift")
- [ ] Implementar a coleta/exposição dessas métricas na ferramenta escolhida (MLflow ou dashboard)
- [ ] Gerar o dashboard de monitoramento (ou ambiente de rastreamento) final evidenciando: saúde do pipeline de dados, estabilidade da infraestrutura e alertas de degradação do modelo

### 3.4 Consolidação e versionamento de evidências

- [ ] Consolidar no repositório todos os artefatos gerados até aqui: script de validação, modelo baseline, datasets (ou scripts de geração), relatório(s) de Drift em HTML, dashboard/ambiente de rastreamento
- [ ] Garantir histórico de commits semântico e organizado cobrindo as Etapas 1 a 3
- [ ] Se optou por MLflow: garantir que os experimentos/runs estejam registrados e acessíveis (ex.: `mlflow ui` funcional ou export dos resultados)

### ✅ Checklist de saída da Etapa 3

- [ ] Ferramenta de rastreamento/monitoramento escolhida e em funcionamento (MLflow ou dashboard)
- [ ] Logs de pipeline e resultados de Drift centralizados em um único local
- [ ] Métricas de saúde documentadas e visíveis no dashboard/ambiente de rastreamento

*(Entregável Etapa 3 do PDF: "Dashboard de monitoramento ou ambiente de rastreamento (ex.: MLflow / Datadog) evidenciando a saúde completa do pipeline, a estabilidade da infraestrutura e os alertas de degradação do modelo")*

---

## Etapa 4 — Governança, Vídeo STAR e Entrega

**Disciplinas do PDF:** Privacy By Design e By Default · **Foco:** documentação arquitetural, compliance e prescrição · **Peso relacionado:** Governança (LGPD) — 15% · Vídeo STAR — 15%

> Pré-requisito: Etapas 1 a 3 concluídas (é preciso ter relatório de Drift e dashboard prontos antes de documentar e gravar o vídeo).

### 4.1 Documentação de Governança e LGPD (no README.md)

- [ ] Escrever o **plano de proteção de dados (LGPD)**, contendo:
  - como dados sensíveis (PII) foram tratados no projeto
  - qual a base legal para a decisão de crédito
  - o plano de retenção de dados
- [ ] Incluir a **mitigação de vieses** na documentação de governança (critério de avaliação distinto da análise causal; ex.: atributos protegidos como gênero e idade no modelo)
- [ ] Incluir a **análise causal do problema** de degradação do modelo (ambiguidade #2)
- [ ] Documentar as ferramentas escolhidas e a justificativa técnica (Great Expectations vs. Pandera; MLflow vs. Datadog/outra alternativa)
- [ ] Revisar se o `README.md` cobre, de forma consolidada, os 5 itens obrigatórios do repositório:
  - [ ] Pipeline de validação de dados (data contracts) configurado
  - [ ] Scripts de simulação e detecção estatística de Drift
  - [ ] Dashboards ou relatórios HTML comprovando o monitoramento
  - [ ] Documentação consolidada de Governança
  - [ ] Histórico de commits semântico e organizado

### 4.2 Vídeo STAR (até 5 minutos)

- [ ] Roteirizar o vídeo seguindo estritamente o método STAR definido no PDF:
  - **Situation:** o risco da degradação silenciosa de modelos de crédito no mundo real
  - **Task:** requisitos técnicos da fase (interceptar dados ruins, detectar drift, garantir compliance)
  - **Action:** ferramentas escolhidas de validação e observabilidade, simulação do drift e plano de governança
  - **Result:** demonstração prática do relatório apontando o drift, alertas gerados e lições aprendidas
- [ ] Gravar demonstração prática em tela mostrando o relatório de Drift (HTML) e o dashboard/alertas gerados
- [ ] Editar o vídeo garantindo duração **≤ 5 minutos**
- [ ] Publicar o vídeo em uma plataforma acessível ao avaliador (ambiguidade #4)
- [ ] Adicionar o link do vídeo no `README.md`

### 4.3 Validação final (todas as etapas)

- [ ] Conferir os **Requisitos Obrigatórios do repositório** (Etapas 1–4) item a item
- [ ] Conferir as **Boas Práticas Obrigatórias**: separação Referência/Produção, ≥ 3 regras de qualidade, relatório PSI/KS, documentação de compliance LGPD
- [ ] Confirmar que as bibliotecas obrigatórias estão de fato em uso e referenciadas no código
- [ ] Priorizar revisão nos itens de maior peso: Validação de Dados (25%) e Detecção de Drift (25%) somam metade da nota
- [X] Executar o projeto do zero em ambiente novo (fim a fim) para confirmar reprodutibilidade (Etapas 1–2 verificadas em 2026-10-25; repetir ao fechar as Etapas 3–4)

### 4.4 Entrega

- [ ] Conferir se o repositório GitHub está com a visibilidade correta para os avaliadores
- [ ] Conferir se o link do vídeo está funcional e com permissão de visualização adequada
- [ ] Submeter o link do repositório e do vídeo conforme instruções da coordenação/plataforma
- [ ] Confirmar que a submissão ocorreu dentro do prazo (atividade obrigatória — vale 90% da nota de todas as disciplinas da fase)

### ✅ Checklist de saída da Etapa 4

- [ ] Documentação completa de Governança e Causalidade no README
- [ ] Link do vídeo STAR (≤ 5 min) no README
- [ ] Repositório e vídeo submetidos dentro do prazo

*(Entregável Etapa 4 do PDF: "Documentação completa de Governança e Causalidade no README + Link do Vídeo")*

---

## Bônus opcional (fora da sequência obrigatória)

- [ ] Integrar métricas com **Prometheus-client**, apenas se houver um projeto de Fase 3 real para integrar
- [ ] Explorar **Alibi Detect** como alternativa/complemento ao Evidently AI
- [ ] Expandir a observabilidade com **Datadog** ou solução Cloud nativa, além (ou no lugar) da opção MLflow

---

## Observações de execução

**Dependências críticas entre etapas:**
Etapa 1 (baseline + contrato de dados) → Etapa 2 (drift depende do baseline e do Dataset de Referência) → Etapa 3 (observabilidade depende do relatório de Drift já existir) → Etapa 4 (documentação e vídeo dependem de todos os artefatos anteriores). Gravar o vídeo antes de concluir as Etapas 2 e 3 é o erro de sequenciamento mais comum.

**Riscos comuns:**

- Confundir Dataset de Referência com Dataset de Produção, ou não versioná-los separadamente.
- Implementar regras de qualidade "frouxas" que não bloqueiam o lote de teste sujo — o PDF exige demonstrar o contrato **falhando**, não apenas existindo.
- Gastar tempo dobrado integrando MLflow e Datadog por insegurança na decisão. O PDF permite as duas opções, mas o tempo é melhor investido em uma só bem feita.
- Documentação de LGPD genérica, sem amarrar aos dados reais do projeto.
- Estourar os 5 minutos do vídeo ou não seguir a estrutura STAR explicitamente.

**Sugestão de ordem de ataque:**
Ataque primeiro as Etapas 1 e 2 (Validação de Dados 25% + Detecção de Drift 25% = 50% da nota) antes de investir tempo em observabilidade e documentação. Só feche a decisão de arquitetura da Etapa 3 depois de ter o relatório de drift pronto. Deixe a gravação do vídeo por último, depois da validação final, para que a demonstração mostre o estado real e finalizado do projeto.
