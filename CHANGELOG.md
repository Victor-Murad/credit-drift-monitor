# Changelog

Todas as mudanças notáveis deste projeto são documentadas aqui.

O formato segue [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/) e o versionamento segue [SemVer 2.0.0](https://semver.org/lang/pt-BR/).

## [Unreleased]
### Adicionado
- Data contract Pandera com 11 regras rígidas, gate fail-closed com quarentena e relatórios HTML/JSON (Etapa 1).
- Dataset de Referência e lote de Produção t0 separados, modelo baseline XGBoost persistido e demonstração do bloqueio de lote inválido (Etapa 1).
- Simulação de Data/Concept Drift e detecção com Evidently, com relatórios HTML/JSON e sumário de impacto no modelo (Etapa 2).
- PSI por coluna (`src/drift/psi.py`) e filtro opcional de efeito `min_psi` / `--min-psi` na detecção de drift ([ADR-0005](docs/adrs/0005-psi-e-queda-de-auc-como-sinais-de-drift.md)).
- Queda de ROC-AUC vs. baseline como sinal de concept drift (`src/drift/performance.py`), exposta no sumário da demo (`performance`).
- Demo de drift usa `min_psi=0,10` (validado com dados reais): t0 sem falso alerta; sumário passa a incluir PSI por coluna e a degradação de AUC.
- Testes de integração do pipeline ponta a ponta em sandbox (`tests/integration/`) e testes de download, PSI e CLIs.

### Alterado
- Piso do Evidently em `pyproject.toml` corrigido para `>=0.7` (a API usada, `from evidently import Report`, não existe antes).
- Meta de cobertura do repositório (`fail_under`) elevada de 68 para 80; cobertura atual ~95%.
- Sumário de drift passa a rotular o teste sobre `target` como *label shift* e a explicar a limitação (não mede mudança em P(y|X)).
- Links para ADRs na documentação das Etapas 1 e 2 corrigidos para `docs/adrs/`.

- Artefatos de `models/` e `reports/` regenerados com o dataset real (reprodução ponta a ponta em 2026-10-25). O retreino nesta máquina (xgboost 3.4.1) deu ROC-AUC 0,7491 (antes 0,7494): pequenas diferenças numéricas entre ambientes, sem mudar as conclusões.
