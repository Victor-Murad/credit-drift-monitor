# ADR-0002 — Data Contracts com Pandera (e não Great Expectations)

- **Status:** aceita
- **Etapa:** 1 (checklist 1.2 e 1.4; ambiguidade #5 do checklist)

## Contexto

O PDF aceita **Great Expectations ou Pandera** e não dá critério de desempate. Por isso usamos o material da própria disciplina da Etapa 1 (*Validação de Dados e Bibliotecas de Qualidade*):

- A **Aula 5 — Validação de Schema e Contratos de Dados**, que trata exatamente do tema da Etapa 1, implementa o contrato com **Pandera** (`DataFrameModel`, `pa.Field`, `validate(lazy=True)`, `SchemaErrors`). A dica prática da aula é: *"comece com Pandera para validação de DataFrames"*.
- A **Aula 7 — Frameworks de Qualidade de Dados** apresenta os dois. O Great Expectations (API v1.x) exige contexto, data source, asset, batch definition, suite e validation definition. O Pandera é declarado como uma classe Python.

## Decisão

Usar **Pandera** (`pandera.pandas.DataFrameModel`) como camada executável do contrato. A camada organizacional (dono, termos de uso, classificação LGPD dos campos) segue o formato **Data Contract Specification** (YAML), também da Aula 5, em `contracts/credit_application.yaml`.

## Justificativa técnica

| Critério | Pandera | Great Expectations |
|---|---|---|
| Contrato como código (versionado e revisado em PR) | Classe Python tipada | Suites em JSON/YAML ou código mais verboso |
| Testável com pytest | Direto (`tests/test_contract.py`) | Exige montar o contexto |
| Relatar todas as violações de uma vez | `lazy=True` → `SchemaErrors.failure_cases` | Sim |
| Bloquear colunas não previstas (ex.: CPF vazando) | `strict=True` | Possível, mas com mais configuração |
| Peso e estabilidade da API | Leve | Pesado; a API mudou muito da 0.x para a 1.x |
| Relatório HTML | Não tem nativo → geramos o nosso (`src/validation/report.py`) | Data Docs nativos |

O único ponto em que o Great Expectations ganha é o relatório HTML nativo (Data Docs). Isso foi resolvido com um relatório HTML próprio gerado a partir dos `failure_cases` do Pandera.

## Consequências

- O contrato tem duas camadas: código Pandera e YAML DCS. `tests/test_contract_spec.py` impede que elas divirjam.
- Política **fail-closed**: qualquer violação bloqueia o lote inteiro (`DataContractViolation`), manda o arquivo para quarentena e gera relatório e log estruturado.
