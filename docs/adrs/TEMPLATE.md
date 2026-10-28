<!--
================================================================================
LOG DE MANUTENÇÃO DE DOCUMENTAÇÃO
--------------------------------------------------------------------------------
Data       | Autor          | Descrição
--------------------------------------------------------------------------------
YYYY-MM-DD | Nome do Autor  | Criação da ADR XXXX detalhando [assunto principal].
================================================================================
-->

# ADR [NUMERO_4_DIGITOS]: [Título Declarativo e Abrangente da Decisão]

| Metadado | Detalhe |
| :--- | :--- |
| **Status** | **[Proposta | Aprovada | Implementada | Substituída por ADR-XXXX | Rejeitada]** (YYYY-MM-DD) |
| **Decisores** | [Nome do Autor, Tech Lead, SecOps, Arquitetura] |
| **Tags** | `[ex: security]`, `[database]`, `[api-gateway]`, `[ai-agents]` |
| **Impacto Sistêmico** | [Alto / Médio / Baixo] ([Módulos/Serviços Impactados]) |
| **ADRs Relacionadas** | [Ex: Supersedes ADR-XXXX; Extends ADR-YYYY / N/A] |

## Contexto
[Descrição detalhada do cenário técnico e de negócio, atores impactados, estado atual do sistema e necessidades a serem atendidas.]

### Riscos Identificados e Problemas a Resolver:
1. **[Risco 1 / Problema 1]**: [Detalhamento do impacto operacional, técnico, de segurança ou de conformidade].
2. **[Risco 2 / Problema 2]**: [Detalhamento do impacto].
3. **[Risco 3 / Problema 3]**: [Detalhamento do impacto].

---

## 1. Fundamentação Jurídica & Fontes Normativas Diretas (Se aplicável)

| Diploma Legal / Normativo | Artigos / Dispositivos Principais | Aplicação Prática no Sistema |
| :--- | :--- | :--- |
| **[Norma / Legislação 1]** | [Artigos / Cláusulas aplicáveis] | [Impacto e comportamento exigido no sistema] |
| **[Norma / Legislação 2]** | [Artigos / Cláusulas aplicáveis] | [Impacto e comportamento exigido no sistema] |

---

## 2. Decisão de Arquitetura

[Explicação detalhada da solução eleita, premissas de engenharia e justificativa comparativa frente a outras alternativas descartadas.]

### Matriz de Avaliação de Alternativas (Trade-off Matrix)

| Critério de Avaliação | Opção A: [Solução A] | Opção B: [Solução B (Eleita)] | Opção C: [Solução C] |
| :--- | :--- | :--- | :--- |
| **Custo / Esforço** | [Alto / Médio / Baixo] | [Alto / Médio / Baixo] | [Alto / Médio / Baixo] |
| **Latência / Performance** | [Alto / Médio / Baixo] | [Alto / Médio / Baixo] | [Alto / Médio / Baixo] |
| **Segurança & Privacidade** | [Alto / Médio / Baixo] | [Alto / Médio / Baixo] | [Alto / Médio / Baixo] |
| **Complexidade Operacional** | [Alto / Médio / Baixo] | [Alto / Médio / Baixo] | [Alto / Médio / Baixo] |
| **Curva de Aprendizado** | [Rápida / Moderada / Longa] | [Rápida / Moderada / Longa] | [Rápida / Moderada / Longa] |

```text
[Diagrama de Arquitetura em ASCII Art ou Mermaid modelando o fluxo e fronteiras de confiança]
```

---

## 3. Matriz de Implementação Técnica (*IN-CODE*)

### 3.1 Camada de Backend / API Gateway
- **[Componente / Módulo 1]**: [Especificação de endpoints, contratos de validação, regras de negócio e middleware].
- **[Componente / Módulo 2]**: [Especificação de segurança, resolução server-side de identidade e logs de auditoria].

### 3.2 Camada de Frontend / Client
- **[Componente / Módulo 1]**: [Especificação de interface, reatividade, gerenciamento de estado e bloqueio de idempotência].
- **[Componente / Módulo 2]**: [Especificação de proteção local de dados, validações e acessibilidade].

### 3.3 Camada de Persistência e Banco de Dados (ANSI SQL)
```sql
-- DDL físico executável com constraints de integridade, índices e imutabilidade
CREATE TABLE ExemploTabelaDecisao (
    idExemplo BIGINT IDENTITY(1,1) NOT NULL,
    codigoEntidade VARCHAR(64) NOT NULL,
    hashIntegridade CHAR(64) NOT NULL,
    dataCriacaoUtc DATETIME2(7) NOT NULL,
    CONSTRAINT PK_ExemploTabela PRIMARY KEY (idExemplo)
);

CREATE NONCLUSTERED INDEX IX_Exemplo_Entidade 
ON ExemploTabelaDecisao (codigoEntidade);
```

---

## 4. Matriz de Governança & Ações Institucionais (*OFF-CODE*) (Se aplicável)

| Ref | Ação Institucional Mandatória | Área Responsável | Base Legal / Normativa | Entregável / Evidência |
| :--- | :--- | :--- | :--- | :--- |
| **OFF-01** | [Descrição da ação obrigatória fora do código] | [Jurídico / RH / Infra / DPO] | [Artigo / Resolução] | [Documento / Parecer / Configuração] |
| **OFF-02** | [Descrição da ação obrigatória fora do código] | [Área Responsável] | [Artigo / Resolução] | [Documento / Parecer / Configuração] |

---

## 5. Prazos Legais de Guarda e Políticas de Retenção (Se aplicável)

| Categoria Documental / Dado | Prazo de Guarda | Fundamento Legal | Política no Storage / Banco |
| :--- | :--- | :--- | :--- |
| **[Categoria 1]** | [Ex: 5 anos] | [Dispositivo legal] | [Retenção em banco / política de expurgo] |
| **[Categoria 2]** | [Ex: 20 anos] | [Dispositivo legal] | [Object Lock no Storage / WORM] |

---

## 6. Matriz de Conformidade e Mitigação de Riscos

| Risco Mapeado no Contexto | Impacto Potencial | Mitigação Arquitetural e Técnica Adotada |
| :--- | :--- | :--- |
| **[Risco 1]** | [Alto / Médio / Baixo] | [Contramedida técnica implementada no código ou banco] |
| **[Risco 2]** | [Alto / Médio / Baixo] | [Contramedida técnica implementada] |

---

## 7. Consequências e Resultados

### Positivas
- **[Benefício 1]**: [Ganho de performance, conformidade, segurança ou escalabilidade].
- **[Benefício 2]**: [Simplificação operacional ou alinhamento arquitetural].

### Mitigações e Desafios Gerenciados
- **[Desafio / Trade-off 1]**: [Complexidade adicional aceita e como ela é gerenciada].
- **[Desafio / Trade-off 2]**: [Impacto de performance ou armazenamento mitigado].
