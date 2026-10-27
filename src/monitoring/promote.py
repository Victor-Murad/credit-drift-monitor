"""Promoção e rollback humanos do champion, com auditoria (spec da Etapa 3, §4.7 e §4.8).

    uv run python -m src.monitoring.promote [--version N]
    uv run python -m src.monitoring.promote --override --reason TEXTO
    uv run python -m src.monitoring.promote rollback [--reason TEXTO]

Comando humano separado: o retreino nunca promove. Ordem da promoção (§4.7):

1. Reavaliar a regra (`decide_promotion`) com as AUCs gravadas como MÉTRICAS no run de
   origem da versão (padrão: a do alias `challenger`), não com as tags da versão, que
   são editáveis. A tag `model_sha256` da versão tem de ser igual à do run de origem, e
   o challenger tem de ter sido comparado com o champion ATUAL (tag `champion_version`
   do run; senão `champion_mismatch`). Inelegível: recusa (código 2) sem tocar em nada,
   a menos de `--override --reason`, que fica no run de auditoria.
2. Baixar o bundle da versão para um temporário em `models/champion/` e conferir o
   sha256 contra a tag `model_sha256` (`registry.staged_champion`). Integridade é falha
   dura: nem o `--override` a dispensa.
3. e 4. Mover `previous` ← champion atual e `champion` ← versão (`promote_alias`).
5. `os.replace` atômico para `models/champion/` (`StagedChampion.install`).
6. Reconferir alias E arquivo (`served_state_problems`) e gravar o estado no run de
   auditoria (`status=CONSISTENT|INCONSISTENT`); divergência encerra com código 1.

Falha nos passos 3 a 5: tenta desfazer o que for reversível (o `champion` volta, e só
então o `previous`; se o arquivo divergir, ele é reexportado da versão anterior), grava
`status=INCONSISTENT` no run de auditoria (com o passo, o erro, o que foi desfeito e o
que ainda diverge) e encerra com código 1. Se a reversão não fechar, o Registry e o
arquivo servido divergem e o `load_bundle(expected_sha256=...)` do `drift_demo` e do
`monitor` recusa o modelo até a correção (fail-closed); o `rollback` é o caminho de
volta, porque o `previous` só é devolvido depois do `champion`.

O rollback segue a mesma ordem: `champion` ← `previous` (que é consumido), arquivo
reexportado do artefato da versão antiga, reconferência e auditoria. Sem `previous`,
erro claro sem alterar nada. `models/baseline_model.joblib` (o `out` do `train`) nunca
é tocado: o champion servido vive em `models/champion/`.

Códigos de saída: 0 (troca feita e reconferida), 2 (recusa da política: challenger
inelegível sem `--override`) e 1 (erro operacional: rastreamento desligado, MLflow
indisponível, sem champion/challenger/previous, versão inexistente ou já champion,
integridade, troca interrompida ou estado divergente, argumento inválido).

Depende do Model Registry: com `MONITORING_TRACKING=off` falha fechado.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlflow.utils.validation import MAX_TAG_VAL_LENGTH

from src import config
from src.logging_utils import get_logger, log_event
from src.models.integrity import (
    ModelIntegrityError,
    file_sha256,
    verify_model_integrity,
)
from src.monitoring import registry, tracking
from src.monitoring.monitor import EXIT_OPERATIONAL_ERROR, CliParser
from src.monitoring.promotion import PromotionDecision, decide_promotion
from src.monitoring.registry import (
    ACTION_PROMOTE,
    ACTION_ROLLBACK,
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    PREVIOUS_ALIAS,
    STATUS_CONSISTENT,
    STATUS_INCONSISTENT,
    RegistryStateError,
    get_alias_version,
)
from src.monitoring.retrain import RETRAIN_RUN_TYPE
from src.monitoring.stage import StagePreconditionError
from src.monitoring.tracking import TRACKING_ENV_VAR, TrackingUnavailableError

logger = get_logger("monitoring.promote")

EXIT_OK = 0
EXIT_REFUSED = 2

ROLLBACK_COMMAND = "rollback"
# O challenger foi comparado com outro champion: as AUCs não valem para o atual.
REASON_CHAMPION_MISMATCH = "champion_mismatch"
# Métricas do run de retreino (`retrain._register`) que a regra reavalia.
AUC_METRICS = (
    "champion_holdout_auc",
    "challenger_holdout_auc",
    "champion_reference_auc",
    "challenger_reference_auc",
)
STEP_ALIASES = "move_aliases"
STEP_FILE = "replace_file"
DEFAULT_PROMOTE_REASON = "Challenger elegível pela regra de promoção (§4.7)."
DEFAULT_ROLLBACK_REASON = "Rollback manual para a versão previous."
# Rótulos das mensagens: (sujeito, conclusão); a ação em si fica ASCII nas tags.
_LABELS = {
    ACTION_PROMOTE: ("A promoção", "Promoção concluída"),
    ACTION_ROLLBACK: ("O rollback", "Rollback concluído"),
}
# O erro vai resumido nos detalhes da auditoria (o limite da tag é do MLflow).
_ERROR_DETAIL_CHARS = 500
# Falhas de domínio no meio da troca: disparam a reversão com mensagem curta.
_TRANSITION_ERRORS = (
    TrackingUnavailableError,
    ModelIntegrityError,
    RegistryStateError,
    OSError,
)


class PromotionRefusedError(RuntimeError):
    """A regra recusa o challenger e não houve `--override` (código 2)."""


class TransitionFailedError(RuntimeError):
    """Troca interrompida ou estado divergente na reconferência (código 1)."""


# MARK: reavaliação da regra


@dataclass(frozen=True)
class Evaluation:
    """Veredito da regra para a versão, reavaliado com as métricas do run de origem."""

    version: int
    champion_version: int
    decision: PromotionDecision
    reasons: tuple[str, ...]
    aucs: dict[str, float]

    @property
    def eligible(self) -> bool:
        return not self.reasons


def _alias_number(alias: str) -> int | None:
    versao = get_alias_version(alias)
    return None if versao is None else int(versao.version)


def _current_champion() -> int:
    atual = _alias_number(CHAMPION_ALIAS)
    if atual is None:
        raise StagePreconditionError(
            f"Nenhuma versão de {config.REGISTERED_MODEL_NAME} tem o alias "
            f"`{CHAMPION_ALIAS}`: não há champion a substituir. Rode `make pipeline` "
            "(o `train` faz o bootstrap do champion) antes de promover."
        )
    return atual


def evaluate_candidate(version: int) -> Evaluation:
    """Reavalia a regra de promoção para a `version` (passo 1 da §4.7).

    As AUCs vêm das métricas do run de origem; métrica ausente vira NaN e reprova o
    eixo. O contrato vale quando o run de origem é um retreino (`enforce_contract` roda
    antes do treino). A integridade do ARQUIVO é conferida no passo 2 (falha dura); aqui
    a tag `model_sha256` da versão tem de ser a mesma do run de origem.
    """
    mv = registry.require_version(version)
    champion = _current_champion()
    if champion == int(version):
        raise RegistryStateError(
            f"Versão {version} já é o champion: nada a promover. Se o arquivo servido "
            "diverge do Registry, use `python -m src.monitoring.promote rollback`."
        )
    if not mv.run_id:
        raise ModelIntegrityError(
            f"Versão {version} sem run de origem: sem métricas para reavaliar a regra."
        )
    with tracking.tracking_errors("Falha ao ler o run de origem da versão"):
        run = tracking.client().get_run(mv.run_id)
    tag_versao = mv.tags.get("model_sha256")
    tag_run = run.data.tags.get("model_sha256")
    if not tag_versao or tag_versao != tag_run:
        raise ModelIntegrityError(
            f"A tag model_sha256 da versão {version} ({str(tag_versao)[:12]}...) "
            f"diverge da do run de origem ({str(tag_run)[:12]}...): a tag foi editada "
            "ou a versão não é a do run. Nada foi promovido."
        )
    metricas = run.data.metrics
    aucs = {chave: float(metricas.get(chave, math.nan)) for chave in AUC_METRICS}
    decisao = decide_promotion(
        contract_ok=run.data.tags.get("run_type") == RETRAIN_RUN_TYPE,
        integrity_ok=True,
        **aucs,
    )
    motivos = list(decisao.reasons)
    if run.data.tags.get("champion_version") != str(champion):
        motivos.append(REASON_CHAMPION_MISMATCH)
    return Evaluation(int(version), champion, decisao, tuple(motivos), aucs)


# MARK: estado servido


def served_state_problems(champion: int, previous: int | None) -> list[str]:
    """O que diverge entre o esperado e o estado real (Registry e `models/champion/`).

    Confere os aliases `champion` e `previous`, a tag `model_sha256` do champion contra
    o sha256 do arquivo servido e o par arquivo + metadata (`verify_model_integrity`).
    Lista vazia = `CONSISTENT`. MLflow indisponível também é divergência: o estado não
    pôde ser confirmado.
    """
    problemas: list[str] = []
    try:
        atual = get_alias_version(CHAMPION_ALIAS)
        anterior = _alias_number(PREVIOUS_ALIAS)
    except TrackingUnavailableError as exc:
        return [f"não foi possível reconferir o Registry: {exc}"]
    numero = None if atual is None else int(atual.version)
    if numero != champion:
        problemas.append(f"alias champion em v{numero}, esperado v{champion}")
    if anterior != previous:
        problemas.append(f"alias previous em v{anterior}, esperado v{previous}")
    arquivo = Path(config.CHAMPION_FILE)
    if not arquivo.is_file():
        return [*problemas, f"arquivo servido ausente: {arquivo}"]
    esperado = None if atual is None else atual.tags.get("model_sha256")
    try:
        obtido = file_sha256(arquivo)
    except OSError as exc:
        return [*problemas, f"arquivo servido ilegível ({arquivo}): {exc.strerror}"]
    if obtido != esperado:
        problemas.append(
            f"sha256 do arquivo servido ({obtido[:12]}...) diverge da tag model_sha256 "
            f"do champion ({str(esperado)[:12]}...)"
        )
    try:
        verify_model_integrity(arquivo, Path(config.CHAMPION_METADATA_FILE))
    except ModelIntegrityError as exc:
        problemas.append(str(exc))
    return problemas


# MARK: troca (passos 2 a 6)


@dataclass(frozen=True)
class TransitionResult:
    """Troca concluída e reconferida (`CONSISTENT`)."""

    action: str
    from_version: int
    to_version: int
    previous_version: int | None
    model_sha256: str
    audit_run_id: str


@dataclass(frozen=True)
class _Transition:
    action: str
    target: int
    expected_previous: int | None
    move: Callable[[], Any]
    reason: str
    operator: str
    details: dict[str, Any]


def _set_or_delete(alias: str, version: int | None) -> None:
    if version is None:
        registry.delete_alias(alias)
    else:
        registry.set_alias(alias, version)


def _revert(champion: int, previous: int | None) -> tuple[list[str], list[str]]:
    """Devolve os aliases e, se preciso, o arquivo; `(desfeito, falhas)`.

    `champion` primeiro e `previous` só se ele voltou: com o `champion` preso na versão
    nova, o `previous` intacto permite o `rollback`. O arquivo só é reexportado se os
    aliases voltaram e ele ainda diverge.
    """
    desfeito: list[str] = []
    falhas: list[str] = []
    for alias, versao in ((CHAMPION_ALIAS, champion), (PREVIOUS_ALIAS, previous)):
        try:
            _set_or_delete(alias, versao)
        except _TRANSITION_ERRORS as exc:
            falhas.append(f"{alias}: {exc}")
            return desfeito, falhas
        desfeito.append(f"{alias}=v{versao}" if versao else f"{alias} removido")
    if served_state_problems(champion, previous):
        try:
            registry.export_champion(champion)
        except _TRANSITION_ERRORS as exc:
            falhas.append(f"arquivo: {exc}")
        else:
            desfeito.append(f"arquivo=v{champion}")
    return desfeito, falhas


def _audit(transicao: _Transition, status: str, details: dict[str, Any]) -> str:
    return registry.log_audit_run(
        action=transicao.action,
        version=transicao.target,
        reason=transicao.reason,
        operator=transicao.operator,
        details={**transicao.details, **details},
        status=status,
    )


def _recover(
    transicao: _Transition,
    antes: tuple[int, int | None],
    passo: str,
    erro: BaseException | None,
) -> str:
    """Reversão + auditoria `INCONSISTENT` de uma troca interrompida; devolve a mensagem."""
    desfeito, falhas = _revert(*antes)
    problemas = served_state_problems(*antes)
    texto_erro = str(erro) if erro is not None else "interrompido"
    detalhes = {
        "failed_step": passo,
        "error": texto_erro[:_ERROR_DETAIL_CHARS],
        "reverted": desfeito,
        "revert_failures": falhas,
        "served_problems": problemas,
    }
    try:
        auditoria = _audit(transicao, STATUS_INCONSISTENT, detalhes)
    except (TrackingUnavailableError, ValueError) as exc:
        auditoria = None
        log_event(
            logger,
            "audit_run_not_recorded",
            level=logging.ERROR,
            action=transicao.action,
            message=str(exc),
        )
    log_event(
        logger,
        "champion_transition_failed",
        level=logging.ERROR,
        action=transicao.action,
        version=transicao.target,
        status=STATUS_INCONSISTENT,
        audit_run_id=auditoria,
        **detalhes,
    )
    onde = f"run de auditoria {auditoria}" if auditoria else "log (auditoria falhou)"
    mensagem = (
        f"{_LABELS[transicao.action][0]} para a versão {transicao.target} parou no "
        f"passo {passo}: {texto_erro}. Desfeito: {', '.join(desfeito) or 'nada'}. Estado "
        f"gravado como {STATUS_INCONSISTENT} ({onde})."
    )
    if problemas:
        mensagem += (
            f" O champion servido diverge do Registry ({'; '.join(problemas)}): o "
            "`load_bundle` recusa o modelo até a correção. Para voltar: "
            "`python -m src.monitoring.promote rollback`."
        )
    return mensagem


def _apply(transicao: _Transition) -> TransitionResult:
    """Passos 2 a 6: confere o arquivo, move os aliases, troca o arquivo e reconfere."""
    antes = (_current_champion(), _alias_number(PREVIOUS_ALIAS))
    if antes[0] != transicao.details["from_version"]:
        raise RegistryStateError(
            f"O champion mudou durante o comando (v{transicao.details['from_version']} "
            f"-> v{antes[0]}; promoção ou rollback em paralelo?): nada foi alterado. "
            "Rode o comando de novo."
        )
    with registry.staged_champion(transicao.target) as preparado:  # passo 2
        passo: str | None = STEP_ALIASES
        recuperado = False
        try:
            transicao.move()  # passos 3 e 4
            passo = STEP_FILE
            preparado.install()  # passo 5
            passo = None
        except _TRANSITION_ERRORS as exc:
            recuperado = True
            raise TransitionFailedError(_recover(transicao, antes, passo, exc)) from exc
        finally:
            # Interrupção não prevista (ex.: Ctrl+C): também tenta desfazer e auditar.
            if passo is not None and not recuperado:
                _recover(transicao, antes, passo, None)

    problemas = served_state_problems(transicao.target, transicao.expected_previous)
    status = STATUS_INCONSISTENT if problemas else STATUS_CONSISTENT
    detalhes = {"model_sha256": preparado.model_sha256, "served_problems": problemas}
    try:
        auditoria = _audit(transicao, status, detalhes)
    except TrackingUnavailableError as exc:
        raise TransitionFailedError(
            f"Troca do champion para a versão {transicao.target} "
            f"({transicao.action}) concluída e reconferida ({status}), mas o run de "
            f"auditoria não foi gravado: {exc} Registre a troca à mão (quem, quando, "
            "por quê)."
        ) from exc
    log_event(
        logger,
        "champion_transition_done",
        action=transicao.action,
        level=logging.INFO if not problemas else logging.ERROR,
        version=transicao.target,
        from_version=antes[0],
        status=status,
        audit_run_id=auditoria,
        served_problems=problemas,
    )
    if problemas:
        raise TransitionFailedError(
            f"{_LABELS[transicao.action][0]} para a versão {transicao.target} "
            f"terminou, mas a reconferência diverge ({'; '.join(problemas)}). Estado "
            f"gravado como {STATUS_INCONSISTENT} no run de auditoria {auditoria}: o "
            "`load_bundle` "
            "recusa o modelo até a correção (ex.: `python -m src.monitoring.promote "
            "rollback`)."
        )
    return TransitionResult(
        action=transicao.action,
        from_version=antes[0],
        to_version=transicao.target,
        previous_version=transicao.expected_previous,
        model_sha256=preparado.model_sha256,
        audit_run_id=auditoria,
    )


# MARK: comandos


def run_promote(
    *,
    version: int | None,
    override: bool,
    reason: str | None,
    operator: str | None = None,
) -> TransitionResult:
    """Promove `version` (padrão: o `challenger`) seguindo a ordem da §4.7."""
    quem = registry.resolve_operator(operator)
    tracking.configure_tracking()
    if version is None:
        version = _alias_number(CHALLENGER_ALIAS)
        if version is None:
            raise RegistryStateError(
                f"{config.REGISTERED_MODEL_NAME} não tem o alias `{CHALLENGER_ALIAS}`: "
                "não há challenger a promover. Rode `make retrain` ou passe "
                "`--version N`."
            )
    avaliacao = evaluate_candidate(version)
    d = avaliacao.decision
    numeros = {
        "gain_on_holdout": d.gain_on_holdout,
        "regression_on_reference": d.regression_on_reference,
        **avaliacao.aucs,
    }
    if not avaliacao.eligible and not override:
        log_event(
            logger,
            "promotion_refused",
            level=logging.WARNING,
            version=int(version),
            reasons=list(avaliacao.reasons),
            **numeros,
        )
        raise PromotionRefusedError(
            f"Versão {version} inelegível à promoção (motivos: "
            f"{', '.join(avaliacao.reasons)}): ganho no holdout "
            f"{d.gain_on_holdout:+.4f} (mínimo {config.PROMOTION_MIN_GAIN}), regressão "
            f"na referência {d.regression_on_reference:+.4f} (máximo "
            f"{config.PROMOTION_MAX_REGRESSION}). Nada foi alterado. Para promover mesmo "
            "assim: `--override --reason <motivo>` (o motivo vai para a auditoria)."
        )
    champion = avaliacao.champion_version
    return _apply(
        _Transition(
            action=ACTION_PROMOTE,
            target=int(version),
            expected_previous=champion,
            move=lambda: registry.promote_alias(int(version)),
            reason=reason or DEFAULT_PROMOTE_REASON,
            operator=quem,
            details={
                "from_version": champion,
                "to_version": int(version),
                "previous_version": champion,
                "override": bool(override),
                "eligible": avaliacao.eligible,
                "reasons": list(avaliacao.reasons),
                **numeros,
            },
        )
    )


def run_rollback(
    *, reason: str | None, operator: str | None = None
) -> TransitionResult:
    """Devolve o `champion` à versão `previous` (arquivo reexportado do artefato dela)."""
    quem = registry.resolve_operator(operator)
    tracking.configure_tracking()
    champion = _current_champion()
    anterior = _alias_number(PREVIOUS_ALIAS)
    if anterior is None:
        raise RegistryStateError(
            f"{config.REGISTERED_MODEL_NAME} não tem o alias `{PREVIOUS_ALIAS}`: não "
            "há versão para onde voltar (nenhuma promoção desde o último rollback). "
            "Nada foi alterado."
        )
    return _apply(
        _Transition(
            action=ACTION_ROLLBACK,
            target=anterior,
            expected_previous=None,
            move=registry.rollback_alias,
            reason=reason or DEFAULT_ROLLBACK_REASON,
            operator=quem,
            details={
                "from_version": champion,
                "to_version": anterior,
                "previous_version": None,
            },
        )
    )


# MARK: CLI


def _version_arg(texto: str) -> int:
    try:
        valor = int(texto)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{texto!r} não é um número de versão"
        ) from exc
    if valor < 1:
        raise argparse.ArgumentTypeError(f"{texto!r}: a versão do Registry começa em 1")
    return valor


def _reason_arg(parser: CliParser, reason: str | None) -> str | None:
    reason = reason.strip() if reason else None
    if reason and len(reason) > MAX_TAG_VAL_LENGTH:
        parser.error(f"--reason passa de {MAX_TAG_VAL_LENGTH} caracteres")
    return reason


def _promote_parser() -> CliParser:
    parser = CliParser(
        prog="python -m src.monitoring.promote",
        description=(
            "Promove o challenger a champion (comando humano): reavalia a regra, "
            "confere o sha256, move previous e champion, troca models/champion/ e "
            "audita. Saída: 0 promovido, 2 recusa (inelegível), 1 erro operacional."
        ),
        epilog="Rollback: python -m src.monitoring.promote rollback [--reason TEXTO]",
    )
    parser.add_argument(
        "--version",
        type=_version_arg,
        default=None,
        help="versão a promover (padrão: a do alias challenger)",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="promove mesmo inelegível (exige --reason; não dispensa a integridade)",
    )
    parser.add_argument(
        "--reason", default=None, help="motivo (gravado no run de auditoria)"
    )
    return parser


def _rollback_parser() -> CliParser:
    parser = CliParser(
        prog="python -m src.monitoring.promote rollback",
        description=(
            "Devolve o champion à versão previous, com o arquivo de models/champion/ "
            "reexportado do artefato dela, reconferência e auditoria. Saída: 0 feito, "
            "1 erro operacional (ex.: sem previous)."
        ),
    )
    parser.add_argument(
        "--reason", default=None, help="motivo (gravado no run de auditoria)"
    )
    return parser


def _summary(resultado: TransitionResult) -> str:
    previous = (
        f"v{resultado.previous_version}"
        if resultado.previous_version is not None
        else "nenhum"
    )
    return (
        f"{_LABELS[resultado.action][1]}: champion v{resultado.from_version} -> "
        f"v{resultado.to_version} (previous: {previous}). "
        f"{config.project_relative(Path(config.CHAMPION_FILE))} conferido (sha256 "
        f"{resultado.model_sha256[:12]}...). Estado {STATUS_CONSISTENT} gravado no run "
        f"de auditoria {resultado.audit_run_id}."
    )


def _run(onde: str, executar: Callable[[], TransitionResult]) -> int:
    """Executa o comando e traduz os erros de domínio em código de saída (`onde`: "na promoção")."""
    if not tracking.tracking_enabled():
        print(
            f"ERRO {onde}: rastreamento desligado ({TRACKING_ENV_VAR}=off). "
            "Promoção e rollback dependem do Model Registry e falham fechado sem ele.",
            file=sys.stderr,
        )
        return EXIT_OPERATIONAL_ERROR
    try:
        resultado = executar()
    except PromotionRefusedError as erro:
        print(f"RECUSADO {onde}: {erro}", file=sys.stderr)
        return EXIT_REFUSED
    except (
        TrackingUnavailableError,
        ModelIntegrityError,
        StagePreconditionError,
        RegistryStateError,
        TransitionFailedError,
        ValueError,
    ) as erro:
        # A mensagem já vem redigida (sem credenciais da URI); nada de traceback.
        print(f"ERRO {onde}: {registry.redact_text(str(erro))}", file=sys.stderr)
        return EXIT_OPERATIONAL_ERROR
    print(_summary(resultado))
    return EXIT_OK


def promote_main(argv: list[str] | None = None) -> int:
    parser = _promote_parser()
    args = parser.parse_args(argv)
    reason = _reason_arg(parser, args.reason)
    if args.override and not reason:
        parser.error("--override exige --reason (o motivo vai para o run de auditoria)")
    return _run(
        "na promoção",
        lambda: run_promote(
            version=args.version, override=args.override, reason=reason
        ),
    )


def rollback_main(argv: list[str] | None = None) -> int:
    parser = _rollback_parser()
    args = parser.parse_args(argv)
    reason = _reason_arg(parser, args.reason)
    return _run("no rollback", lambda: run_rollback(reason=reason))


def main(argv: list[str] | None = None) -> int:
    """`rollback` como primeiro argumento vai para `rollback_main`; o resto, promoção."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == [ROLLBACK_COMMAND]:
        return rollback_main(argv[1:])
    return promote_main(argv)


if __name__ == "__main__":
    sys.exit(main())
