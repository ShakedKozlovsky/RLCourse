"""Guardrails that stop a bad submission before it hits SMTP.

Extracted from ``cli/commands.py`` in v1.18 to keep that file under the
≤ 250-LOC audit gate. The guards themselves are the interesting part —
placeholder metadata + missing checkpoint have both been observed as
real footguns during test-sends (see CHANGELOG § v1.18)."""

from __future__ import annotations

# Values that indicate the yaml default was never overridden by env var
# or hand-edit. Empty string catches "" from a bad env var too.
PLACEHOLDER_STRINGS: frozenset[str] = frozenset(
    {"TBD", "TODO", "?", "TBD-8CHR", ""}
)


def resolve_or_refuse_checkpoint(args, sdk_factory) -> None:
    """Fill ``args.checkpoint`` from yaml default; refuse if none.

    Called at the top of ``cmd_play_and_send``. If the user passed
    ``--checkpoint``, we're done. Else look up
    ``submission.default_checkpoint`` in yaml (via ``sdk_factory()`` —
    threaded as a factory so unit tests can inject a stub without
    building a real SDK). If neither source has a checkpoint, and this
    isn't a ``--dry-run``, refuse with a helpful message."""
    if args.checkpoint:
        return
    sdk = sdk_factory()
    default_ckpt = sdk.config.get("submission.default_checkpoint")
    if default_ckpt:
        args.checkpoint = default_ckpt
        return
    if getattr(args, "dry_run", False):
        return
    raise SystemExit(
        "refusing to send random-play as your submission: no "
        "--checkpoint provided and no submission.default_checkpoint in "
        "yaml. Pass --checkpoint saved_models/maddpg_shaped.pt or set "
        "submission.default_checkpoint in configs/setup.yaml. "
        "(Use --dry-run to bypass for code-path testing.)"
    )


def refuse_placeholder_metadata(report, *, dry_run: bool) -> None:
    """Raise ``SystemExit`` if the report still carries placeholders.

    Checks ``group_name``, ``group_code``, and every ``student.{id,
    full_name}``. ``dry_run=True`` skips the check — dry runs don't
    actually send, so metadata validity is moot there.

    Bypass for testing: ``send-report --dry-run`` still exercises the
    whole build path with any input JSON."""
    if dry_run:
        return
    bad: list[str] = []
    if report.group_name in PLACEHOLDER_STRINGS:
        bad.append(f"group_name={report.group_name!r}")
    if report.group_code in PLACEHOLDER_STRINGS:
        bad.append(f"group_code={report.group_code!r}")
    for s in report.students:
        if s.id in PLACEHOLDER_STRINGS:
            bad.append(f"student[{s.role}].id={s.id!r}")
        if s.full_name in PLACEHOLDER_STRINGS:
            bad.append(f"student[{s.role}].full_name={s.full_name!r}")
    if bad:
        raise SystemExit(
            "refusing to send: submission metadata contains placeholders "
            "(" + ", ".join(bad) + "). Set the MARL_STUDENT_A_ID / "
            "MARL_STUDENT_A_NAME / MARL_GROUP_CODE / MARL_GROUP_NAME env "
            "vars, or fix configs/setup.yaml::submission. Use --dry-run "
            "to bypass this check for code-path testing."
        )
