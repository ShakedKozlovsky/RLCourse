"""Bonus-game tests (spec § 9): scoring rule + runner + formatter + agreement."""

from __future__ import annotations

import json

import numpy as np
import pytest

from marl_lab.environment.reward import RewardConfig
from marl_lab.gmail.bonus_formatter import (
    bonus_email_subject,
    bonus_report_to_json,
    build_bonus_idempotency_key,
    verify_peer_agreement,
)
from marl_lab.model.recurrent_q import QPerAgent
from marl_lab.sensor.partial_observation import obs_dim
from marl_lab.services.bonus_game_runner import (
    BonusGameRunner,
    BonusRunnerConfig,
    make_local_policy_from_qnet,
)
from marl_lab.services.bonus_scoring import (
    BONUS_LOSS,
    BONUS_TIE,
    BONUS_WIN,
    compute_bonus_claim,
)
from marl_lab.shared.types import BonusGameReport, BonusSubGameResult, StudentEntry

# ----- Scoring rule (§ 9.2) -----

def test_bonus_scoring_winner_gets_10_loser_gets_7() -> None:
    claim = compute_bonus_claim({"Team-A": 60, "Team-B": 80})
    assert claim == {"Team-A": BONUS_LOSS, "Team-B": BONUS_WIN}
    assert claim["Team-B"] == 10
    assert claim["Team-A"] == 7


def test_bonus_scoring_reverse_order() -> None:
    claim = compute_bonus_claim({"Team-A": 90, "Team-B": 30})
    assert claim["Team-A"] == 10
    assert claim["Team-B"] == 7


def test_bonus_scoring_tie_gives_5_each() -> None:
    claim = compute_bonus_claim({"Team-A": 60, "Team-B": 60})
    assert claim == {"Team-A": BONUS_TIE, "Team-B": BONUS_TIE}


def test_bonus_scoring_rejects_wrong_group_count() -> None:
    with pytest.raises(ValueError):
        compute_bonus_claim({"Team-A": 10})
    with pytest.raises(ValueError):
        compute_bonus_claim({"A": 10, "B": 20, "C": 30})


# ----- BonusGameRunner (in-process peer via same untrained Q-net) -----

@pytest.fixture
def runner() -> BonusGameRunner:
    return BonusGameRunner(
        cfg=BonusRunnerConfig(n_sub_games_per_side=3, grid_size=(4, 4),
                                max_moves=10, max_barriers=2,
                                enable_barriers=False, observation_radius=1),
        reward_cfg=RewardConfig(),
        rng=np.random.default_rng(0),
    )


@pytest.fixture
def two_policies() -> tuple:
    o = obs_dim(1)
    a = QPerAgent(obs_dim=o, n_actions=6, hidden_sizes=(16,), gru_hidden_size=8)
    b = QPerAgent(obs_dim=o, n_actions=6, hidden_sizes=(16,), gru_hidden_size=8)
    return make_local_policy_from_qnet(a), make_local_policy_from_qnet(b)


def test_runner_produces_6_sub_games(runner, two_policies) -> None:
    local, peer = two_policies
    r = runner.play_bonus_match(
        local_group_name="Team-A", peer_group_name="Team-B",
        local_students=[StudentEntry(role="A", full_name="X", id="1")],
        peer_students=[StudentEntry(role="A", full_name="Y", id="2")],
        local_github_repo="https://x", peer_github_repo="https://y",
        local_policy=local, peer_policy=peer, seed=0,
    )
    assert len(r.sub_games) == 6
    assert [sg.id for sg in r.sub_games] == [1, 2, 3, 4, 5, 6]


def test_runner_role_alternation_after_3(runner, two_policies) -> None:
    """Spec § 9.1: sub-games 1–3 local=cop; sub-games 4–6 swap."""
    local, peer = two_policies
    r = runner.play_bonus_match(
        local_group_name="Team-A", peer_group_name="Team-B",
        local_students=[StudentEntry(role="A", full_name="X", id="1")],
        peer_students=[StudentEntry(role="A", full_name="Y", id="2")],
        local_github_repo="https://x", peer_github_repo="https://y",
        local_policy=local, peer_policy=peer, seed=0,
    )
    for sg in r.sub_games[:3]:
        assert sg.cop_group == "Team-A"
        assert sg.thief_group == "Team-B"
    for sg in r.sub_games[3:]:
        assert sg.cop_group == "Team-B"
        assert sg.thief_group == "Team-A"


def test_runner_totals_match_sum_per_group(runner, two_policies) -> None:
    local, peer = two_policies
    r = runner.play_bonus_match(
        local_group_name="Team-A", peer_group_name="Team-B",
        local_students=[], peer_students=[],
        local_github_repo="x", peer_github_repo="y",
        local_policy=local, peer_policy=peer, seed=0,
    )
    expected_a = 0
    expected_b = 0
    for sg in r.sub_games:
        expected_a += sg.scores[
            "cop" if sg.cop_group == "Team-A" else "thief"
        ]
        expected_b += sg.scores[
            "cop" if sg.cop_group == "Team-B" else "thief"
        ]
    assert r.totals_by_group["Team-A"] == expected_a
    assert r.totals_by_group["Team-B"] == expected_b


def test_runner_bonus_claim_matches_scoring_rule(runner, two_policies) -> None:
    local, peer = two_policies
    r = runner.play_bonus_match(
        local_group_name="Team-A", peer_group_name="Team-B",
        local_students=[], peer_students=[],
        local_github_repo="x", peer_github_repo="y",
        local_policy=local, peer_policy=peer, seed=0,
    )
    assert r.bonus_claim == compute_bonus_claim(r.totals_by_group)


def test_runner_mutual_agreement_false_by_default(runner, two_policies) -> None:
    local, peer = two_policies
    r = runner.play_bonus_match(
        local_group_name="Team-A", peer_group_name="Team-B",
        local_students=[], peer_students=[],
        local_github_repo="x", peer_github_repo="y",
        local_policy=local, peer_policy=peer, seed=0,
    )
    assert r.mutual_agreement is False


# ----- Bonus formatter (§ 9.4 JSON + subject + idempotency) -----

def _dummy_report(seed_int: int = 0) -> BonusGameReport:
    sub_games = []
    for i in range(1, 7):
        cop_g = "Team-A" if i <= 3 else "Team-B"
        thief_g = "Team-B" if i <= 3 else "Team-A"
        winner = "cop" if (i + seed_int) % 2 == 0 else "thief"
        scores = {"cop": 20 if winner == "cop" else 5,
                    "thief": 5 if winner == "cop" else 10}
        sub_games.append(BonusSubGameResult(
            id=i, cop_group=cop_g, thief_group=thief_g,
            winner=winner, scores=scores,
        ))
    totals = {"Team-A": 0, "Team-B": 0}
    for sg in sub_games:
        totals[sg.cop_group] += sg.scores["cop"]
        totals[sg.thief_group] += sg.scores["thief"]
    return BonusGameReport(
        groups={"group_1": "Team-A", "group_2": "Team-B"},
        github_repo_group_1="https://gh/a", github_repo_group_2="https://gh/b",
        timezone="Asia/Jerusalem",
        students_group_1=[StudentEntry(role="A", full_name="Alice", id="1")],
        students_group_2=[StudentEntry(role="A", full_name="Bob", id="2")],
        sub_games=sub_games, totals_by_group=totals,
        bonus_claim=compute_bonus_claim(totals),
        mutual_agreement=False,
    )


def test_bonus_json_carries_report_type() -> None:
    js = bonus_report_to_json(_dummy_report(), include_provenance=False)
    payload = json.loads(js)
    assert payload["report_type"] == "bonus_game"


def test_bonus_json_carries_all_spec_9_4_fields() -> None:
    js = bonus_report_to_json(_dummy_report(), include_provenance=False)
    payload = json.loads(js)
    for k in ("groups", "github_repo_group_1", "github_repo_group_2",
                "timezone", "students_group_1", "students_group_2",
                "sub_games", "totals_by_group", "bonus_claim",
                "mutual_agreement"):
        assert k in payload, f"missing field {k}"


def test_bonus_email_subject_format() -> None:
    subj = bonus_email_subject(_dummy_report())
    assert subj == "[MARL Bonus Game] Team-A vs Team-B – Final Report"


def test_bonus_idempotency_key_deterministic() -> None:
    a = build_bonus_idempotency_key(_dummy_report())
    b = build_bonus_idempotency_key(_dummy_report())
    assert a == b
    assert len(a) == 16


def test_bonus_idempotency_key_changes_with_content() -> None:
    a = build_bonus_idempotency_key(_dummy_report(seed_int=0))
    b = build_bonus_idempotency_key(_dummy_report(seed_int=1))
    assert a != b


def test_bonus_idempotency_key_independent_of_mutual_agreement() -> None:
    """Flipping mutual_agreement AFTER agreement check must not change the id."""
    r1 = _dummy_report()
    r2 = _dummy_report()
    r2.mutual_agreement = True
    assert build_bonus_idempotency_key(r1) == build_bonus_idempotency_key(r2)


# ----- Peer agreement -----

def test_verify_peer_agreement_matching_reports() -> None:
    r = _dummy_report()
    peer_json = bonus_report_to_json(r, include_provenance=False)
    agreed, reason = verify_peer_agreement(r, peer_json)
    assert agreed is True
    assert reason == "match"


def test_verify_peer_agreement_disagreement_on_totals() -> None:
    r = _dummy_report()
    peer_json = bonus_report_to_json(r, include_provenance=False)
    peer_payload = json.loads(peer_json)
    peer_payload["totals_by_group"] = {"Team-A": 999, "Team-B": 0}
    agreed, reason = verify_peer_agreement(r, json.dumps(peer_payload))
    assert agreed is False
    assert "totals_by_group" in reason


def test_verify_peer_agreement_rejects_non_bonus_report_type() -> None:
    r = _dummy_report()
    fake_peer = json.dumps({"report_type": "regular_game", "groups": {}})
    agreed, reason = verify_peer_agreement(r, fake_peer)
    assert agreed is False
    assert "report_type" in reason


def test_verify_peer_agreement_bad_json() -> None:
    r = _dummy_report()
    agreed, reason = verify_peer_agreement(r, "{not valid json")
    assert agreed is False
    assert "parse error" in reason


# ----- CLI parser wiring -----

def test_cli_play_bonus_subcommand_registered() -> None:
    from marl_lab.cli.main import build_parser
    parser = build_parser()
    sub_action = next(a for a in parser._actions if a.choices)   # noqa: SLF001
    assert "play-bonus" in sub_action.choices


# ----- v1.17 regressions -----

def test_verify_peer_agreement_ignores_group_1_group_2_label_flip() -> None:
    """The group_1 / group_2 dict keys are per-team-arbitrary positional
    labels — each team is free to call themselves group_1. Mutual
    agreement must depend on the SET of teams, not the labelling.

    Before v1.17 the canonicaliser sorted by dict key, so two identical
    matches labelled from opposite perspectives would spuriously fail
    to agree — breaking every real cross-team bonus submission.
    """
    r = _dummy_report()
    peer_json = bonus_report_to_json(r, include_provenance=False)
    peer_payload = json.loads(peer_json)
    # Simulate the peer's naturally-flipped label assignment:
    g1, g2 = peer_payload["groups"]["group_1"], peer_payload["groups"]["group_2"]
    peer_payload["groups"] = {"group_1": g2, "group_2": g1}
    agreed, reason = verify_peer_agreement(r, json.dumps(peer_payload))
    assert agreed is True, f"label-flip broke agreement: {reason}"


def test_bonus_runner_raises_on_invalid_winner(monkeypatch) -> None:
    """v1.17: bonus runner used to `info["winner"] or "thief"` — silently
    defaulted an invalid winner to a thief-win, corrupting bonus scoring.
    Now must raise."""
    runner = BonusGameRunner(cfg=BonusRunnerConfig(observation_radius=1),
                              reward_cfg=RewardConfig(),
                              rng=np.random.default_rng(0))
    # Monkeypatch env.step to always return done=True with a garbage winner
    def bad_step(_actions):
        return ({"cop": np.zeros(runner.env.obs_dim, dtype=np.float32),
                 "thief": np.zeros(runner.env.obs_dim, dtype=np.float32)},
                {"cop": 0.0, "thief": 0.0},
                True,
                {"winner": "banana"})
    monkeypatch.setattr(runner.env, "step", bad_step)

    def zero_policy(_role, _obs):
        return 0
    with pytest.raises(RuntimeError, match="invalid winner"):
        runner._play_one(cop_policy=zero_policy, thief_policy=zero_policy,
                          sub_game_id=1, cop_group="A", thief_group="B",
                          seed=0)
