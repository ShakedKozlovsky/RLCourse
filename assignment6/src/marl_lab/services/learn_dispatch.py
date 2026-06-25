"""Algorithm dispatch for ``MarlTrainer.learn_step()`` — extracted to keep
marl_trainer.py lean. Each branch builds the right kwargs for its update
fn and normalises the diagnostic dict into the same shape across algos."""

from __future__ import annotations

from marl_lab.memory.centralised_buffer import BatchTensors
from marl_lab.services.iql_update import apply_iql_update
from marl_lab.services.maddpg_update import apply_maddpg_update
from marl_lab.services.qmix_update import apply_qmix_update
from marl_lab.services.qplex_update import apply_qplex_update
from marl_lab.services.vdn_update import apply_vdn_update


def dispatch_learn_step(trainer, batch: BatchTensors) -> dict:
    """Dispatch one TD update to the right algorithm-specific function.

    ``trainer`` is the MarlTrainer instance (we read q_nets, target_q_nets,
    mixer/target_mixer or critics/target_critics, opts, device, cfg from it).
    Returns a normalised dict with keys: critic_loss, mean_q_cop, mean_q_thief,
    and optionally target_drift."""
    algo = trainer.cfg.algo
    common = {
        "q_nets": trainer.q_nets,
        "target_q_nets": trainer.target_q_nets,
        "batch": batch,
        "gamma": trainer.cfg.gamma,
        "tau": trainer.cfg.tau,
        "device": trainer.device,
    }
    if algo == "qmix":
        d = apply_qmix_update(
            mixer=trainer.mixer, target_mixer=trainer.target_mixer,
            critic_opt=trainer.opts, **common,
        )
        return {"skipped": False, "critic_loss": d.critic_loss,
                "mean_q_cop": d.mean_q_cop, "mean_q_thief": d.mean_q_thief,
                "target_drift": d.target_drift}
    if algo == "vdn":
        d = apply_vdn_update(
            mixer=trainer.mixer, target_mixer=trainer.target_mixer,
            critic_opt=trainer.opts, **common,
        )
        return {"skipped": False, "critic_loss": d.critic_loss,
                "mean_q_cop": d.mean_q_cop, "mean_q_thief": d.mean_q_thief}
    if algo == "qplex":
        d = apply_qplex_update(
            mixer=trainer.mixer, target_mixer=trainer.target_mixer,
            critic_opt=trainer.opts, **common,
        )
        return {"skipped": False, "critic_loss": d.critic_loss,
                "mean_q_cop": d.mean_q_cop, "mean_q_thief": d.mean_q_thief,
                "target_drift": d.target_drift}
    if algo == "maddpg":
        d = apply_maddpg_update(
            critics=trainer.critics, target_critics=trainer.target_critics,
            critic_opt=trainer.opts, **common,
        )
        return {"skipped": False,
                "critic_loss": (d.critic_loss_cop + d.critic_loss_thief) / 2.0,
                "mean_q_cop": d.mean_q_cop, "mean_q_thief": d.mean_q_thief,
                "target_drift": d.target_drift}
    # iql
    d_iql = apply_iql_update(critic_opts=trainer.opts, **common)
    return {"skipped": False,
            "critic_loss": (d_iql.critic_loss_cop + d_iql.critic_loss_thief) / 2.0,
            "mean_q_cop": d_iql.mean_q_cop, "mean_q_thief": d_iql.mean_q_thief}
