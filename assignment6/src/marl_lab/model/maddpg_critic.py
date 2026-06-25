"""Per-agent centralised critic for MADDPG-discrete (Lowe et al. NeurIPS 2017).

Unlike QMIX/VDN/QPLEX which use ONE mixer to factor Q_tot, MADDPG runs
N independent centralised critics, **one per agent**. Each critic
``Q_i^C(s, ā)`` takes the global state PLUS the joint action (one-hot
encoded) and outputs a scalar Q value for agent i. This is the canonical
POSG-respecting alternative — each critic can have its own reward signal,
which makes MADDPG suitable for the cooperative-adversarial setting
where the cop and thief have OPPOSITE rewards (POSG, not Dec-POMDP).

For discrete-action environments (ours), we one-hot encode each agent's
action and concatenate to the state vector. The original paper uses
Gumbel-Softmax for the policy-gradient through actor outputs; we use a
critic-only variant: the local per-agent Q-net (already trained via
ε-greedy + DQN) acts as the implicit policy, and the centralised critic
provides the **per-agent** TD target — recovering the CTDE story without
the cooperative-reward assumption."""

from __future__ import annotations

import torch
import torch.nn as nn

from marl_lab.model.init import init_hidden, init_q_head


class MADDPGCritic(nn.Module):
    """Centralised per-agent critic ``Q_i^C(s, ā) → ℝ``.

    Input:  state ⊕ one-hot(a_cop) ⊕ one-hot(a_thief)
    Output: scalar Q value for one agent.

    Construct ONE instance per agent; each agent's critic is trained on
    that agent's own reward stream."""

    def __init__(self, state_dim: int, n_actions_per_agent: tuple[int, int],
                  hidden_sizes: tuple[int, ...] = (128, 128)) -> None:
        super().__init__()
        if any(n < 1 for n in n_actions_per_agent):
            raise ValueError(
                f"each n_actions must be >= 1, got {n_actions_per_agent}"
            )
        self.state_dim = int(state_dim)
        self.n_actions_per_agent = tuple(n_actions_per_agent)
        total_action_dim = sum(self.n_actions_per_agent)
        layers: list[nn.Module] = []
        prev = state_dim + total_action_dim
        for h in hidden_sizes:
            layer = nn.Linear(prev, h)
            init_hidden(layer)
            layers.extend([layer, nn.ReLU()])
            prev = h
        head = nn.Linear(prev, 1)
        init_q_head(head)
        layers.append(head)
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor,
                 joint_action_onehot: torch.Tensor) -> torch.Tensor:
        """Args:
            state: (..., state_dim) global state
            joint_action_onehot: (..., sum(n_actions)) joint action concat
        Returns:
            (...) scalar Q values (last dim collapsed)."""
        if state.shape[-1] != self.state_dim:
            raise ValueError(
                f"state last dim must be {self.state_dim}, "
                f"got {tuple(state.shape)}"
            )
        if joint_action_onehot.shape[-1] != sum(self.n_actions_per_agent):
            raise ValueError(
                f"joint_action_onehot last dim must be "
                f"{sum(self.n_actions_per_agent)}, "
                f"got {tuple(joint_action_onehot.shape)}"
            )
        x = torch.cat([state, joint_action_onehot], dim=-1)
        return self.net(x).squeeze(-1)


def one_hot_joint_action(
    actions_per_agent: dict[str, torch.Tensor],
    n_actions_per_agent: dict[str, int],
    agent_order: tuple[str, ...] = ("cop", "thief"),
) -> torch.Tensor:
    """Build one-hot joint-action tensor by concatenating per-agent one-hots.

    Args:
        actions_per_agent: dict {agent_id: (...) int64 indices}
        n_actions_per_agent: dict {agent_id: n_actions}
        agent_order: fixed iteration order for stable concatenation
    Returns:
        (..., sum_n_actions) float32 tensor.
    """
    parts: list[torch.Tensor] = []
    for a in agent_order:
        idx = actions_per_agent[a]
        n = n_actions_per_agent[a]
        # one_hot needs int64
        oh = torch.nn.functional.one_hot(idx.long(), num_classes=n).float()
        parts.append(oh)
    return torch.cat(parts, dim=-1)
