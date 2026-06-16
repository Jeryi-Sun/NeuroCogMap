import torch
import torch.nn as nn
import torch.nn.functional as F

from models import DualSystems


class DualSystemsHierarchicalBanditCogneuromapFull(DualSystems):
    def __init__(
        self,
        variant,
        store_data=False,
        enable_attention_lapse=True,
        enable_history_sensitive_stickiness=True,
        enable_transition_learning=True,
        enable_dynamic_arbitration=True,
        enable_stage2_rule_policy=True,
        enable_stage2_wsls=True,
        enable_forgetting=True,
    ):
        super().__init__(variant=variant, store_data=store_data)

        self.enable_attention_lapse = enable_attention_lapse
        self.enable_history_sensitive_stickiness = enable_history_sensitive_stickiness
        self.enable_transition_learning = enable_transition_learning
        self.enable_dynamic_arbitration = enable_dynamic_arbitration
        self.enable_stage2_rule_policy = enable_stage2_rule_policy
        self.enable_stage2_wsls = enable_stage2_wsls
        self.enable_forgetting = enable_forgetting

        self.lapse0 = nn.Parameter(0.01 * torch.randn([]))
        self.lapse_fatigue = nn.Parameter(0.01 * torch.randn([]))
        self.lapse_runlen = nn.Parameter(0.01 * torch.randn([]))
        self.lapse_omit_prev = nn.Parameter(0.01 * torch.randn([]))
        self.fatigue_lr = nn.Parameter(0.01 * torch.randn([]))

        self.tau0 = nn.Parameter(0.01 * torch.randn([]))
        self.tau_u = nn.Parameter(0.01 * torch.randn([]))
        self.tau_v = nn.Parameter(0.01 * torch.randn([]))
        self.tau_runlen = nn.Parameter(0.01 * torch.randn([]))
        self.tau_omit_prev = nn.Parameter(0.01 * torch.randn([]))

        self.vol_lr = nn.Parameter(0.01 * torch.randn([]))

        self.stick_runlen = nn.Parameter(0.01 * torch.randn([]))
        self.stick_prev_reward = nn.Parameter(0.01 * torch.randn([]))

        self.stage2_persev = nn.Parameter(0.01 * torch.randn([]))
        self.stage2_wsls = nn.Parameter(0.01 * torch.randn([]))

        self.policy_strength = nn.Parameter(0.01 * torch.randn([]))
        self.policy_switch0 = nn.Parameter(0.01 * torch.randn([]))
        self.policy_switch_pe = nn.Parameter(0.01 * torch.randn([]))

        self.forgetting = nn.Parameter(0.01 * torch.randn([]))

        self.dirichlet_prior = 1.0

    def forward(self, data):
        logits = self.forward_two_step(data) if self.variant == "two_step" else self.forward_one_step(data)
        return logits

    @staticmethod
    def _onehot2(action_idx, device):
        v = torch.zeros(2, device=device)
        if action_idx in (0, 1):
            v[action_idx] = 1.0
        return v

    @staticmethod
    def _dirichlet_beta_var(a, b):
        t = a + b
        return (a * b) / (t * t * (t + 1.0))

    def _transition_uncertainty(self, counts):
        a0 = counts[:, 0]
        a1 = counts[:, 1]
        var = self._dirichlet_beta_var(a0, a1)
        return var.mean()

    def _lapse_rate(self, fatigue, runlen, omit_prev):
        return torch.sigmoid(
            self.lapse0
            + self.lapse_fatigue * fatigue
            + self.lapse_runlen * runlen
            + self.lapse_omit_prev * omit_prev
        )

    @staticmethod
    def _mix_with_uniform_logits(logits_2, lapse):
        probs = (1.0 - lapse) * F.softmax(logits_2, dim=-1) + lapse * 0.5
        probs = probs.clamp_min(1e-8)
        return probs.log()

    def forward_two_step(self, data):
        if self.store_data:
            self.data = []

        alpha_base = torch.sigmoid(self.alpha)
        lambd = torch.sigmoid(self.lambd)
        stick_base = torch.tanh(self.stickiness)

        action_1 = data["choice"][:, :, 0].long()
        action_2 = data["choice"][:, :, 1].long()
        state = data["current_state"][:, :, 1].long()
        reward = data["reward"][:, :, 1]

        device = action_1.device
        n_participants = action_1.shape[0]
        n_trials = action_1.shape[1]

        fixed_T = torch.tensor([[0.7, 0.3], [0.3, 0.7]], device=device)

        logits = torch.zeros(n_participants, n_trials, 2, 2, device=device)

        for par in range(n_participants):
            q_mf = torch.zeros(3, 2, device=device)
            action_repeat = torch.zeros(2, device=device)

            runlen = torch.zeros([], device=device)
            fatigue = torch.zeros([], device=device)
            omit_prev = torch.zeros([], device=device)

            volatility = torch.zeros([], device=device)

            # 注意：这些“内部状态”虽然不需要梯度，但会参与带参数的前向计算。
            # 若后续对同一个 tensor 做 in-place 更新，可能触发 autograd 的 version counter 报错。
            # 因此这里用 python 缓存（或 clone+重绑定）来避免 in-place 修改已被保存用于反传的张量。
            trans_counts = torch.ones(2, 2, device=device) * self.dirichlet_prior

            last_a2_by_state = [-1, -1]   # python int
            last_r_by_state = [0.0, 0.0]  # python float
            policy_pref = torch.zeros(2, 2, device=device)

            prev_a1 = -1
            prev_reward = torch.zeros([], device=device)

            for trial in range(n_trials):
                a1 = int(action_1[par, trial].item())
                omit_this = torch.tensor(float(a1 == self.ignore_index), device=device)

                if self.enable_transition_learning:
                    T = trans_counts / trans_counts.sum(dim=1, keepdim=True)
                else:
                    T = fixed_T

                max_q, _ = torch.max(q_mf[1:], dim=1)
                q_mb = T @ max_q

                if self.enable_dynamic_arbitration:
                    u_t = self._transition_uncertainty(trans_counts)
                    w_t = torch.sigmoid(
                        self.tau0
                        + self.tau_u * (-u_t)
                        + self.tau_v * (-volatility)
                        + self.tau_runlen * runlen
                        + self.tau_omit_prev * omit_prev
                    )
                else:
                    w_t = torch.sigmoid(self.tau)

                q_net = w_t * q_mb + (1.0 - w_t) * q_mf[0]

                if self.enable_history_sensitive_stickiness:
                    stick_scale = 1.0 + torch.tanh(self.stick_runlen) * runlen + torch.tanh(self.stick_prev_reward) * prev_reward
                    stick_term = action_repeat * stick_base * stick_scale
                else:
                    stick_term = action_repeat * stick_base

                stage1_logits = self.value_logits(q_net + stick_term)
                if self.enable_attention_lapse:
                    lapse = self._lapse_rate(fatigue, runlen, omit_prev)
                    stage1_logits = self._mix_with_uniform_logits(stage1_logits, lapse)
                logits[par, trial, 0] = stage1_logits

                if a1 == self.ignore_index:
                    fatigue_lr = torch.sigmoid(self.fatigue_lr)
                    fatigue = (1.0 - fatigue_lr) * fatigue + fatigue_lr * omit_this
                    omit_prev = omit_this
                    action_repeat = torch.zeros(2, device=device)
                    runlen = torch.zeros([], device=device)
                    prev_a1 = -1
                    prev_reward = torch.zeros([], device=device)
                    continue

                a2 = int(action_2[par, trial].item())
                s2 = int(state[par, trial].item())
                r2 = reward[par, trial]

                q2 = q_mf[s2 + 1].clone()
                if self.enable_stage2_wsls:
                    prev_a2 = int(last_a2_by_state[s2])
                    if prev_a2 in (0, 1):
                        prev2 = self._onehot2(prev_a2, device=device)
                        prev_r = torch.tensor(float(last_r_by_state[s2]), device=device)
                        q2 = q2 + torch.tanh(self.stage2_persev) * prev2 + torch.tanh(self.stage2_wsls) * prev_r * prev2

                if self.enable_stage2_rule_policy:
                    q2 = q2 + torch.tanh(self.policy_strength) * policy_pref[s2]

                stage2_logits = self.value_logits(q2)
                if self.enable_attention_lapse:
                    lapse2 = self._lapse_rate(fatigue, runlen, omit_prev)
                    stage2_logits = self._mix_with_uniform_logits(stage2_logits, lapse2)
                logits[par, trial, 1] = stage2_logits

                if prev_a1 == a1:
                    runlen = runlen + 1.0
                else:
                    runlen = torch.ones([], device=device)
                prev_a1 = a1

                if s2 in (0, 1):
                    trans_counts_new = trans_counts.clone()
                    trans_counts_new[a1, s2] = trans_counts_new[a1, s2] + 1.0
                    trans_counts = trans_counts_new

                if self.enable_forgetting:
                    phi = torch.sigmoid(self.forgetting)
                    q_mf = (1.0 - phi) * q_mf

                if a2 != self.ignore_index:
                    delta_1 = q_mf[s2 + 1, a2] - q_mf[0, a1]
                    delta_2 = r2 - q_mf[s2 + 1, a2]
                    delta_2 = torch.nan_to_num(delta_2, nan=0.0)

                    # 重要：避免对 q_mf 做 in-place 更新（q_mf 参与带参数的计算图，in-place 会触发 version mismatch）
                    onehot_a1 = F.one_hot(torch.tensor(a1, device=device), num_classes=2).float()
                    onehot_a2 = F.one_hot(torch.tensor(a2, device=device), num_classes=2).float()
                    dq = torch.zeros_like(q_mf)
                    dq[0] = (alpha_base * delta_1 + lambd * alpha_base * delta_2) * onehot_a1
                    dq[s2 + 1] = (alpha_base * delta_2) * onehot_a2
                    q_mf = q_mf + dq

                    vol_lr = torch.sigmoid(self.vol_lr)
                    volatility = (1.0 - vol_lr) * volatility + vol_lr * delta_2.abs()

                    if self.enable_stage2_rule_policy:
                        switch_p = torch.sigmoid(self.policy_switch0 + self.policy_switch_pe * delta_2.abs())
                        direction = (2.0 * torch.nan_to_num(r2, nan=0.0) - 1.0).clamp(min=-1.0, max=1.0)
                        update = direction * self._onehot2(a2, device=device)
                        policy_pref_new = policy_pref.clone()
                        policy_pref_new[s2] = (1.0 - switch_p) * policy_pref[s2] + switch_p * update
                        policy_pref = policy_pref_new

                    last_a2_by_state[s2] = int(a2)
                    last_r_by_state[s2] = float(torch.nan_to_num(r2, nan=0.0).item())
                    prev_reward = torch.nan_to_num(r2, nan=0.0)

                action_repeat = torch.zeros(2, device=device)
                action_repeat[a1] = 1.0

                fatigue_lr = torch.sigmoid(self.fatigue_lr)
                fatigue = (1.0 - fatigue_lr) * fatigue + fatigue_lr * omit_this
                omit_prev = omit_this

        return logits


