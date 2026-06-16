import torch
import torch.nn as nn
import torch.nn.functional as F

from models import DualSystems


class DualSystemsHierarchicalBanditCogneuromapSimple(DualSystems):
    """
    DualSystems 的可替换扩展版（two-step），在不改变数据接口的前提下增加：
    - omission/lapse：并允许 omission 影响下一 trial 的随机性/默认化
    - 动态仲裁：w_t 随近期预测误差与 omission 变化
    - 二阶段（按 state/planet）WSLS 与 perseveration
    - 遗忘/重置与自适应学习率：用于捕捉非平稳与“重启”动态
    """

    def __init__(
        self,
        variant,
        store_data: bool = False,
        use_lapse: bool = True,
        use_dynamic_arbitration: bool = True,
        use_stage2_wsls: bool = True,
        use_forgetting: bool = True,
        use_adaptive_alpha: bool = True,
        use_post_omission_reset: bool = True,
    ):
        super().__init__(variant=variant, store_data=store_data)

        self.use_lapse = use_lapse
        self.use_dynamic_arbitration = use_dynamic_arbitration
        self.use_stage2_wsls = use_stage2_wsls
        self.use_forgetting = use_forgetting
        self.use_adaptive_alpha = use_adaptive_alpha
        self.use_post_omission_reset = use_post_omission_reset

        # ---------- lapse / omission ----------
        # stage-specific base lapse + post-omission增量（logit 空间）
        self.lapse1_logit = nn.Parameter(0.01 * torch.randn([]))
        self.lapse2_logit = nn.Parameter(0.01 * torch.randn([]))
        self.lapse1_post_omit = nn.Parameter(0.01 * torch.randn([]))
        self.lapse2_post_omit = nn.Parameter(0.01 * torch.randn([]))

        # ---------- dynamic arbitration ----------
        # 在基线 self.tau (logit) 上加入由 |PE| 与 omission 调制的增量
        self.tau_pe = nn.Parameter(0.01 * torch.randn([]))
        self.tau_post_omit = nn.Parameter(0.01 * torch.randn([]))

        # ---------- stage-2 WSLS / perseveration (planet/state-specific) ----------
        # state in {0,1} -> 对应二阶段两个“星球/planet”
        self.rho2 = nn.Parameter(0.01 * torch.randn(2))    # 纯重复倾向
        self.kappa2 = nn.Parameter(0.01 * torch.randn(2))  # 奖励门控重复（WSLS 形式）

        # ---------- forgetting / reset / adaptive alpha ----------
        self.forgetting_logit = nn.Parameter(0.01 * torch.randn([]))  # phi in (0,1)
        self.reset_logit = nn.Parameter(0.01 * torch.randn([]))       # post-omission reset strength
        self.alpha_pe = nn.Parameter(0.01 * torch.randn([]))          # adaptive alpha gain

    def _safe_log_probs_from_mixture(self, logits, lapse_prob):
        """
        将 (1-lapse)*softmax(logits) + lapse*Uniform 的混合策略
        转为 log-prob 形式的 logits（softmax(logp)=p），便于 cross_entropy 直接使用。
        """
        probs = F.softmax(logits, dim=-1)
        uniform = torch.ones_like(probs) / probs.shape[-1]
        mixed = (1.0 - lapse_prob) * probs + lapse_prob * uniform
        mixed = mixed.clamp_min(1e-12)
        return torch.log(mixed)

    def forward(self, data):
        logits = self.forward_two_step(data) if self.variant == "two_step" else self.forward_one_step(data)
        return logits

    def forward_two_step(self, data):
        if self.store_data:
            self.data = []

        # base parameters（与 DualSystems 保持一致的变换）
        alpha_base = torch.sigmoid(self.alpha)
        lambd = torch.sigmoid(self.lambd)
        stickiness = torch.tanh(self.stickiness)

        action_1 = data["choice"][:, :, 0].long()
        action_2 = data["choice"][:, :, 1].long()
        state = data["current_state"][:, :, 1].long()
        reward = data["reward"][:, :, 1]

        transition_matrix = torch.tensor([[0.7, 0.3], [0.3, 0.7]], device=action_1.device, dtype=torch.float32)
        n_participants, n_trials = action_1.shape

        logits_out = torch.zeros(n_participants, n_trials, 2, 2, device=action_1.device, dtype=torch.float32)

        for par in range(n_participants):
            q_mf = torch.zeros(3, 2, device=action_1.device, dtype=torch.float32)
            prev_a1 = None

            prev_omit = False
            prev_pe_abs = torch.tensor(0.0, device=action_1.device)
            prev_a2 = -1
            prev_s2 = -1
            prev_r2 = 0.0

            for trial in range(n_trials):
                # forgetting / reset（用于非平稳与“重启”）
                if self.use_forgetting:
                    phi = torch.sigmoid(self.forgetting_logit)
                    q_mf = (1.0 - phi) * q_mf
                if self.use_post_omission_reset and prev_omit:
                    reset_strength = torch.sigmoid(self.reset_logit)
                    q_mf = (1.0 - reset_strength) * q_mf

                # model-based value from model-free second-stage values
                max_q, _ = torch.max(q_mf[1:], dim=1)
                q_mb = transition_matrix @ max_q

                # dynamic arbitration (w_t)
                if self.use_dynamic_arbitration:
                    tau_t = torch.sigmoid(self.tau + self.tau_pe * prev_pe_abs + self.tau_post_omit * float(prev_omit))
                else:
                    tau_t = torch.sigmoid(self.tau)

                q_net = tau_t * q_mb.clone() + (1.0 - tau_t) * q_mf[0].clone()

                # ---------- stage 1 logits ----------
                if self.ignore_index != action_1[par, trial].item():
                    a1 = int(action_1[par, trial].item())
                    if prev_a1 is None:
                        action_repeat_1 = torch.zeros(2, device=action_1.device, dtype=torch.float32)
                    else:
                        action_repeat_1 = F.one_hot(
                            torch.tensor(prev_a1, device=action_1.device),
                            num_classes=2,
                        ).float()

                    raw_logits_1 = q_net + action_repeat_1 * stickiness
                    scaled_logits_1 = self.value_logits(raw_logits_1)

                    if self.use_lapse:
                        lapse1 = torch.sigmoid(self.lapse1_logit + self.lapse1_post_omit * float(prev_omit))
                        logits_out[par, trial, 0] = self._safe_log_probs_from_mixture(scaled_logits_1, lapse1)
                    else:
                        logits_out[par, trial, 0] = scaled_logits_1

                    # stickiness 使用“上一 trial 的一阶段动作”
                    prev_a1 = a1
                else:
                    # omission：保持与基线一致（该步 loss 会被 ignore_index 忽略）
                    prev_omit = True
                    continue

                # ---------- stage 2 logits ----------
                s2 = int(state[par, trial].item())
                raw_logits_2 = q_mf[s2 + 1].clone()

                if self.use_stage2_wsls:
                    rho_s = torch.tanh(self.rho2[s2])
                    kappa_s = torch.tanh(self.kappa2[s2])
                    if (not prev_omit) and (prev_s2 == s2) and (prev_a2 in (0, 1)):
                        rep = torch.zeros(2, device=action_1.device)
                        rep[prev_a2] = 1.0
                        raw_logits_2 = raw_logits_2 + rho_s * rep
                        if prev_r2 == 1.0:
                            raw_logits_2 = raw_logits_2 + kappa_s * rep

                scaled_logits_2 = self.value_logits(raw_logits_2)
                if self.use_lapse:
                    lapse2 = torch.sigmoid(self.lapse2_logit + self.lapse2_post_omit * float(prev_omit))
                    logits_out[par, trial, 1] = self._safe_log_probs_from_mixture(scaled_logits_2, lapse2)
                else:
                    logits_out[par, trial, 1] = scaled_logits_2

                # ---------- learning updates ----------
                omit2 = self.ignore_index == action_2[par, trial].item()
                if not omit2:
                    # adaptive alpha for volatility / change-point sensitivity
                    if self.use_adaptive_alpha:
                        alpha_t = torch.sigmoid(self.alpha + self.alpha_pe * prev_pe_abs)
                    else:
                        alpha_t = alpha_base

                    a2 = int(action_2[par, trial].item())

                    delta_1 = q_mf[s2 + 1, a2] - q_mf[0, a1]
                    delta_2 = reward[par, trial] - q_mf[s2 + 1, a2]
                    delta_2 = torch.nan_to_num(delta_2, nan=0.0)

                    # 重要：避免任何对参与反传的张量做 in-place 修改（否则会触发 version mismatch）
                    onehot_a1 = F.one_hot(torch.tensor(a1, device=action_1.device), num_classes=2).float()
                    onehot_a2 = F.one_hot(torch.tensor(a2, device=action_1.device), num_classes=2).float()

                    dq = torch.zeros_like(q_mf)
                    dq[0] = (alpha_t * delta_1 + lambd * alpha_t * delta_2) * onehot_a1
                    dq[s2 + 1] = (alpha_t * delta_2) * onehot_a2
                    q_mf = q_mf + dq

                    prev_pe_abs = torch.abs(delta_2.detach())
                    prev_omit = False
                    prev_a2 = a2
                    prev_s2 = s2
                    prev_r2 = float(reward[par, trial].item()) if not torch.isnan(reward[par, trial]) else 0.0
                else:
                    # 二阶段 omission：不更新，但作为事件影响下一 trial
                    prev_omit = True

        return logits_out

    def forward_one_step(self, data):
        # 当前 openloop 主要使用 two_step；保持接口但不实现额外逻辑
        return super().forward_one_step(data)


