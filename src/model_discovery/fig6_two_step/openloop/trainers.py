import torch
import torch.optim as optim
import torch.nn.functional as F
import schedulefree
from tqdm import tqdm

class Trainer:
    def __init__(self, model, num_iter=1000):
        self.model = model
        self.num_iter = num_iter
        # 检查模型是否有可训练参数
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if len(trainable_params) > 0:
            self.optimizer = schedulefree.AdamWScheduleFree(self.model.parameters(), lr=0.1)
            self.has_trainable_params = True
        else:
            self.optimizer = None
            self.has_trainable_params = False
        # store最近一次评估时的一些统计量，便于在外部计算信息准则
        self.last_nll = None              # 平均负对数似然（与原返回值一致）
        self.last_aic = None              # AIC
        self.last_num_params = None       # 自由参数个数 k
        self.last_num_observations = None # 有效观测数 N

    def fit_and_evaluate(self, train_df, eval_df):
        ### PREPROCESS DATA ###
        train_data, eval_data = self.model.preprocess_data(train_df, eval_df)

        ### FITTING ###
        if self.has_trainable_params:
            self.model.train()
            self.optimizer.train()
            for _ in tqdm(range(self.num_iter)):
                self.optimizer.zero_grad()
                logits = self.model(train_data)
                loss = F.cross_entropy(logits.flatten(0, -2), train_data['choice'].flatten().long())
                loss.backward()
                print(loss.item(), flush=True)
                self.optimizer.step()
        else:
            # 对于没有可训练参数的模型（如 RandomGuess），跳过训练步骤
            print("模型没有可训练参数，跳过训练步骤", flush=True)

        ### EVALUATION ###
        self.model.eval()
        if self.optimizer is not None:
            self.optimizer.eval()
        logits = self.model(eval_data)
        # 平均负对数似然（保持与原实现一致）
        ignore_index = getattr(self.model, "ignore_index", -100)
        flat_logits = logits.flatten(0, -2)
        flat_choices = eval_data["choice"].flatten().long()
        nll_mean = F.cross_entropy(
            flat_logits,
            flat_choices,
            ignore_index=ignore_index,
        )

        # 计算总负对数似然（仅在有效观测上求和），用于 AIC
        valid_mask = flat_choices != ignore_index
        if valid_mask.any():
            valid_logits = flat_logits[valid_mask]
            valid_choices = flat_choices[valid_mask]
            nll_sum = F.cross_entropy(
                valid_logits,
                valid_choices,
                reduction="sum",
            )
            num_observations = int(valid_choices.numel())
        else:
            # 极端情况：没有有效观测，此时无法定义 AIC，仅返回平均损失
            nll_sum = torch.tensor(0.0, device=logits.device)
            num_observations = 0

        # 计算参数个数 k
        num_params = 0
        for p in self.model.parameters():
            if p.requires_grad:
                num_params += p.numel()

        # 计算 AIC = 2k - 2 ln(L^)，其中 ln(L^) = - NLL_total
        # => AIC = 2k + 2 * NLL_total
        if num_observations > 0:
            aic = 2 * num_params + 2 * float(nll_sum.item())
        else:
            aic = None

        # 缓存统计量，便于在外部读取
        self.last_nll = nll_mean.detach()
        self.last_aic = aic
        self.last_num_params = num_params
        self.last_num_observations = num_observations

        return nll_mean
