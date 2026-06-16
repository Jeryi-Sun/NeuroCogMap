import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from models import RescorlaWagnerModel, DualSystems, RandomGuess
from improved_cog_model_cogneuromap_simple import DualSystemsHierarchicalBanditCogneuromapSimple as DualSystemsSimple
from improved_cog_model_cogneuromap_full import DualSystemsHierarchicalBanditCogneuromapFull as DualSystemsFull
from trainers import Trainer
import pandas as pd
import torch
from datasets import load_dataset

# 实验重复次数（用于误差分析）
NUM_REPEATS = 5

# 如果结果文件已存在，是否跳过当前运行（可根据需要修改）
SKIP_IF_RESULT_EXISTS = True


experiments = [
    #{'name': 'horizon1', 'agent': 'centaur', 'path': 'wilson2014humans/simulation0.csv', 'model': RescorlaWagnerModel(num_options=2)},
    #{'name': 'horizon1', 'agent': 'human', 'path': 'wilson2014humans/exp1.csv', 'model': RescorlaWagnerModel(num_options=2)},
    #{'name': 'horizon2', 'agent': 'centaur', 'path': 'wilson2014humans/simulation2.csv', 'model': RescorlaWagnerModel(num_options=2)},
    #{'name': 'horizon2', 'agent': 'human', 'path': 'wilson2014humans/exp3.csv', 'model': RescorlaWagnerModel(num_options=2)},
    #{'name': 'horizon3', 'agent': 'centaur', 'path': 'wilson2014humans/simulation3.csv', 'model': RescorlaWagnerModel(num_options=2)},
    #{'name': 'horizon3', 'agent': 'human', 'path': 'wilson2014humans/exp4.csv', 'model': RescorlaWagnerModel(num_options=2)},
    #{'name': 'horizon4', 'agent': 'centaur', 'path': 'wilson2014humans/simulation4.csv', 'model': RescorlaWagnerModel(num_options=2)},
    #{'name': 'horizon4', 'agent': 'human', 'path': 'wilson2014humans/exp5.csv', 'model': RescorlaWagnerModel(num_options=2)},
    #{'name': 'twostep1', 'agent': 'centaur', 'path': 'kool2016when/simulation.csv', 'model': DualSystems(variant='two_step')},
    # RandomGuess baseline 示例（已注释，需要时可取消注释）:
    #{'name': 'twostep1', 'agent': 'human', 'path': 'kool2016when/exp2.csv', 'model': RandomGuess(variant='two_step'), 'name_suffix': 'random_guess'},
    {'name': 'twostep1', 'agent': 'human', 'path': 'kool2016when/exp2.csv', 'model': DualSystems(variant='two_step'), 'name_suffix': 'baseline_fix_bug_v2'},
    {'name': 'twostep1', 'agent': 'human', 'path': 'kool2016when/exp2.csv', 'model': DualSystemsSimple(variant='two_step'), 'name_suffix': 'simple_fix_bug_v2'},
    {'name': 'twostep1', 'agent': 'human', 'path': 'kool2016when/exp2.csv', 'model': DualSystemsFull(variant='two_step'), 'name_suffix': 'full_fix_bug_v2'},
    #{'name': 'twostep2', 'agent': 'human', 'path': 'kool2017cost/exp2.csv', 'model': RandomGuess(variant='two_step'), 'name_suffix': 'random_guess'},
    {'name': 'twostep2', 'agent': 'human', 'path': 'kool2017cost/exp2.csv', 'model': DualSystems(variant='two_step'), 'name_suffix': 'baseline_fix_bug_v2'},
    {'name': 'twostep2', 'agent': 'human', 'path': 'kool2017cost/exp2.csv', 'model': DualSystemsSimple(variant='two_step'), 'name_suffix': 'simple_fix_bug_v2'},
    {'name': 'twostep2', 'agent': 'human', 'path': 'kool2017cost/exp2.csv', 'model': DualSystemsFull(variant='two_step'), 'name_suffix': 'full_fix_bug_v2'},
]

for repeat_id in range(1, NUM_REPEATS + 1):
    print(f"========== 第 {repeat_id} 次重复运行 ==========")

    for index in range(len(experiments)):
        data = []

        df = pd.read_csv(experiments[index]['path'])
        
        # 删除包含-1的行，并删除其配对行（current_state为(999.0, 0.0)或(999.0, 1.0)的对）
        # 配对关系：如果current_state=999.0，则下一行（相同participant和task，trial+1）是配对行
        # 如果current_state=0.0或1.0，则上一行（相同participant和task，trial-1，且current_state=999.0）是配对行
        
        # 首先找到所有包含-1的行（任何列包含-1或-1.0）
        rows_with_minus_one = set()
        for idx, row in df.iterrows():
            if (row == -1.0).any() or (row == -1).any():
                rows_with_minus_one.add(idx)
        
        # 找到需要删除的配对行
        rows_to_delete = set(rows_with_minus_one)
        
        # 按participant和task分组处理，确保配对关系正确
        for participant in df['participant'].unique():
            for task in df[df['participant'] == participant]['task'].unique():
                df_group = df[(df['participant'] == participant) & (df['task'] == task)].copy()
                df_group = df_group.sort_values('trial').reset_index()
                # 保存原始索引映射：新索引 -> 原始索引
                original_index_map = dict(zip(range(len(df_group)), df_group['index']))
                
                for i in range(len(df_group)):
                    row = df_group.iloc[i]
                    original_idx = original_index_map[i]
                    
                    # 如果这一行包含-1，需要找到并删除其配对行
                    if original_idx in rows_with_minus_one:
                        # 如果current_state是999.0，配对行是下一行（trial+1）
                        if row['current_state'] == 999.0:
                            if i + 1 < len(df_group):
                                next_original_idx = original_index_map[i + 1]
                                rows_to_delete.add(next_original_idx)
                        # 如果current_state是0.0或1.0，配对行是上一行（trial-1，且current_state=999.0）
                        elif row['current_state'] in [0.0, 1.0]:
                            if i > 0:
                                prev_row = df_group.iloc[i - 1]
                                if prev_row['current_state'] == 999.0:
                                    prev_original_idx = original_index_map[i - 1]
                                    rows_to_delete.add(prev_original_idx)
                        # 如果current_state是-1.0，配对行是上一行（trial-1，且current_state=999.0）
                        elif row['current_state'] == -1.0:
                            if i > 0:
                                prev_row = df_group.iloc[i - 1]
                                if prev_row['current_state'] == 999.0:
                                    prev_original_idx = original_index_map[i - 1]
                                    rows_to_delete.add(prev_original_idx)
        
        # 删除所有需要删除的行
        if rows_to_delete:
            print(f"删除 {len(rows_to_delete)} 行包含-1的数据及其配对行")
            df = df.drop(index=list(rows_to_delete)).reset_index(drop=True)

        # select human participants
        # if (('twostep' in experiments[index]['name']) and (experiments[index]['agent'] == 'human')) or (('horizon' in experiments[index]['name']) and (experiments[index]['agent'] == 'human')):
        #     # dataset = load_dataset("marcelbinz/Psych-101-test",cache_dir="/path/to/project_root/Human_LLM_align/Llama-3.1-Centaur-70B-main/openloop/psych_101",token=os.getenv("HF_TOKEN"))
        #     # eval_dataset = dataset['test'].filter(lambda example: example['experiment'].startswith(experiments[index]['path']))
        #     eval_participants = [22, 89, 92, 138]
        #     df = df[df['participant'].isin(eval_participants)]
        #     print(eval_participants)

        # match simulated data
        # if ('horizon' in experiments[index]['name']):
        #     df = df[df['participant'] < 100]
        #     df = df[df['task'] < 100]
        
        # 保存原始模型实例的引用，用于确定模型类型和参数
        original_model = experiments[index]['model']
        
        for participant in df['participant'].unique():
            print("experiments[index]['path']: ", experiments[index]['path'])
            df_participant = df[df['participant'] == participant]

            # 为每个participant创建新的模型实例，避免参数状态被污染
            # 这是修复nll固定在0.693问题的关键：每个participant都应该从初始参数开始优化
            if isinstance(original_model, RandomGuess):
                model = RandomGuess(variant=original_model.variant, num_options=original_model.num_options)
            elif isinstance(original_model, RescorlaWagnerModel):
                model = RescorlaWagnerModel(num_options=original_model.num_options)
            elif isinstance(original_model, DualSystems):
                model = DualSystems(variant=original_model.variant, store_data=original_model.store_data)
            elif isinstance(original_model, DualSystemsSimple):
                model = DualSystemsSimple(
                    variant=original_model.variant,
                    store_data=original_model.store_data,
                    enable_attention_lapse=original_model.enable_attention_lapse,
                    enable_history_sensitive_stickiness=original_model.enable_history_sensitive_stickiness,
                    enable_transition_learning=original_model.enable_transition_learning,
                    enable_dynamic_arbitration=original_model.enable_dynamic_arbitration,
                    enable_stage2_rule_policy=original_model.enable_stage2_rule_policy,
                    enable_stage2_wsls=original_model.enable_stage2_wsls,
                    enable_forgetting=original_model.enable_forgetting,
                )
            elif isinstance(original_model, DualSystemsFull):
                model = DualSystemsFull(
                    variant=original_model.variant,
                    store_data=original_model.store_data,
                    enable_attention_lapse=original_model.enable_attention_lapse,
                    enable_history_sensitive_stickiness=original_model.enable_history_sensitive_stickiness,
                    enable_transition_learning=original_model.enable_transition_learning,
                    enable_dynamic_arbitration=original_model.enable_dynamic_arbitration,
                    enable_stage2_rule_policy=original_model.enable_stage2_rule_policy,
                    enable_stage2_wsls=original_model.enable_stage2_wsls,
                    enable_forgetting=original_model.enable_forgetting,
                )
            else:
                # 如果无法识别模型类型，使用原始模型（可能会有问题，但至少不会报错）
                print(f"警告：无法识别模型类型 {type(original_model)}，使用原始模型实例（可能导致参数污染）")
                model = original_model

            trainer = Trainer(model)
            nll = trainer.fit_and_evaluate(df_participant, df_participant).item()
            aic = trainer.last_aic

            # 处理 RandomGuess 模型（没有可训练参数）
            if isinstance(model, RandomGuess):
                params = 0.0  # RandomGuess 模型没有参数
            elif ('horizon' in experiments[index]['name']):
                params = trainer.model.information_logits.beta.item()
            elif ('twostep' in experiments[index]['name']):
                params = torch.sigmoid(trainer.model.tau).item()
            else:
                params = 0.0  # 默认值
            
            if ('horizon' in experiments[index]['name']):
                data.append([participant, params, df_participant[df_participant['forced'] == 0]['reward'].mean(), nll, aic])
            elif ('twostep' in experiments[index]['name']):
                data.append([participant, params, df_participant['reward'].mean(), nll, aic])
            
        df = pd.DataFrame(data, columns=['participant', 'param', 'reward', 'nll', 'aic'])
        print(df)
        # 添加平均值行
        mean_row = pd.DataFrame({
            'participant': ['mean'],
            'param': [df['param'].mean()],
            'reward': [df['reward'].mean()],
            'nll': [df['nll'].mean()],
            'aic': [df['aic'].mean()]
        })
        df = pd.concat([df, mean_row], ignore_index=True)
        print(df)

        # 构造带有重复 id 的输出文件名
        output_path = (
            '/path/to/project_root/Human_LLM_align/'
            'Llama-3.1-Centaur-70B-main/openloop/results/repeat_results/'
            + experiments[index]['agent']
            + '_'
            + experiments[index]['path'].replace('/', '_')
            + '_'
            + experiments[index]['name_suffix']
            + f'_rep{repeat_id}.csv'
        )

        # 如果开启跳过功能且文件已存在，则直接跳过
        if SKIP_IF_RESULT_EXISTS and os.path.exists(output_path):
            print(f"结果文件已存在，跳过当前运行：{output_path}")
        else:
            df.to_csv(output_path)
