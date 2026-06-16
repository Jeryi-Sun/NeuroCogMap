import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from natsort import natsorted
import glob
import argparse
import os
import sys
import json
from pathlib import Path
from scipy.stats import pearsonr
from statsmodels.stats.multitest import fdrcorrection

try:
    from neurocogmap_release.paths import artifact_path, data_path, output_path
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from neurocogmap_release.paths import artifact_path, data_path, output_path

# 添加路径以导入 vendored LITcoder helpers.
LITCODER_CORE = Path(__file__).resolve().parents[3] / "brain_alignment" / "fig5_language_parcels" / "litcoder_core"
sys.path.insert(0, str(LITCODER_CORE))
from encoding.models.nested_cv import _calculate_correlations_pvalues, _create_full_cv_metrics_dict, _combine_pvalues_across_folds

LAYER_BASED_EXTRACTORS = ['language_model', 'bert_model', 'language_model_attention']
SAE_BASED_EXTRACTORS = ['sae_model', 'saeact_model', 'sae_model_pre_nozero_patch']
GENERAL_EXTRACTORS = ['embeddings']
SUPPORTED_EXTRACTORS = LAYER_BASED_EXTRACTORS + SAE_BASED_EXTRACTORS + GENERAL_EXTRACTORS

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run Ridge regression with specified parameters.')
    parser.add_argument('--model', type=str, default='google-gemma-2-2b', 
                       help='模型名称，用于构建文件路径（斜杠会被替换为横杠）')
    parser.add_argument('--extractor_type', type=str, 
                       choices=SUPPORTED_EXTRACTORS,
                       default='language_model',
                       help='提取器类型')
    parser.add_argument('--layer', type=int, default=None,
                       help='要使用的层索引（仅用于 language_model）')
    parser.add_argument('--parcel_id', type=int, default=None,
                       help='要使用的 Parcel ID（仅用于 sae_model 和 saeact_model）')
    parser.add_argument('--all_parcels', dest='all_parcels', action='store_true',
                       help='多 Parcel 模式：读取并拼接全部/指定 Parcel 特征（仅用于 sae_model / saeact_model / sae_model_pre_nozero_patch）')
    parser.add_argument('--single_parcel', dest='all_parcels', action='store_false',
                       help='单 Parcel 模式：仅使用 --parcel_id 指定的 Parcel（默认）')
    parser.set_defaults(all_parcels=False)
    parser.add_argument('--parcel_ids', type=str, default='',
                       help='多 Parcel 模式下可选：逗号分隔的 parcel id 列表。若留空，则从 --parcel_mapping_path 自动读取全部 parcel。')
    parser.add_argument('--parcel_mapping_path', type=str,
                       default=str(artifact_path("neurocogmap_atlas", "gemma2_2b", "latent_parcel_assignments.json")),
                       help='all_parcels 且未提供 --parcel_ids 时，用于读取全部 parcel 列表的 mapping 文件路径')
    parser.add_argument('--neural_data_csv', type=str,
                       default=str(data_path("model_discovery", "two_step", "neural", "schaefer_parcels_100.csv")),
                       help='two-step fMRI target CSV；默认使用 release 内 data/model_discovery/two_step/neural/schaefer_parcels_100.csv')
    parser.add_argument('--feature_cache_root', type=str, default=None,
                       help='预提取模型特征 cache 根目录；默认写入 NEUROCOGMAP_OUTPUT_DIR 或 /tmp/neurocogmap_release_outputs')
    parser.add_argument('--output_dir', type=str,
                       default=str(output_path("model_discovery", "fig6_two_step", "neural", "fits")),
                       help='拟合结果输出目录；默认写入 NEUROCOGMAP_OUTPUT_DIR 或 /tmp/neurocogmap_release_outputs')
    parser.add_argument('--roi', type=str, default=None, 
                       help='ROI to predict. 可以是单个 ROI 名称，或多个 ROI 名称用逗号分隔。例如: "ROI1" 或 "ROI1,ROI2,ROI3"。如果不提供此参数，将预测所有可用的 ROI。')
    parser.add_argument('--alpha_fdr', type=float, default=0.05,
                       help='FDR 校正的显著性水平 (默认: 0.05)')
    parser.add_argument('--participant_num', type=int, default=None,
                       help='在非测试模式下使用的参与者数量；不设置则使用所有参与者，设置为 N 则只使用前 N 个参与者')
    parser.add_argument('--test', action='store_true',
                       help='测试模式：如果开启，只加载前两个参与者进行测试')
    return parser.parse_args()

def _parse_parcel_ids(parcel_ids_str):
    parcel_ids = []
    if parcel_ids_str is None:
        return parcel_ids
    for token in parcel_ids_str.split(','):
        token = token.strip()
        if not token:
            continue
        parcel_ids.append(int(token))
    return sorted(set(parcel_ids))

def _load_all_parcel_ids_from_mapping(parcel_mapping_path):
    if not parcel_mapping_path:
        raise ValueError("all_parcels 模式下未提供 parcel_mapping_path")
    if not os.path.exists(parcel_mapping_path):
        raise FileNotFoundError(f"parcel_mapping_path 不存在: {parcel_mapping_path}")
    with open(parcel_mapping_path, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)
    parcel_to_latents = mapping_data.get("parcel_to_latents")
    if not isinstance(parcel_to_latents, dict) or not parcel_to_latents:
        raise ValueError("mapping 文件缺少 parcel_to_latents 或内容为空")
    parcel_ids = []
    for parcel_name in parcel_to_latents.keys():
        if isinstance(parcel_name, str) and parcel_name.startswith("parcel_"):
            parcel_ids.append(int(parcel_name.split("_")[-1]))
        else:
            raise ValueError(f"无法解析 parcel 名称: {parcel_name}")
    if not parcel_ids:
        raise ValueError("未从 mapping 文件解析到任何 parcel_id")
    return sorted(set(parcel_ids))

def _resolve_cache_dir(model, extractor_type, feature_cache_root=None):
    root = feature_cache_root or str(output_path("model_discovery", "fig6_two_step", "neural", "feature_cache"))
    model_safe = model.replace('/', '-')
    return os.path.join(root, model_safe, f'cache_{extractor_type}')

def build_feature_file_path(model, extractor_type, participant_id, layer=None, parcel_id=None, feature_cache_root=None):
    """
    根据参数构建特征文件路径
    
    Args:
        model: 模型名称（斜杠会被替换为横杠）
        extractor_type: 提取器类型
        participant_id: 参与者 ID
        layer: 层索引（仅用于 language_model）
        parcel_id: Parcel ID（仅用于 sae_model 和 saeact_model）
    
    Returns:
        文件路径
    """
    model_safe = model.replace('/', '-')
    cache_dir = _resolve_cache_dir(model, extractor_type, feature_cache_root)
    
    if extractor_type in LAYER_BASED_EXTRACTORS:
        if layer is None:
            raise ValueError(f"{extractor_type} 需要指定 --layer 参数")
        sub_dir = f'layer_{layer}'  # 目录名使用下划线
        filename = f'model={model_safe}_extractor={extractor_type}_layer={layer}_participant={participant_id}.pth'  # 文件名使用等号匹配实际格式
    elif extractor_type in SAE_BASED_EXTRACTORS:
        if parcel_id is None:
            raise ValueError(f"{extractor_type} 需要指定 --parcel_id 参数")
        sub_dir = f'parcel_{parcel_id}'
        filename = f'model={model_safe}_extractor={extractor_type}_{sub_dir}_participant={participant_id}.pth'
    elif extractor_type in GENERAL_EXTRACTORS:
        sub_dir = "general"
        filename = f'model={model_safe}_extractor={extractor_type}_{sub_dir}_participant={participant_id}.pth'
    else:
        raise ValueError(f"不支持的提取器类型: {extractor_type}")
    
    return os.path.join(cache_dir, sub_dir, filename)

def _load_feature_array(file_path, extractor_type, layer=None, parcel_id=None):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"特征文件不存在: {file_path}")
    features = torch.load(file_path)
    if isinstance(features, torch.Tensor):
        return features.numpy()
    if isinstance(features, dict):
        if extractor_type in LAYER_BASED_EXTRACTORS and layer is not None:
            if layer not in features:
                raise KeyError(f"字典中不存在层 {layer}，可用键: {list(features.keys())}")
            return features[layer].numpy()
        if extractor_type in SAE_BASED_EXTRACTORS and parcel_id is not None:
            candidate_keys = [parcel_id, str(parcel_id), f"parcel_{parcel_id}"]
            for key in candidate_keys:
                if key in features:
                    value = features[key]
                    return value.numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
            raise KeyError(
                f"字典中不存在 parcel_id {parcel_id}（尝试键: {candidate_keys}），可用键示例: {list(features.keys())[:10]}"
            )
        raise ValueError(f"无法从字典格式中提取特征: extractor_type={extractor_type}, layer={layer}, parcel_id={parcel_id}")
    raise ValueError(f"无法解析特征文件格式: {file_path}，类型: {type(features)}")

def load_features(model, extractor_type, participants, layer=None, parcel_id=None, parcel_ids=None, all_parcels=False, feature_cache_root=None):
    """
    加载所有参与者的特征
    
    Args:
        model: 模型名称
        extractor_type: 提取器类型
        participants: 参与者 ID 列表（映射后的 ID，从 0 开始连续编号）
        layer: 层索引（仅用于 language_model）
        parcel_id: Parcel ID（单 Parcel 模式）
        parcel_ids: Parcel ID 列表（多 Parcel 模式）
        all_parcels: 是否启用多 Parcel 模式
    
    Returns:
        X: 特征列表，按参与者顺序排列
    """
    X = []
    if all_parcels:
        if extractor_type not in SAE_BASED_EXTRACTORS:
            raise ValueError(f"all_parcels 模式仅支持 SAE 类提取器，当前 extractor_type={extractor_type}")
        if not parcel_ids:
            raise ValueError("all_parcels 模式下 parcel_ids 为空，无法加载特征")

    for participant_id in participants:
        # 直接使用映射后的 ID 加载特征文件
        participant_id_str = str(participant_id)
        if all_parcels:
            participant_features = []
            expected_samples = None
            for pid in parcel_ids:
                file_path = build_feature_file_path(model, extractor_type, participant_id_str, layer, pid, feature_cache_root)
                feature_array = _load_feature_array(file_path, extractor_type, layer, pid)
                if feature_array.ndim != 2:
                    raise ValueError(f"特征矩阵必须是二维数组，实际 shape={feature_array.shape}, file={file_path}")
                if expected_samples is None:
                    expected_samples = feature_array.shape[0]
                elif feature_array.shape[0] != expected_samples:
                    raise ValueError(
                        f"参与者 {participant_id} 的 parcel 特征样本数不一致："
                        f"parcel_{pid} 样本数={feature_array.shape[0]}，期望={expected_samples}，file={file_path}"
                    )
                participant_features.append(feature_array)
            X.append(np.concatenate(participant_features, axis=1))
        else:
            file_path = build_feature_file_path(model, extractor_type, participant_id_str, layer, parcel_id, feature_cache_root)
            feature_array = _load_feature_array(file_path, extractor_type, layer, parcel_id)
            X.append(feature_array)
    
    return X

def main():
    args = parse_arguments()

    # process ROIs - 支持多个 ROI
    Y = pd.read_csv(args.neural_data_csv)
    
    # 获取所有参与者并重新编号为从 0 开始的连续编号
    all_participants_original = np.sort(Y['participant'].unique())
    
    print(f"总参与者数: {len(all_participants_original)}", flush=True)
    print(f"原始参与者 ID: {all_participants_original.tolist()}", flush=True)
    
    # 创建映射：原始 ID -> 新 ID (0, 1, 2, ...)
    participant_id_mapping = {orig_id: new_id for new_id, orig_id in enumerate(all_participants_original)}
    
    # 应用映射，将 participant ID 重新编号为从 0 开始的连续编号
    Y['participant'] = Y['participant'].map(participant_id_mapping)
    
    # 验证重新编号是否成功
    participants_new = np.sort(Y['participant'].unique())
    expected_participants = np.arange(len(all_participants_original))
    
    if not np.array_equal(participants_new, expected_participants):
        raise ValueError(f"参与者 ID 重新编号失败！期望: {expected_participants.tolist()}, 实际: {participants_new.tolist()}")
    
    print(f"重新编号后的参与者 ID: {participants_new.tolist()} (从 0 开始连续编号)", flush=True)
    print(f"ID 映射关系: {participant_id_mapping}", flush=True)
    
    # 测试模式：只使用前两个参与者
    if args.test:
        print(f"\n⚠️  测试模式已开启：只使用前两个参与者", flush=True)
        if len(participants_new) < 2:
            print(f"⚠️  警告：参与者数量 ({len(participants_new)}) 少于 2 个，测试模式将使用所有可用参与者", flush=True)
            selected_participants = participants_new
        else:
            selected_participants = participants_new[:2]
        # 过滤数据，只保留前两个参与者的数据
        Y = Y[Y['participant'].isin(selected_participants)].copy()
        # 更新参与者列表
        participants_new = np.sort(Y['participant'].unique())
        print(f"测试模式：参与者数量从 {len(all_participants_original)} 减少到 {len(participants_new)}", flush=True)
        print(f"测试模式：使用的参与者 ID: {participants_new.tolist()}", flush=True)
    # 非测试模式下，根据 participant_num 只使用前 N 个参与者
    elif args.participant_num is not None:
        if args.participant_num <= 0:
            raise ValueError(f"participant_num 必须为正整数，但得到 {args.participant_num}")
        if args.participant_num > len(participants_new):
            print(
                f"⚠️  警告：请求的参与者数量 participant_num={args.participant_num} 大于总参与者数 {len(participants_new)}，将使用全部参与者",
                flush=True,
            )
            selected_participants = participants_new
        else:
            selected_participants = participants_new[:args.participant_num]
        Y = Y[Y['participant'].isin(selected_participants)].copy()
        participants_new = np.sort(Y['participant'].unique())
        print(
            f"非测试模式：使用前 {len(participants_new)} 个参与者，ID: {participants_new.tolist()}",
            flush=True,
        )
    
    # 计算 block（在重新编号后）
    Y['block'] = pd.factorize(pd._libs.lib.fast_zip([Y.participant.values, Y.run_no.values]))[0]
    
    # 解析 ROI 列表（支持逗号分隔的多个 ROI，或使用所有 ROI）
    if args.roi is None:
        # 如果不指定 ROI，使用所有以 'X_b' 开头的列（只保留 Schaefer parcel，对应列名前缀为 X_b）
        roi_columns = [col for col in Y.columns if col.startswith('X_b')]
        if len(roi_columns) == 0:
            raise ValueError("数据中没有找到以 'X_b' 开头的 ROI 列（例如 X_b'7Networks_LH_Vis_1'）")
        roi_list = [col.replace('X_', '') for col in roi_columns]
        print(f"未指定 ROI，将预测所有可用的 ROI ({len(roi_list)} 个): {roi_list[:5]}{'...' if len(roi_list) > 5 else ''}", flush=True)
    else:
        # 解析指定的 ROI 列表
        roi_list = [roi.strip() for roi in args.roi.split(',')]
        print(f"预测指定的 ROI 列表 ({len(roi_list)} 个): {roi_list}", flush=True)
        roi_columns = []
        for roi in roi_list:
            # 只允许使用以 X_b 开头的 ROI 列
            roi_col = f'X_{roi}'
            if roi_col not in Y.columns:
                available_rois = [col.replace('X_', '') for col in Y.columns if col.startswith('X_b')]
                raise ValueError(f"ROI 列不存在或不是以 'X_b' 开头: {roi_col}。可用 ROI 示例: {available_rois[:10]}...")
            roi_columns.append(roi_col)
    
    # 构建目标列
    target_columns = ['participant', 'block', 'sub_trial_type'] + roi_columns
    Y = Y[target_columns]

    # 获取重新编号后的参与者列表（现在是从 0 开始的连续编号）
    participants = np.sort(Y['participant'].unique())
    if args.test:
        print(f"最终使用的参与者 ID (测试模式，重新编号后): {participants.tolist()}", flush=True)
    else:
        print(f"最终使用的参与者 ID (重新编号后): {participants.tolist()}", flush=True)
    
    # 验证参数
    if args.extractor_type in LAYER_BASED_EXTRACTORS and args.layer is None:
        raise ValueError(f"{args.extractor_type} 需要指定 --layer 参数")
    active_parcel_ids = None
    if args.extractor_type in SAE_BASED_EXTRACTORS:
        if args.all_parcels:
            parsed_parcel_ids = _parse_parcel_ids(args.parcel_ids)
            active_parcel_ids = parsed_parcel_ids if parsed_parcel_ids else _load_all_parcel_ids_from_mapping(args.parcel_mapping_path)
            print(f"  all_parcels=True, parcel 数量={len(active_parcel_ids)}, 前10个={active_parcel_ids[:10]}", flush=True)
        elif args.parcel_id is None:
            raise ValueError(f"{args.extractor_type} 需要指定 --parcel_id 参数（或开启 --all_parcels）")

    # 加载特征
    # 使用映射后的 participant ID 直接加载特征文件
    print(f"加载特征: extractor_type={args.extractor_type}, model={args.model}")
    if args.extractor_type in LAYER_BASED_EXTRACTORS:
        print(f"  layer={args.layer}")
    elif args.extractor_type in GENERAL_EXTRACTORS:
        print("  general 特征（不使用 layer/parcel）")
    else:
        if args.all_parcels:
            print(f"  all_parcels=True, parcel 数量={len(active_parcel_ids)}")
        else:
            print(f"  parcel_id={args.parcel_id}")
    print(f"  使用映射后的 participant ID: {participants.tolist()}", flush=True)
    
    X = load_features(
        args.model,
        args.extractor_type,
        participants,
        layer=args.layer,
        parcel_id=args.parcel_id,
        parcel_ids=active_parcel_ids,
        all_parcels=args.all_parcels,
        feature_cache_root=args.feature_cache_root,
    )

    # run models
    all_results = run(X, Y, alpha_fdr=args.alpha_fdr, 
                     model=args.model, extractor_type=args.extractor_type, 
                     layer=args.layer, parcel_id=args.parcel_id, parcel_ids=active_parcel_ids,
                     all_parcels=args.all_parcels, feature_cache_root=args.feature_cache_root)
    
    # 构建输出文件名
    if args.extractor_type in LAYER_BASED_EXTRACTORS:
        output_suffix = f"layer={args.layer}"
    elif args.extractor_type in SAE_BASED_EXTRACTORS:
        if args.all_parcels:
            if active_parcel_ids is None:
                raise RuntimeError("all_parcels 模式下 active_parcel_ids 不应为空")
            if len(active_parcel_ids) <= 10:
                output_suffix = f"parcel_ids={'_'.join(map(str, active_parcel_ids))}"
            else:
                output_suffix = f"all_parcels_n={len(active_parcel_ids)}"
        else:
            output_suffix = f"parcel_id={args.parcel_id}"
    else:
        output_suffix = "general"
    
    # 使用第一个 ROI 作为文件名的一部分（如果只有一个 ROI），否则使用 "all_rois" 或 "multiple_rois"
    if len(roi_list) == 1:
        roi_name_for_file = roi_list[0]
    elif args.roi is None:
        roi_name_for_file = "all_rois"
    else:
        roi_name_for_file = "multiple_rois"
    
    # 如果开启测试模式，在文件名中添加标识
    test_suffix = "_test" if args.test else ""
    # 注意：model 名称中可能包含斜杠（例如 google/gemma-2-2b），需要替换为横杠以避免被解释为子目录
    model_safe_for_output = args.model.replace('/', '-')
    extractor_output_dir = os.path.join(args.output_dir, args.extractor_type)
    output_file = os.path.join(
        extractor_output_dir,
        f"model={model_safe_for_output}_extractor={args.extractor_type}_{output_suffix}_roi={roi_name_for_file}{test_suffix}.json",
    )
    os.makedirs(extractor_output_dir, exist_ok=True)
    
    # 保存为 JSON 格式（包含所有 metrics）
    # 确保所有 numpy 类型都被转换为 Python 原生类型
    def convert_to_serializable(obj):
        """递归转换 numpy 类型为 Python 原生类型"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_to_serializable(all_results)
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"结果已保存到: {output_file}")

def run(X, Y, alpha_fdr=0.05, model=None, extractor_type=None, layer=None, parcel_id=None, parcel_ids=None, all_parcels=False, feature_cache_root=None):
    """
    运行嵌套交叉验证，使用相关性评估和 FDR 校正
    
    Args:
        X: 特征列表，按参与者顺序排列
        Y: 目标 DataFrame，包含 participant, block, sub_trial_type 和 ROI 列
        alpha_fdr: FDR 校正的显著性水平
    
    Returns:
        包含所有参与者结果的字典
    """
    alphas = [10 ** c for c in range(0, 20)]
    participants = Y['participant'].unique()
    n_rois = len([col for col in Y.columns if col.startswith('X_')])

    # 存储所有参与者的结果
    all_participant_results = {}

    ### nested cross validation to identify optimal regularization values ###
    # 使用相关性而不是 R² 来选择最佳 alpha
    # 使用字典存储每个参与者的结果，因为不同参与者的 fold 数量可能不同
    corr_scores_dict = {}
    
    for participant_idx, participant_id in enumerate(participants):
        print(f"Participant {participant_idx} (ID: {participant_id})", flush=True)
        logo_outer = LeaveOneGroupOut()
        logo_inner = LeaveOneGroupOut()
        X_participant = X[participant_idx]
        Y_participant = Y[Y['participant'] == participant_id].copy()

        # 检查并对齐 X 和 Y 的样本数量
        n_samples_X = X_participant.shape[0]
        n_samples_Y = len(Y_participant)
        
        if n_samples_X != n_samples_Y:
            print(f"\n  ⚠️  错误: X 和 Y 的样本数量不一致！", flush=True)
            print(f"     - 特征文件 (X) 样本数: {n_samples_X}", flush=True)
            print(f"     - 数据文件 (Y) 样本数: {n_samples_Y}", flush=True)
            print(f"     - 差异: {abs(n_samples_X - n_samples_Y)} 个样本", flush=True)
            print(f"     - 参与者 ID: {participant_id}", flush=True)
            
            # 检查数据文件的列，看是否有可用于对齐的标识符
            print(f"     - Y 数据列: {list(Y_participant.columns)}", flush=True)
            
            # 检查是否有 sub_trial_type 或其他标识符
            if 'sub_trial_type' in Y_participant.columns:
                print(f"     - Y 数据中的 sub_trial_type 唯一值数量: {Y_participant['sub_trial_type'].nunique()}", flush=True)
                print(f"     - Y 数据中的 block 唯一值数量: {Y_participant['block'].nunique()}", flush=True)
            
            # 检查特征文件的路径和大小
            if model is not None and extractor_type is not None:
                participant_id_str = str(participant_id)
                if all_parcels and parcel_ids:
                    preview_ids = parcel_ids[:3]
                    for pid in preview_ids:
                        file_path = build_feature_file_path(model, extractor_type, participant_id_str, layer, pid, feature_cache_root)
                        if os.path.exists(file_path):
                            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                            print(f"     - 特征文件路径(预览): {file_path}", flush=True)
                            print(f"     - 特征文件大小(预览): {file_size:.2f} MB", flush=True)
                    print(f"     - 当前为 all_parcels 模式，总 parcel 数: {len(parcel_ids)}", flush=True)
                else:
                    file_path = build_feature_file_path(model, extractor_type, participant_id_str, layer, parcel_id, feature_cache_root)
                    if os.path.exists(file_path):
                        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                        print(f"     - 特征文件路径: {file_path}", flush=True)
                        print(f"     - 特征文件大小: {file_size:.2f} MB", flush=True)
            else:
                print(f"     - 请检查特征提取和数据准备流程是否一致", flush=True)
            
            raise ValueError(
                f"样本数量不一致，无法安全对齐。\n"
                f"特征文件 (X) 有 {n_samples_X} 个样本，但数据文件 (Y) 有 {n_samples_Y} 个样本。\n"
                f"这可能是由于：\n"
                f"  1. 特征提取时某些样本被过滤掉了\n"
                f"  2. 数据文件中有额外的样本\n"
                f"  3. 特征文件和数据文件来自不同的处理流程\n"
                f"请检查特征提取脚本和数据准备脚本，确保它们处理了相同数量的样本。"
            )

        # 对每个 ROI 进行 block-wise 标准化
        for column in Y_participant.columns:
            if column.startswith('X_'):
                print(f"  处理 {column}", flush=True)
                # 兼容较低版本 pandas：不使用 include_groups 参数
                Y_participant[column] = Y_participant.groupby('block', group_keys=False).apply(
                    lambda g: (g[column] - g[column].mean()) / g[column].std()
                )

        blocks_participant = Y_participant['block'].values
        Y_participant_values = Y_participant.drop(columns=['participant', 'block', 'sub_trial_type']).values

        print(f"  最终 X shape: {X_participant.shape}, Y shape: {Y_participant_values.shape}, blocks shape: {blocks_participant.shape}")
        
        # 再次检查样本数量是否一致
        if X_participant.shape[0] != Y_participant_values.shape[0] or X_participant.shape[0] != len(blocks_participant):
            raise ValueError(f"样本数量仍然不一致: X={X_participant.shape[0]}, Y={Y_participant_values.shape[0]}, blocks={len(blocks_participant)}")
        
        # 计算该参与者的 outer fold 数量
        outer_splits = list(logo_outer.split(X_participant, Y_participant_values, blocks_participant))
        n_outer_folds = len(outer_splits)
        print(f"  参与者 {participant_id} 有 {n_outer_folds} 个 outer folds", flush=True)
        
        # 先遍历一次，计算最大 inner fold 数量
        # LeaveOneGroupOut 的 fold 数量等于唯一组的数量
        max_inner_folds = 0
        for train_index_outer, _ in outer_splits:
            blocks_train = blocks_participant[train_index_outer]
            unique_blocks_train = len(np.unique(blocks_train))
            max_inner_folds = max(max_inner_folds, unique_blocks_train)
        
        print(f"  参与者 {participant_id} 最大 inner folds 数量: {max_inner_folds}", flush=True)
        
        # 为该参与者初始化存储数组（动态大小）
        corr_scores_participant = np.zeros((n_outer_folds, max_inner_folds, len(alphas), n_rois, 3))  # [outer_fold, inner_fold, alpha, roi, train/val/test]

        for fold_outer, (train_index_outer, test_index_outer) in enumerate(outer_splits):
            X_train, X_test = X_participant[train_index_outer], X_participant[test_index_outer]
            Y_train, Y_test = Y_participant_values[train_index_outer], Y_participant_values[test_index_outer]
            blocks_train, blocks_test = blocks_participant[train_index_outer], blocks_participant[test_index_outer]

            # 计算该 outer fold 的 inner fold 数量
            inner_splits = list(logo_inner.split(X_train, Y_train, blocks_train))
            n_inner_folds = len(inner_splits)

            for fold_inner, (train_index_inner, test_index_inner) in enumerate(inner_splits):
                X_train_inner, X_validation = X_train[train_index_inner], X_train[test_index_inner]
                Y_train_inner, Y_validation = Y_train[train_index_inner], Y_train[test_index_inner]

                scaler = StandardScaler()
                X_train_inner = scaler.fit_transform(X_train_inner)
                X_validation = scaler.transform(X_validation)
                X_test_scaled = scaler.transform(X_test)

                pca = PCA(n_components=0.95)
                X_train_inner = pca.fit_transform(X_train_inner)
                X_validation = pca.transform(X_validation)
                X_test_scaled = pca.transform(X_test_scaled)
                print(f"  Fold {fold_outer}-{fold_inner}: PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}", flush=True)
                if np.isnan(pca.explained_variance_ratio_.sum()):
                    print(f"  Fold {fold_outer}-{fold_inner}: PCA explained variance: {pca.explained_variance_ratio_}", flush=True)
                X_train_inner = torch.from_numpy(X_train_inner).float().to("cuda")
                X_validation = torch.from_numpy(X_validation).float().to("cuda")
                Y_train_inner = torch.from_numpy(Y_train_inner).float().to("cuda")
                Y_validation = torch.from_numpy(Y_validation).float().to("cuda")
                X_test_scaled = torch.from_numpy(X_test_scaled).float().to("cuda")
                # 检查 Y_test 是否已经是 Tensor
                if isinstance(Y_test, torch.Tensor):
                    Y_test = Y_test.float().to("cuda")
                else:
                    Y_test = torch.from_numpy(Y_test).float().to("cuda")

                A = X_train_inner.T @ X_train_inner
                I = torch.eye(A.shape[0]).to("cuda")
                c = X_train_inner.T @ Y_train_inner
                
                for alpha_idx, alpha in enumerate(alphas):
                    alpha_I = alpha * I
                    B = A + alpha_I
                    w = torch.linalg.solve(B, c)

                    Y_train_inner_pred = X_train_inner @ w
                    Y_validation_pred = X_validation @ w
                    Y_test_pred = X_test_scaled @ w

                    # 计算相关性（每个 ROI）
                    for roi_idx in range(n_rois):
                        # Train set
                        corr_train, _ = pearsonr(
                            Y_train_inner[:, roi_idx].detach().cpu().numpy(),
                            Y_train_inner_pred[:, roi_idx].detach().cpu().numpy()
                        )
                        corr_scores_participant[fold_outer, fold_inner, alpha_idx, roi_idx, 0] = 0.0 if np.isnan(corr_train) else corr_train
                        
                        # Validation set
                        corr_val, _ = pearsonr(
                            Y_validation[:, roi_idx].detach().cpu().numpy(),
                            Y_validation_pred[:, roi_idx].detach().cpu().numpy()
                        )
                        corr_scores_participant[fold_outer, fold_inner, alpha_idx, roi_idx, 1] = 0.0 if np.isnan(corr_val) else corr_val
                        
                        # Test set
                        corr_test, _ = pearsonr(
                            Y_test[:, roi_idx].detach().cpu().numpy(),
                            Y_test_pred[:, roi_idx].detach().cpu().numpy()
                        )
                        corr_scores_participant[fold_outer, fold_inner, alpha_idx, roi_idx, 2] = 0.0 if np.isnan(corr_test) else corr_test
        
        # 存储该参与者的结果
        # 由于不同 outer fold 可能有不同数量的 inner fold，我们需要存储实际使用的部分
        # 但为了简化，我们存储整个数组（未使用的部分为 0）
        corr_scores_dict[participant_id] = corr_scores_participant

    ### refit with optimal regularization values ###
    # 为每个参与者存储结果
    for participant_idx, participant_id in enumerate(participants):
        print(f"\n处理参与者 {participant_idx} (ID: {participant_id}) 的最终拟合...", flush=True)
        
        # 选择最佳 alpha（基于验证集的相关性，对每个 ROI 分别选择）
        participant_corr_val = corr_scores_dict[participant_id][:, :, :, :, 1]  # [outer_fold, inner_fold, alpha, roi]
        # 对每个 outer_fold 和每个 ROI，选择最佳 alpha
        mean_corr_val = participant_corr_val.mean(axis=1)  # [outer_fold, alpha, roi]
        best_alphas_per_fold_roi = mean_corr_val.argmax(axis=1)  # [outer_fold, roi]
        
        print(f"  最佳 alpha 索引 (每个 outer fold 和 ROI):\n{best_alphas_per_fold_roi}", flush=True)
        
        logo_outer = LeaveOneGroupOut()
        X_participant = X[participant_idx]
        Y_participant = Y[Y['participant'] == participant_id].copy()

        # 检查并对齐 X 和 Y 的样本数量
        n_samples_X = X_participant.shape[0]
        n_samples_Y = len(Y_participant)
        
        if n_samples_X != n_samples_Y:
            print(f"\n  ⚠️  错误: X 和 Y 的样本数量不一致！", flush=True)
            print(f"     - 特征文件 (X) 样本数: {n_samples_X}", flush=True)
            print(f"     - 数据文件 (Y) 样本数: {n_samples_Y}", flush=True)
            print(f"     - 差异: {abs(n_samples_X - n_samples_Y)} 个样本", flush=True)
            print(f"     - 参与者 ID: {participant_id}", flush=True)
            
            # 检查数据文件的列，看是否有可用于对齐的标识符
            print(f"     - Y 数据列: {list(Y_participant.columns)}", flush=True)
            
            # 检查是否有 sub_trial_type 或其他标识符
            if 'sub_trial_type' in Y_participant.columns:
                print(f"     - Y 数据中的 sub_trial_type 唯一值数量: {Y_participant['sub_trial_type'].nunique()}", flush=True)
                print(f"     - Y 数据中的 block 唯一值数量: {Y_participant['block'].nunique()}", flush=True)
            
            # 检查特征文件的路径和大小
            if model is not None and extractor_type is not None:
                participant_id_str = str(participant_id)
                if all_parcels and parcel_ids:
                    preview_ids = parcel_ids[:3]
                    for pid in preview_ids:
                        file_path = build_feature_file_path(model, extractor_type, participant_id_str, layer, pid, feature_cache_root)
                        if os.path.exists(file_path):
                            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                            print(f"     - 特征文件路径(预览): {file_path}", flush=True)
                            print(f"     - 特征文件大小(预览): {file_size:.2f} MB", flush=True)
                    print(f"     - 当前为 all_parcels 模式，总 parcel 数: {len(parcel_ids)}", flush=True)
                else:
                    file_path = build_feature_file_path(model, extractor_type, participant_id_str, layer, parcel_id, feature_cache_root)
                    if os.path.exists(file_path):
                        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                        print(f"     - 特征文件路径: {file_path}", flush=True)
                        print(f"     - 特征文件大小: {file_size:.2f} MB", flush=True)
            else:
                print(f"     - 请检查特征提取和数据准备流程是否一致", flush=True)
            
            raise ValueError(
                f"样本数量不一致，无法安全对齐。\n"
                f"特征文件 (X) 有 {n_samples_X} 个样本，但数据文件 (Y) 有 {n_samples_Y} 个样本。\n"
                f"这可能是由于：\n"
                f"  1. 特征提取时某些样本被过滤掉了\n"
                f"  2. 数据文件中有额外的样本\n"
                f"  3. 特征文件和数据文件来自不同的处理流程\n"
                f"请检查特征提取脚本和数据准备脚本，确保它们处理了相同数量的样本。"
            )

        # 对每个 ROI 进行 block-wise 标准化
        for column in Y_participant.columns:
            if column.startswith('X_'):
                # 兼容较低版本 pandas：不使用 include_groups 参数
                Y_participant[column] = Y_participant.groupby('block', group_keys=False).apply(
                    lambda g: (g[column] - g[column].mean()) / g[column].std()
                )

        blocks_participant = Y_participant['block'].values
        roi_column_names = [col for col in Y_participant.columns if col.startswith('X_')]
        Y_participant_values = Y_participant[roi_column_names].values
        
        # 再次检查样本数量是否一致
        if X_participant.shape[0] != Y_participant_values.shape[0] or X_participant.shape[0] != len(blocks_participant):
            raise ValueError(f"样本数量仍然不一致: X={X_participant.shape[0]}, Y={Y_participant_values.shape[0]}, blocks={len(blocks_participant)}")

        # 存储每个 fold 的结果
        fold_correlations = []
        fold_pvalues = []
        fold_valphas = []
        fold_weights = []
        fold_significant_masks = []

        for fold_outer, (train_index_outer, test_index_outer) in enumerate(logo_outer.split(X_participant, Y_participant_values, blocks_participant)):
            X_train, X_test = X_participant[train_index_outer], X_participant[test_index_outer]
            Y_train, Y_test = Y_participant_values[train_index_outer], Y_participant_values[test_index_outer]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            pca = PCA(n_components=0.95)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

            X_train = torch.from_numpy(X_train).float().to("cuda")
            Y_train = torch.from_numpy(Y_train).float().to("cuda")
            X_test = torch.from_numpy(X_test).float().to("cuda")
            # 检查 Y_test 是否已经是 Tensor
            if isinstance(Y_test, torch.Tensor):
                Y_test = Y_test.float().to("cuda")
            else:
                Y_test = torch.from_numpy(Y_test).float().to("cuda")

            A = X_train.T @ X_train
            I = torch.eye(A.shape[0]).to("cuda")
            c = X_train.T @ Y_train

            # 为每个 fold 选择最佳 alpha
            # 由于 Ridge 回归使用共享的权重矩阵，我们需要为所有 ROI 选择一个 alpha
            # 我们使用该 fold 中所有 ROI 的最佳 alpha 的中位数索引
            best_alpha_idx = int(np.median(best_alphas_per_fold_roi[fold_outer, :]))
            best_alpha = alphas[best_alpha_idx]
            
            alpha_I = best_alpha * I
            B = A + alpha_I
            w = torch.linalg.solve(B, c)
            fold_weights.append(w.detach().cpu().numpy())
            fold_valphas.append([best_alpha] * n_rois)

            Y_test_pred = X_test @ w

            # 使用 _calculate_correlations_pvalues 计算相关性和 p 值
            y_test_np = Y_test.detach().cpu().numpy()
            y_pred_np = Y_test_pred.detach().cpu().numpy()
            
            correlations, pvalues = _calculate_correlations_pvalues(y_test_np, y_pred_np)
            fold_correlations.append(correlations)
            fold_pvalues.append(pvalues)

            # 应用 FDR 校正
            significant, corrected_pvals = fdrcorrection(pvalues, alpha=alpha_fdr)
            fold_significant_masks.append(significant)
            
            print(f"  Fold {fold_outer+1}: 中位数相关性 = {np.median(correlations):.3f}, "
                  f"显著 ROI = {np.sum(significant)}/{len(significant)}", flush=True)

        # 计算跨 fold 的平均相关性
        all_correlations = np.mean(fold_correlations, axis=0)  # 平均跨 folds

        # 使用 Fisher 方法合并 p 值
        all_pvalues = _combine_pvalues_across_folds(fold_pvalues, logger=None)

        # 对合并的 p 值应用 FDR 校正
        significant_mask, corrected_pvalues = fdrcorrection(all_pvalues, alpha=alpha_fdr)
        n_significant = np.sum(significant_mask)

        # 计算多数投票的显著 ROI
        significance_counts = np.sum(fold_significant_masks, axis=0)
        majority_significant_mask = significance_counts >= (len(fold_significant_masks) // 2 + 1)
        n_majority_significant = np.sum(majority_significant_mask)

        # 计算平均最佳 alpha
        mean_valphas = np.mean(fold_valphas, axis=0)

        # 注意：不计算平均权重，因为不同 fold 的 PCA 降维后特征数量可能不同
        # 导致权重矩阵形状不一致，无法直接求平均
        # 如果需要权重信息，可以存储 fold_weights 列表，但不求平均
        # mean_weights = np.mean(fold_weights, axis=0)  # 已移除：形状不一致

        # 创建 metrics 字典
        metrics = _create_full_cv_metrics_dict(
            all_correlations,
            all_pvalues,
            corrected_pvalues,
            significant_mask,
            majority_significant_mask,
            mean_valphas,
            n_significant,
            n_majority_significant,
        )

        # 添加 ROI 名称信息
        metrics['roi_names'] = [col.replace('X_', '') for col in roi_column_names]
        metrics['participant_id'] = str(participant_id)

        all_participant_results[f'participant_{participant_id}'] = metrics

        print(f"\n参与者 {participant_id} 最终结果:")
        print(f"  中位数相关性: {metrics['median_score']:.3f}")
        print(f"  显著 ROI (Fisher 方法): {n_significant}/{len(all_correlations)} ({metrics['percent_significant']:.1f}%)")
        print(f"  显著 ROI (多数投票): {n_majority_significant}/{len(all_correlations)} ({metrics['percent_majority_significant']:.1f}%)")
        if 'median_significant_score' in metrics:
            print(f"  显著 ROI 的中位数相关性: {metrics['median_significant_score']:.3f}")

    return all_participant_results

if __name__ == "__main__":
    main()
