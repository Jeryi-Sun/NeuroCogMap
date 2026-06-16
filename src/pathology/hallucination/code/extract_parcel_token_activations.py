#!/usr/bin/env python3
import argparse
import json
import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
import sys
from typing import Dict, List, Tuple, Union, DefaultDict, Optional
from collections import defaultdict
import re

import torch
from sae_lens import SAE, HookedSAETransformer
from tqdm import tqdm

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract token-level SAE activations and aggregate to Parcels for correct/incorrect splits."
        )
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="/path/to/project_root/safety_explanation/hallucination/results",
        help="Root directory where <dataset>_<model>/correct.jsonl and incorrect.jsonl are stored.",
    )
    parser.add_argument(
        "--combo-name",
        type=str,
        required=True,
        help="Combined name: <dataset>_<model>, e.g., dolly_close_gemma-2-2b",
    )
    parser.add_argument(
        "--parcel-mapping",
        type=str,
        required=True,
        help="Absolute path to latent_parcel_topsamples_functionality_analysis.json",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="both",
        choices=["correct", "incorrect", "both"],
        help="Which split(s) to process.",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="jsonl",
        choices=["jsonl", "npz"],
        help="Output storage format.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip processing when target outputs already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned actions without writing.",
    )
    # Model / SAE related
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Optional HF model id. Default inferred from combo-name suffix after last underscore.",
    )
    parser.add_argument(
        "--sae-release",
        type=str,
        default="gemma-scope-2b-pt-res",
        help="SAE release id.",
    )
    parser.add_argument(
        "--sae-local-base-dir",
        type=str,
        default="/path/to/local_models/gemma-scope-2b-pt-res",
        help="Local base dir where layer_x/.../params.npz exist.",
    )
    parser.add_argument(
        "--sae-paths",
        type=str,
        default="",
        help="Comma-separated SAE paths relative to release (e.g., layer_0/width_16k/average_l0_105,layer_1/...). If empty, use 2B-PT defaults.",
    )
    parser.add_argument(
        "--layers-per-batch",
        type=int,
        default=8,
        help="How many SAE layers to load per batch.",
    )
    parser.add_argument(
        "--n-devices",
        type=int,
        default=1,
        help="Number of devices for model (passed to HookedSAETransformer).",
    )
    parser.add_argument(
        "--baseline-direct-neural",
        action="store_true",
        help="Use direct neural activations baseline instead of SAE activations.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Max tokenizer length for prompts.",
    )
    parser.add_argument(
        "--aggregate-mode",
        type=str,
        default="sum",
        choices=["sum", "mean"],
        help="How to aggregate across layers for the same token/parcel.",
    )
    parser.add_argument(
        "--is_instruct",
        action="store_true",
        help="Whether the model is an instruct/chat model; use chat template to format prompt.",
    )
    parser.add_argument(
        "--sae_paths",
        type=str,
        default="",
        help="Comma-separated SAE paths relative to release (e.g., layer_0/width_16k/average_l0_105,layer_1/...). If empty, use 2B-PT defaults.",
    )
    return parser.parse_args()


class ParcelMapping:
    def __init__(self, latent_parcel_assignments: Dict, available_layers: Optional[List[int]] = None, latents_per_layer: int = 16384):
        self.latent_parcel_assignments = latent_parcel_assignments
        self.num_parcels = self._infer_num_parcels()
        self.available_layers = available_layers or []
        self.latents_per_layer = latents_per_layer

    def _infer_num_parcels(self) -> int:
        try:
            return len(self.latent_parcel_assignments.get("parcel_to_latents", {}))
        except Exception as ex:
            import traceback
            print(f"[ERROR] Infer num parcels failed: {ex}", file=sys.stderr)
            print(f"[ERROR] Exception type: {type(ex).__name__}", file=sys.stderr)
            print(f"[ERROR] Full traceback:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return 0

    def set_available_layers(self, available_layers: List[int]) -> None:
        """设置可用的SAE层号列表"""
        self.available_layers = sorted(available_layers)
        print(f"[INFO] 设置可用SAE层: {self.available_layers}")

    def validate_parcel_indices(self) -> bool:
        """验证 Parcel 索引的连续性和正确性"""
        parcel_to_latents = self.latent_parcel_assignments.get('parcel_to_latents', {})
        found_indices = set()
        
        for parcel_name in parcel_to_latents.keys():
            try:
                parcel_idx = int(parcel_name.split('_')[-1])
                found_indices.add(parcel_idx)
            except Exception:
                print(f"[WARN] 无法解析 Parcel 名称: {parcel_name}", file=sys.stderr)
                continue
        
        # 检查索引是否连续且从0开始
        expected_indices = set(range(self.num_parcels))
        missing_indices = expected_indices - found_indices
        extra_indices = found_indices - expected_indices
        
        if missing_indices:
            print(f"[WARN] 缺少 Parcel 索引: {sorted(missing_indices)}", file=sys.stderr)
        if extra_indices:
            print(f"[WARN] 超出范围的 Parcel 索引: {sorted(extra_indices)}", file=sys.stderr)
        
        is_valid = len(missing_indices) == 0 and len(extra_indices) == 0
        if is_valid:
            print(f"[INFO] Parcel 索引验证通过: 0-{self.num_parcels-1} 连续且完整")
        else:
            print(f"[ERROR] Parcel 索引验证失败: 期望 0-{self.num_parcels-1}，实际找到 {sorted(found_indices)}", file=sys.stderr)
        
        return is_valid

    def get_parcel_latent_mapping(self, parcel_input: Union[str, List[str]]) -> List[Tuple[int, int]]:
        if isinstance(parcel_input, str):
            parcel_names = [parcel_input]
        elif isinstance(parcel_input, list):
            parcel_names = parcel_input
        else:
            raise TypeError(f"parcel_input 必须是字符串或列表，当前类型: {type(parcel_input)}")

        mapping: List[Tuple[int, int]] = []
        
        # 如果没有设置可用层，使用旧的连续层映射方式
        if not self.available_layers:
            for parcel_name in parcel_names:
                if parcel_name not in self.latent_parcel_assignments['parcel_to_latents']:
                    raise KeyError(f"未找到 parcel: {parcel_name} 于 parcel_to_latents 映射中")
                latent_ids = self.latent_parcel_assignments['parcel_to_latents'][parcel_name]
                for latent_id in latent_ids:
                    layer_id = latent_id // self.latents_per_layer
                    latent_in_layer = latent_id % self.latents_per_layer
                    mapping.append((layer_id, latent_in_layer))
            return mapping
        
        # 使用实际SAE层号进行映射
        for parcel_name in parcel_names:
            if parcel_name not in self.latent_parcel_assignments['parcel_to_latents']:
                raise KeyError(f"未找到 parcel: {parcel_name} 于 parcel_to_latents 映射中")
            latent_ids = self.latent_parcel_assignments['parcel_to_latents'][parcel_name]
            for latent_id in latent_ids:
                # 先通过latent_id计算出原始的层索引
                original_layer_idx = latent_id // self.latents_per_layer
                
                # 检查原始层索引是否在可用层范围内
                if original_layer_idx < len(self.available_layers):
                    actual_layer_id = self.available_layers[original_layer_idx]
                else:
                    # 如果超出范围，使用模运算回退到可用层
                    layer_index = original_layer_idx % len(self.available_layers)
                    actual_layer_id = self.available_layers[layer_index]
                
                latent_in_layer = latent_id % self.latents_per_layer
                mapping.append((actual_layer_id, latent_in_layer))
        return mapping

    def build_latent_to_parcels_index(self) -> DefaultDict[Tuple[int, int], List[int]]:
        latent_to_parcels: DefaultDict[Tuple[int, int], List[int]] = defaultdict(list)
        parcel_to_latents = self.latent_parcel_assignments.get('parcel_to_latents', {})
        
        # 确保 Parcel 按索引顺序处理，保证输出数组的顺序正确
        parcel_items = sorted(parcel_to_latents.items(), key=lambda x: int(x[0].split('_')[-1]) if x[0].split('_')[-1].isdigit() else float('inf'))
        
        for parcel_name, latent_ids in parcel_items:
            try:
                parcel_idx = int(parcel_name.split('_')[-1])
                # 验证 Parcel 索引是否在有效范围内
                if parcel_idx < 0 or parcel_idx >= self.num_parcels:
                    print(f"[WARN] Parcel 索引 {parcel_idx} 超出范围 [0, {self.num_parcels-1}]，跳过 {parcel_name}", file=sys.stderr)
                    continue
            except Exception as ex:
                import traceback
                print(f"[WARN] 无法解析 Parcel 名称 {parcel_name}: {ex}，跳过", file=sys.stderr)
                print(f"[WARN] Exception type: {type(ex).__name__}", file=sys.stderr)
                print(f"[WARN] Full traceback:", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                continue
                
            for latent_id in latent_ids:
                if not self.available_layers:
                    # 使用旧的连续层映射方式
                    layer_id = latent_id // self.latents_per_layer
                    latent_in_layer = latent_id % self.latents_per_layer
                else:
                    # 使用实际SAE层号进行映射
                    original_layer_idx = latent_id // self.latents_per_layer
                    if original_layer_idx < len(self.available_layers):
                        layer_id = self.available_layers[original_layer_idx]
                    else:
                        # 如果超出范围，使用模运算回退到可用层
                        layer_index = original_layer_idx % len(self.available_layers)
                        layer_id = self.available_layers[layer_index]
                    latent_in_layer = latent_id % self.latents_per_layer
                latent_to_parcels[(layer_id, latent_in_layer)].append(parcel_idx)
        return latent_to_parcels


def load_parcel_mapping(mapping_path: str, available_layers: Optional[List[int]] = None, latents_per_layer: int = 16384) -> ParcelMapping:
    try:
        with open(mapping_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return ParcelMapping(latent_parcel_assignments=data, available_layers=available_layers, latents_per_layer=latents_per_layer)
    except Exception as ex:
        import traceback
        print(f"[ERROR] Failed to load parcel mapping: {ex}", file=sys.stderr)
        print(f"[ERROR] Exception type: {type(ex).__name__}", file=sys.stderr)
        print(f"[ERROR] Full traceback:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise


def ensure_dir(path: str, dry_run: bool = False) -> None:
    if dry_run:
        print(f"[DRY-RUN] mkdir -p {path}")
        return
    os.makedirs(path, exist_ok=True)


def get_input_paths(results_root: str, combo_name: str, split: str) -> List[Tuple[str, str]]:
    combo_dir = os.path.join(results_root, combo_name)
    paths: List[Tuple[str, str]] = []
    if split in ("correct", "both"):
        paths.append(("correct", os.path.join(combo_dir, "correct.jsonl")))
    if split in ("incorrect", "both"):
        paths.append(("incorrect", os.path.join(combo_dir, "incorrect.jsonl")))
    if split in ("both") and os.path.exists(os.path.join(combo_dir, "anticorrect.jsonl")):
        paths.append(("anticorrect", os.path.join(combo_dir, "anticorrect.jsonl")))
    return paths


def output_dir_for_split(results_root: str, combo_name: str, split: str) -> str:
    return os.path.join(results_root, combo_name, "parcels_token_acts", split)


def infer_model_from_combo(combo_name: str) -> Optional[str]:
    # Split on last underscore: dataset may contain underscores
    if "_" not in combo_name:
        return None
    parts = combo_name.rsplit("_", 1)
    return parts[1] if len(parts) == 2 else None


def detect_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_latents_per_layer(model_name: str, sae_release: str) -> int:
    """
    根据模型名称和 SAE release 自动选择每层 latent 数。
    支持 Gemma (16384) 和 Llama-3.1-8B + LXR-8x (32768)。
    """
    ml = model_name.lower()
    rl = sae_release.lower()
    if "gemma" in ml or "gemma" in rl:
        return 16384
    if "llama" in ml or "llama" in rl or "lxr" in rl:
        return 32768
    # 默认回退 16384，并打印 warning
    import warnings
    warnings.warn(f"未知模型/SAE 组合 (model={model_name}, release={sae_release})，使用默认 latents_per_layer=16384")
    return 16384


def extract_layer_num(path_str: str, sae_release: str) -> Optional[int]:
    """
    从 SAE 路径中解析层号，根据 sae_release 自动区分 Gemma 和 Llama 格式。
    """
    if "gemma" in sae_release.lower():
        # Gemma: layer_9/width_16k/...
        m = re.search(r"layer_(\d+)/", path_str)
        return int(m.group(1)) if m else None
    else:
        # Llama LXR-8x: l0r_8x,l1r_8x
        m = re.search(r"l(\d+)r_8x", path_str)
        return int(m.group(1)) if m else None


def load_model(model_name: str, device: str, n_devices: int) -> HookedSAETransformer:
    try:
        if "9b" in model_name:
            return HookedSAETransformer.from_pretrained(model_name, device=device, dtype=torch.float16, n_devices=n_devices)
        return HookedSAETransformer.from_pretrained(model_name, device=device, dtype=torch.float16)
    except Exception as ex:
        import traceback
        print(f"[ERROR] Failed to load model {model_name}: {ex}", file=sys.stderr)
        print(f"[ERROR] Exception type: {type(ex).__name__}", file=sys.stderr)
        print(f"[ERROR] Full traceback:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise


def load_sae_for_layers(layer_paths: List[str], sae_release: str, sae_local_base_dir: str, device: str, n_devices: int = 1, force_device: Optional[str] = None):
    """
    兼容 Gemma 与 Llama Scope LXR-8x:
    1. 优先使用本地 Llama Scope final.safetensors
    2. 其次尝试 Gemma 本地 params.npz
    3. 最后退回到 SAE.from_pretrained 默认下载/缓存
    """
    sae_list = []
    hook_names = []
    for path in layer_paths:
        # 检查是否是 Llama Scope LXR-8x 格式 (l0r_8x, l1r_8x, ...)
        m = re.search(r"l(\d+)r", path)
        if m:
            sae_id_num = m.group(1)
            real_path = f"Llama3_1-8B-Base-L{sae_id_num}R-8x"
            llama_scope_local_path = os.path.join(
                sae_local_base_dir, real_path, "checkpoints", "final.safetensors"
            )
        else:
            real_path = path
            llama_scope_local_path = None
        
        gemma_local_path = os.path.join(
            sae_local_base_dir, path, "params.npz"
        )
        
        try:
            if llama_scope_local_path and os.path.exists(llama_scope_local_path):
                # 优先使用本地 Llama Scope final.safetensors
                sae, cfg_dict, sparsity = SAE.from_pretrained(
                    release=sae_release,
                    sae_id=path,
                    device=device,
                    local_path=llama_scope_local_path,
                )
            elif os.path.exists(gemma_local_path):
                # 其次尝试 Gemma 本地 params.npz
                sae, cfg_dict, sparsity = SAE.from_pretrained(
                    release=sae_release,
                    sae_id=path,
                    device=device,
                    local_path=gemma_local_path,
                )
            else:
                # 最后退回到 SAE.from_pretrained 默认下载/缓存
                sae, cfg_dict, sparsity = SAE.from_pretrained(
                    release=sae_release,
                    sae_id=path,
                    device=device,
                )
        except Exception as ex:
            import traceback
            print(f"[ERROR] Load SAE failed for {path}: {ex}", file=sys.stderr)
            print(f"[ERROR] Exception type: {type(ex).__name__}", file=sys.stderr)
            print(f"[ERROR] Full traceback:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            raise
        
        if force_device is not None:
            sae.to(force_device)
        elif n_devices > 1:
            sae.to("cuda:1")
        else:
            sae.to(device)
        sae.use_error_term = True
        sae_list.append(sae)
        hook_names.append(
            sae.cfg.metadata.hook_name
            if hasattr(sae.cfg, "metadata")
            else sae.cfg.hook_name
        )
    return sae_list, hook_names


def format_qa_prompt(question: str, answer: str, context: str) -> str:
    if context != "" and context is not None:
        return f"Context: {context}\nQuestion: {question}\nAnswer: {answer}"
    else:
        return f"Question: {question}\nAnswer: {answer}"


def format_qa_prompt_instruct(question: str, answer: str, context: str, tokenizer) -> str:
    try:
        if context != "" and context is not None:
            messages = [
                {"role": "user", "content": f"Context: {context}\nQuestion: {question}"},
                {"role": "assistant", "content": f"Answer: {answer}"},
            ]
        else:
            messages = [
                {"role": "user", "content": f"Question: {question}"},
                {"role": "assistant", "content": f"Answer: {answer}"},
            ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return prompt
    except Exception as e:
        print(f"Warning: Failed to apply chat template: {e}")
        return format_qa_prompt(question, answer)


def get_formatted_prompt(question: str, answer: str, context: str, is_instruct: bool, tokenizer) -> str:
    if is_instruct and tokenizer is not None:
        return format_qa_prompt_instruct(question, answer, context, tokenizer)
    return format_qa_prompt(question, answer, context)


def parse_sae_paths_or_default(args: argparse.Namespace) -> List[str]:
    if args.sae_paths.strip():
        return [p.strip() for p in args.sae_paths.split(',') if p.strip()]
    # default for 2B-PT
    return [
        "layer_0/width_16k/average_l0_105",
        "layer_1/width_16k/average_l0_102",
        "layer_2/width_16k/average_l0_141",
        "layer_3/width_16k/average_l0_59",
        "layer_4/width_16k/average_l0_124",
        "layer_5/width_16k/average_l0_68",
        "layer_6/width_16k/average_l0_70",
        "layer_7/width_16k/average_l0_69",
        "layer_8/width_16k/average_l0_71",
        "layer_9/width_16k/average_l0_73",
        "layer_10/width_16k/average_l0_77",
        "layer_11/width_16k/average_l0_80",
        "layer_12/width_16k/average_l0_82",
        "layer_13/width_16k/average_l0_84",
        "layer_14/width_16k/average_l0_84",
        "layer_15/width_16k/average_l0_78",
        "layer_16/width_16k/average_l0_78",
        "layer_17/width_16k/average_l0_77",
        "layer_18/width_16k/average_l0_74",
        "layer_19/width_16k/average_l0_73",
        "layer_20/width_16k/average_l0_71",
        "layer_21/width_16k/average_l0_70",
        "layer_22/width_16k/average_l0_72",
        "layer_23/width_16k/average_l0_75",
        "layer_24/width_16k/average_l0_73",
        "layer_25/width_16k/average_l0_116",
    ]


def batch_layers(layer_paths: List[str], batch_size: int) -> List[List[str]]:
    return [layer_paths[i:i+batch_size] for i in range(0, len(layer_paths), batch_size)]


def layer_num_from_hook_name(hook_name: str) -> Optional[int]:
    try:
        m = re.search(r"\.(\d+)\.", hook_name)
        return int(m.group(1)) if m else None
    except Exception:
        return None


def extract_layer_numbers_from_sae_paths(sae_paths: List[str], sae_release: str) -> List[int]:
    """从SAE路径中提取层号，根据 sae_release 自动区分 Gemma 和 Llama 格式"""
    layer_numbers = []
    for path in sae_paths:
        layer_num = extract_layer_num(path, sae_release)
        if layer_num is not None:
            layer_numbers.append(layer_num)
    return sorted(layer_numbers)


def process_split(
    split_name: str,
    input_path: str,
    out_dir: str,
    model: HookedSAETransformer,
    device: str,
    args: argparse.Namespace,
    parcel_mapping: 'ParcelMapping',
    latent_to_parcels: DefaultDict[Tuple[int, int], List[int]],
) -> None:
    out_file = os.path.join(out_dir, "token_parcels.jsonl")
    if args.skip_existing and os.path.exists(out_file):
        print(f"[SKIP] Output exists for split={split_name}: {out_file}")
        return

    if args.dry_run:
        print(f"[DRY-RUN] Would process {input_path} -> {out_file}")
        return

    sae_paths_all = parse_sae_paths_or_default(args)
    layer_batches = batch_layers(sae_paths_all, args.layers_per_batch)

    aggregate_sum = args.aggregate_mode == "sum"
    wrote_any = False
    out_records: List[Dict] = []
    # First load and parse all records
    records: List[Dict] = []
    with open(input_path, "r", encoding="utf-8") as fin:
        for line_idx, line in tqdm(enumerate(fin, start=1)):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception as ex:
                import traceback
                print(f"[ERROR] JSON parse failed {input_path}:{line_idx}: {ex}", file=sys.stderr)
                print(f"[ERROR] Exception type: {type(ex).__name__}", file=sys.stderr)
                print(f"[ERROR] Full traceback:", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                continue
            records.append(rec)

    # Then process each record and collect outputs in memory
    for rec_idx, rec in tqdm(enumerate(records, start=1), desc=f"Processing records for {split_name}", total=len(records)):
        try:
            question = rec.get("question", None)
            context = rec.get("context", None)
            answer = rec.get("model_answer", None)
            if question is None or answer is None:
                print(f"[WARN] Missing question/model_answer at rec {rec_idx}; skipping", file=sys.stderr)
                continue

            prompt = get_formatted_prompt(
                question=question,
                context=context,
                answer=answer,
                is_instruct=args.is_instruct,
                tokenizer=model.tokenizer,
            )
            try:
                tokens = model.tokenizer(prompt, return_tensors="pt", add_special_tokens=True, return_offsets_mapping=True, padding=True, max_length=args.max_length, truncation=True)
            except Exception as ex:
                import traceback
                print(f"[ERROR] Tokenization failed at rec {rec_idx}: {ex}", file=sys.stderr)
                print(f"[ERROR] Exception type: {type(ex).__name__}", file=sys.stderr)
                print(f"[ERROR] Full traceback:", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                continue

            input_ids = tokens["input_ids"].to(device)
            offsets = tokens["offset_mapping"][0]
            seq_len = input_ids.shape[1]
            
            # 找到 answer 部分在 prompt 中的位置
            answer_start = prompt.rfind(answer)
            answer_end = answer_start + len(answer)
            
            # 计算 answer 对应的 token 索引
            answer_token_indices = [i for i, (ofs_s, ofs_e) in enumerate(offsets.tolist()) 
                                    if (ofs_e > answer_start and ofs_e <= answer_end) or 
                                    (ofs_s < answer_start and ofs_e > answer_start)]
            
            # 过滤特殊 token
            ids = input_ids[0].tolist()
            special_ids = set(model.tokenizer.all_special_ids) if hasattr(model.tokenizer, 'all_special_ids') else set()
            valid_indices = [i for i in answer_token_indices if ids[i] not in special_ids]
            
            if not valid_indices:
                print(f"[WARN] No valid tokens found for answer at rec {rec_idx}; skipping", file=sys.stderr)
                continue
            
            # 只处理 answer 部分的 token
            pos_slice = slice(valid_indices[0], valid_indices[-1]+1)
            answer_seq_len = len(valid_indices)
            
            # 提取 answer 部分的 token 文本
            answer_tokens = []
            for idx in valid_indices:
                token_id = ids[idx]
                token_text = model.tokenizer.decode([token_id])
                answer_tokens.append(token_text)

            # Initialize per-token parcel aggregation buffer
            num_parcels = parcel_mapping.num_parcels
            # To reduce memory, we accumulate as list per token (only for answer tokens)
            token_parcel_vals = [None] * answer_seq_len  # each will be a list[float] length=num_parcels
            counts_per_token = [0] * answer_seq_len  # for mean aggregation

            # Process batches of layers
            for batch_idx, layer_paths in enumerate(layer_batches):
                # Load SAEs with simple device retry strategy similar to get_sae_act.py
                max_retries = 2
                retry = 0
                sae_list = None
                hook_names = None
                success = False
                
                while retry <= max_retries and not success:
                    try:
                        force_dev = None
                        if retry == 1:
                            force_dev = "cuda:0"
                        elif retry == 2:
                            force_dev = "cuda:1"
                        print(f"[INFO] Loading SAEs with force_device: {force_dev}")
                        sae_list, hook_names = load_sae_for_layers(
                            layer_paths=layer_paths,
                            sae_release=args.sae_release,
                            sae_local_base_dir=args.sae_local_base_dir,
                            device=device,
                            n_devices=args.n_devices,
                            force_device=force_dev,
                        )
                        print(f"[INFO] Loaded SAEs over with force_device: {force_dev}")
                        # Build names filter
                        if args.baseline_direct_neural:
                            sae_keys = [f"{h}.hook_sae_input" for h in hook_names]
                        else:
                            sae_keys = [f"{h}.hook_sae_acts_post" for h in hook_names]
                        max_layer_num = -1
                        for h in hook_names:
                            ln = layer_num_from_hook_name(h)
                            if ln is not None:
                                max_layer_num = max(max_layer_num, ln)
                        _, cache = model.run_with_cache_with_saes(
                            input_ids,
                            saes=sae_list,
                            names_filter=sae_keys,
                            stop_at_layer=max_layer_num + 1 if max_layer_num >= 0 else None,
                            pos_slice=pos_slice,
                        )

                        # For each layer acts: [1, answer_seq_len, d]
                        for key, hname in zip(sae_keys, hook_names):
                            acts = cache[key]  # tensor [1, answer_seq, d_sae]
                            if torch.isnan(acts).any():
                                import pdb; pdb.set_trace()
                                print(f"[ERROR] SAE acts contains NaN at {key}", file=sys.stderr)
                                continue
                            layer_num = layer_num_from_hook_name(hname)
                            if layer_num is None:
                                print(f"[WARN] Cannot infer layer number for {hname}", file=sys.stderr)
                                continue
                            d_sae = acts.shape[2]
                            # Aggregate per token (only answer tokens)
                            with torch.no_grad():
                                acts_cpu = acts[0].detach().cpu()  # [answer_seq, d_sae]
                                for tok_idx in range(answer_seq_len):
                                    vec = acts_cpu[tok_idx]  # [d_sae]
                                    # Initialize buffer lazily
                                    if token_parcel_vals[tok_idx] is None:
                                        token_parcel_vals[tok_idx] = [0.0] * num_parcels
                                    # Map latents to parcels
                                    for latent_idx in range(d_sae):
                                        parcels = latent_to_parcels.get((layer_num, latent_idx))
                                        if not parcels:
                                            continue
                                        val = float(vec[latent_idx].item())
                                        for pid in parcels:
                                            if 0 <= pid < num_parcels:
                                                token_parcel_vals[tok_idx][pid] += val
                                            else:
                                                print(f"[WARN] Invalid parcel index: {pid}", file=sys.stderr)
                                    counts_per_token[tok_idx] += 1
                        
                        # 成功执行完成，跳出循环
                        success = True
                        print(f"[INFO] Successfully processed batch {batch_idx} with force_device: {force_dev}")
            
                    except RuntimeError as e:
                        msg = str(e)
                        print(f"[ERROR] RuntimeError: {msg}")
                        if "Expected all tensors to be on the same device" in msg and retry < max_retries:
                            retry += 1
                            # if torch.cuda.is_available():
                            #     torch.cuda.empty_cache()
                            print(f"[INFO] Retrying with attempt {retry}/{max_retries}")
                            continue
                        else:
                            print(f"[ERROR] Failed after {retry + 1} attempts, giving up")
                            raise e
                
            # Mean aggregation if requested
            if not aggregate_sum:
                for tok_idx in range(answer_seq_len):
                    c = counts_per_token[tok_idx]
                    if c > 0 and token_parcel_vals[tok_idx] is not None:
                        token_parcel_vals[tok_idx] = [v / c for v in token_parcel_vals[tok_idx]]

            # 验证第一个样本的 Parcel 激活数组长度和索引
            if rec_idx == 1 and token_parcel_vals:
                first_token_parcels = token_parcel_vals[0]
                if first_token_parcels is not None:
                    actual_length = len(first_token_parcels)
                    expected_length = parcel_mapping.num_parcels
                    print(f"[DEBUG] 第一个样本第一个 token 的 Parcel 激活数组长度: {actual_length}, 期望: {expected_length}")
                    if actual_length != expected_length:
                        print(f"[ERROR] Parcel 激活数组长度不匹配！", file=sys.stderr)
                    else:
                        print(f"[DEBUG] Parcel 激活数组长度验证通过")

            # Collect one record per sample
            out_rec = {
                "index": rec.get("index"),
                "num_tokens": answer_seq_len,  # 只保存 answer 部分的 token 数量
                "parcel_dim": parcel_mapping.num_parcels,
                "answer_tokens": answer_tokens,  # 保存 answer 部分的 token 文本
                "token_parcel_acts": token_parcel_vals,
            }
            out_records.append(out_rec)
            wrote_any = True
        except Exception as ex:
            import traceback
            print(f"[ERROR] Failed processing split {split_name}: {ex}", file=sys.stderr)
            print(f"[ERROR] Exception type: {type(ex).__name__}", file=sys.stderr)
            print(f"[ERROR] Full traceback:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            
            if "cuda out of memory" in str(ex).lower():
                print(f"[INFO] CUDA out of memory, skipping this record")
            else:
                print(f"[ERROR] Unexpected error occurred, re-raising exception", file=sys.stderr)
                raise ex
    # Finally write all outputs in one pass
    with open(out_file, "w", encoding="utf-8") as fout:
        for out_rec in out_records:
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")


    if wrote_any:
        print(f"[DONE] Wrote token-level parcel activations: {out_file}")
    else:
        print(f"[INFO] No records written for split {split_name}")


def main() -> None:
    args = parse_args()

    results_root = os.path.abspath(args.results_root)
    combo_name = args.combo_name
    mapping_path = os.path.abspath(args.parcel_mapping)

    # 解析SAE路径并提取层号
    sae_paths_all = parse_sae_paths_or_default(args)
    available_layers = extract_layer_numbers_from_sae_paths(sae_paths_all, args.sae_release)
    print(f"[INFO] 从SAE路径提取的层号: {available_layers}")

    # 计算每层 latent 数
    model_name = args.model_name or infer_model_from_combo(combo_name)
    if not model_name:
        print(f"[ERROR] Cannot infer model name from combo '{combo_name}'. Provide --model-name.", file=sys.stderr)
        sys.exit(1)
    latents_per_layer = get_latents_per_layer(model_name, args.sae_release)
    print(f"[INFO] 每层 latent 数: {latents_per_layer} (model={model_name}, release={args.sae_release})")

    # Load parcel mapping with available layers and latents_per_layer
    parcel_mapping = load_parcel_mapping(mapping_path, available_layers, latents_per_layer)
    
    # 验证 Parcel 索引的连续性和正确性
    if not parcel_mapping.validate_parcel_indices():
        print(f"[ERROR] Parcel 索引验证失败，请检查映射文件: {mapping_path}", file=sys.stderr)
        sys.exit(1)
    
    latent_to_parcels = parcel_mapping.build_latent_to_parcels_index()
    print(f"[INFO] Parcels: {parcel_mapping.num_parcels}; latent index size: {len(latent_to_parcels)}")

    input_splits = get_input_paths(results_root, combo_name, args.split)
    if not input_splits:
        print(f"[INFO] No input splits to process under {results_root}/{combo_name}")
        return

    # model_name 已经在上面计算 latents_per_layer 时获取了，这里直接使用
    device = detect_device()
    print(f"[INFO] Loading model: {model_name} on device: {device}")
    model = load_model(model_name, device, args.n_devices)

    # Prepare outputs (directories); files are written later when activations are available
    for split_name, input_path in input_splits:
        if not os.path.exists(input_path):
            print(f"[WARN] Missing input file for split {split_name}: {input_path}", file=sys.stderr)
            continue
        out_dir = output_dir_for_split(results_root, combo_name, split_name)
        ensure_dir(out_dir, dry_run=args.dry_run)
        print(f"[INFO] Ready to process split='{split_name}' input={input_path} -> out_dir={out_dir}")
        # Execute processing
        process_split(
            split_name=split_name,
            input_path=input_path,
            out_dir=out_dir,
            model=model,
            device=device,
            args=args,
            parcel_mapping=parcel_mapping,
            latent_to_parcels=latent_to_parcels,
        )


if __name__ == "__main__":
    main()


