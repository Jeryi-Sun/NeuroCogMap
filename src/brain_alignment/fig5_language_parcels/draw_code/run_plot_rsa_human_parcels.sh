#!/bin/bash
# 运行脚本：计算并绘制 Human Parcel 之间的 Representational Similarity Analysis (RSA)
# 支持对多个 UST_ID 和 METHOD 批量运行
# 新增：支持合并多个 UST_ID 的 CSV（同一 Human Parcel 对应的 LLM Parcel 合并去重后按 semantic_similarity 排序取 topk）
#
# 注意：`plot_rsa_human_parcels.py` 已改为 **只输出 SVG**（不再输出 PNG/PDF）。

# 设置脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# =========================
# 基本配置（直接在这里改）
# =========================
# UST 被试 ID 列表（如：uts02 / uts03）
UST_IDS=("uts02" "uts03")
# 数据名称 / 故事名称（如：whereisthesmoke / adollshouse）
STORY_NAME="whereisthesmoke" # Story 名称（用于构建路径和加载激活数据）
# 方法名称列表（目前支持：sae / saeact）
METHODS=("saeact")

PROJECT_ROOT="/path/to/project_root/Human_LLM_align/litcoder_core/data_analysis/draw_graphs"

# =========================
# 其他默认参数
# =========================
CSV_FILE=""        # 留空表示按每个组合自动从 draw_result 里找
RESULTS_BASENAME=""  # 当前仅作为占位，保留向后兼容
LLM_PARCEL_JSON="/path/to/project_root/neural_area/divide_area_by_sae_act/cluster_output_2b_pt/clustering_results_sentence_prep0.03_0.8_svdvar0p80_parcels20_iter50_spatial0.01_nparcels270/latent_parcel_topsamples_functionality_analysis.json"
PARCEL_DESC_JSON="/path/to/project_root/Human_LLM_align/litcoder_core/dataset/brain_parcel_description/parcel_descriptions.json"
TOP_K=3
MODEL_NAME="/path/to/local_models/Qwen3-Embedding-8B"
BATCH_SIZE=32
DEVICE="cuda"  # 使用 GPU，如果 GPU 不可用会自动回退到 CPU
OUTPUT_FILE=""  # 留空表示按每个组合自动生成到 draw_result/.../rsa_files
OVERWRITE="--overwrite"  # 空字符串表示不覆盖，需要显式指定 --overwrite 才会覆盖
SKIP_EXISTING=""  # 空字符串表示不跳过已存在的文件
FILTER_NETWORKS=("SalVentAttn" "Cont" "Default") #("Vis" "SomMot" "Default" "SalVentAttn" "Limbic" "Cont" "DorsAttn")
USE_KTH=""  # 空字符串表示使用topk模式，设置为"--use_kth"则使用k-th模式
COMPARE_ACTIVATION=""  # 空字符串表示不比较激活 RSM
COMPARE_COGNITION_TERMS=""  # 空字符串表示不比较cognition terms RSM
COMPARE_FUNCTION_DESCRIPTION="--compare_function_description"
SIMILARITY_METHOD="cosine"  # 相似度计算方法：cosine 或 pearson
SPLIT_BY_NETWORK_PLOTS="--split_by_network_plots"  # 空字符串表示不按 network 拆分绘图（LLM / function_description 都会跟随该开关）
PLOT_CORRELATION_BAR="--plot_correlation_bar"  # 空字符串表示不绘制相关性柱状图（输出为 SVG）
# 默认只重绘（更快）：基于已有 CSV/JSON 生成 SVG；如需重新计算 embedding/RSM，请把它设为空字符串
PLOT_ONLY="" #"--plot_only"
COGNITION_TERMS_CSV="/path/to/project_root/Human_LLM_align/litcoder_core/dataset/brain_parcel_description/ns_scale100.csv"
MERGE_SUBJECTS="--merge_subjects"  # 空字符串表示不合并；设置为"--merge_subjects" 表示合并 UST_IDS 中所有被试

# 解析命令行参数（逻辑基本保持原样）
while [[ $# -gt 0 ]]; do
    case $1 in
        --csv_file)
            CSV_FILE="$2"
            shift 2
            ;;
        --llm_parcel_json)
            LLM_PARCEL_JSON="$2"
            shift 2
            ;;
        --parcel_desc_json)
            PARCEL_DESC_JSON="$2"
            shift 2
            ;;
        --top_k)
            TOP_K="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --output_file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --story_name)
            STORY_NAME="$2"
            shift 2
            ;;
        --overwrite)
            OVERWRITE="--overwrite"
            shift
            ;;
        --skip_existing)
            SKIP_EXISTING="--skip_existing"
            shift
            ;;
        --filter_networks)
            FILTER_NETWORKS=()
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                FILTER_NETWORKS+=("$1")
                shift
            done
            if [ ${#FILTER_NETWORKS[@]} -eq 0 ]; then
                FILTER_NETWORKS=()
            fi
            ;;
        --use_kth)
            USE_KTH="--use_kth"
            shift
            ;;
        --compare_activation)
            COMPARE_ACTIVATION="--compare_activation"
            shift
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --similarity_method)
            SIMILARITY_METHOD="$2"
            shift 2
            ;;
        --compare_cognition_terms)
            COMPARE_COGNITION_TERMS="--compare_cognition_terms"
            shift
            ;;
        --cognition_terms_csv)
            COGNITION_TERMS_CSV="$2"
            shift 2
            ;;
        --compare_function_description)
            COMPARE_FUNCTION_DESCRIPTION="--compare_function_description"
            shift
            ;;
        --split_by_network_plots)
            SPLIT_BY_NETWORK_PLOTS="--split_by_network_plots"
            shift
            ;;
        --plot_correlation_bar)
            PLOT_CORRELATION_BAR="--plot_correlation_bar"
            shift
            ;;
        --plot_only)
            PLOT_ONLY="--plot_only"
            shift
            ;;
        --merge_subjects)
            MERGE_SUBJECTS="--merge_subjects"
            shift
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            # ... 原 help 文本可照抄你现有版本 ...
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# =========================
# 工具函数：合并多个被试的 CSV（按 human_parcel_name + llm_parcel 去重，保留最大 semantic_similarity）
# 输出列保持为：human_parcel_name,llm_parcel,semantic_similarity
# =========================
merge_subject_csvs () {
  local out_csv="$1"
  shift
  local in_csvs=("$@")

  python - "$out_csv" "${in_csvs[@]}" << 'PY'
import sys
from pathlib import Path
import pandas as pd

out_csv = Path(sys.argv[1])
in_csvs = [Path(p) for p in sys.argv[2:]]

required_cols = {"human_parcel_name", "llm_parcel", "semantic_similarity"}

dfs = []
for p in in_csvs:
    if not p.exists():
        raise FileNotFoundError(f"找不到输入 CSV: {p}")
    df = pd.read_csv(p)
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少必要列 {sorted(missing)}: {p}\n当前列: {df.columns.tolist()}")
    df = df[list(required_cols)].copy()
    # 强制转 float，遇到异常应抛出（不要静默吞掉）
    df["semantic_similarity"] = df["semantic_similarity"].astype(float)
    dfs.append(df)

merged = pd.concat(dfs, axis=0, ignore_index=True)

# 去重：同一 human_parcel_name + llm_parcel，保留更大的 semantic_similarity
merged = (
    merged.groupby(["human_parcel_name", "llm_parcel"], as_index=False)["semantic_similarity"]
    .max()
)

# 排序：保证 plot_rsa_human_parcels.py 的 topk/head() 稳定按相似度取
merged = merged.sort_values(
    ["human_parcel_name", "semantic_similarity", "llm_parcel"],
    ascending=[True, False, True],
    kind="mergesort",
)

out_csv.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(out_csv, index=False)
print(f"[merge_subject_csvs] 合并完成: {out_csv} (rows={len(merged)})")
PY
}

# 检查 Python 脚本是否存在
PYTHON_SCRIPT="$SCRIPT_DIR/plot_rsa_human_parcels.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "错误: 找不到 Python 脚本: $PYTHON_SCRIPT"
    exit 1
fi

OVERALL_EXIT_CODE=0

# =========================
# 合并模式：把多个 UST 的 CSV 合并后跑一遍 RSA
# 输出到：draw_result/merged_subjects_test/rsa （内容与单 UST 一致）
# =========================
if [ -n "$MERGE_SUBJECTS" ]; then
  for METHOD in "${METHODS[@]}"; do
    # 准备输入 CSV 列表
    INPUT_CSVS=()
    for UST_ID in "${UST_IDS[@]}"; do
      DRAW_RESULT_ROOT="${PROJECT_ROOT}/draw_result/${UST_ID}/${STORY_NAME}/${METHOD}"
      if [ -n "$CSV_FILE" ]; then
        # 若用户手工指定了 --csv_file，则无法自动为不同 UST 生成对应路径，直接报错提示
        echo "错误: 合并模式下不支持同时指定 --csv_file。请留空让脚本自动从每个 UST 的 draw_result 读取。"
        OVERALL_EXIT_CODE=1
        continue 2
      fi
      CURRENT_CSV="${DRAW_RESULT_ROOT}/top_human_parcels_per_llm.csv"
      if [ ! -f "$CURRENT_CSV" ]; then
        echo "错误: 找不到 CSV 文件: $CURRENT_CSV"
        echo "UST_ID=${UST_ID}, METHOD=${METHOD}，请先确保已生成 top_human_parcels_per_llm.csv"
        OVERALL_EXIT_CODE=1
        continue 2
      fi
      INPUT_CSVS+=("$CURRENT_CSV")
    done

    # 合并输出 CSV（按要求：合并去重 + 按相似度排序）
    MERGED_ROOT="${PROJECT_ROOT}/draw_result/merged_subjects_test"
    MERGED_CSV="${MERGED_ROOT}/merged_top_human_parcels_per_llm_${STORY_NAME}_${METHOD}.csv"
    merge_subject_csvs "$MERGED_CSV" "${INPUT_CSVS[@]}"

    # 输出目录（不含 topK 子目录）
    BASE_OUTPUT_DIR="${MERGED_ROOT}/rsa/topk"

    # 循环 top_k = 1..10
    for TOP_K in $(seq 3 3); do
      CURRENT_OUTPUT_FILE="${BASE_OUTPUT_DIR}/top${TOP_K}"
      mkdir -p "$CURRENT_OUTPUT_FILE"

      CMD="python \"$PYTHON_SCRIPT\" \
          --csv_file \"$MERGED_CSV\" \
          --llm_parcel_json \"$LLM_PARCEL_JSON\" \
          --parcel_desc_json \"$PARCEL_DESC_JSON\" \
          --top_k $TOP_K \
          --model_name \"$MODEL_NAME\" \
          --batch_size $BATCH_SIZE \
          --device \"$DEVICE\" \
          --output_file \"$CURRENT_OUTPUT_FILE\""

      # 开启全脑 RSM 聚类，识别块结构
      CMD="$CMD" # --cluster_rsa --cluster_n 6 --cluster_method ward

      if [ -n "$OVERWRITE" ]; then
        CMD="$CMD --overwrite"
      fi

      if [ -n "$SKIP_EXISTING" ]; then
        CMD="$CMD --skip_existing"
      fi

      if [ ${#FILTER_NETWORKS[@]} -gt 0 ]; then
        CMD="$CMD --filter_networks ${FILTER_NETWORKS[*]}"
      fi

      if [ -n "$USE_KTH" ]; then
        CMD="$CMD --use_kth"
      fi

      if [ -n "$COMPARE_COGNITION_TERMS" ]; then
        CMD="$CMD --compare_cognition_terms"
        if [ -n "$COGNITION_TERMS_CSV" ]; then
          CMD="$CMD --cognition_terms_csv \"$COGNITION_TERMS_CSV\""
        fi
        if [ -n "$SIMILARITY_METHOD" ]; then
          CMD="$CMD --similarity_method \"$SIMILARITY_METHOD\""
        fi
      fi

      if [ -n "$COMPARE_FUNCTION_DESCRIPTION" ]; then
        CMD="$CMD --compare_function_description"
        if [ -n "$SIMILARITY_METHOD" ]; then
          CMD="$CMD --similarity_method \"$SIMILARITY_METHOD\""
        fi
      fi

      if [ -n "$SPLIT_BY_NETWORK_PLOTS" ]; then
        CMD="$CMD --split_by_network_plots"
      fi

      if [ -n "$PLOT_CORRELATION_BAR" ]; then
        CMD="$CMD --plot_correlation_bar"
      fi

      if [ -n "$PLOT_ONLY" ]; then
        CMD="$CMD --plot_only"
      fi

      if [ -n "$COMPARE_ACTIVATION" ]; then
        CMD="$CMD --compare_activation"
        if [ -n "$STORY_NAME" ]; then
          CMD="$CMD --story_name \"$STORY_NAME\""
        fi
        if [ -n "$DATA_DIR" ]; then
          CMD="$CMD --data_dir \"$DATA_DIR\""
        fi
        if [ -n "$SIMILARITY_METHOD" ]; then
          CMD="$CMD --similarity_method \"$SIMILARITY_METHOD\""
        fi
      fi

      echo "========================================="
      echo "开始计算 RSA (合并被试): UST_IDS=${UST_IDS[*]}, STORY_NAME=${STORY_NAME}, METHOD=${METHOD}, TOP_K=${TOP_K}"
      echo "合并 CSV: $MERGED_CSV"
      echo "输出目录: $CURRENT_OUTPUT_FILE"
      echo "========================================="
      echo ""

      eval $CMD
      exit_code=$?

      if [ $exit_code -eq 0 ]; then
        echo ""
        echo "✓ 合并模式 METHOD=${METHOD}, TOP_K=${TOP_K} 完成！"
        echo "结果已保存到: $CURRENT_OUTPUT_FILE"
      else
        echo ""
        echo "✗ 合并模式 METHOD=${METHOD}, TOP_K=${TOP_K} 执行失败，退出码: $exit_code"
        OVERALL_EXIT_CODE=$exit_code
      fi
    done
  done

  exit $OVERALL_EXIT_CODE
fi

# =========================
# 原有模式：逐个 UST 运行
# =========================
for UST_ID in "${UST_IDS[@]}"; do
  for METHOD in "${METHODS[@]}"; do
    DRAW_RESULT_ROOT="${PROJECT_ROOT}/draw_result/${UST_ID}/${STORY_NAME}/${METHOD}"

    CURRENT_CSV_FILE="$CSV_FILE"
    BASE_OUTPUT_DIR="$OUTPUT_FILE"

    if [ -z "$CURRENT_CSV_FILE" ]; then
        CURRENT_CSV_FILE="${DRAW_RESULT_ROOT}/top_human_parcels_per_llm.csv"
    fi

    # 基础输出目录（不含 topK 子目录）
    if [ -z "$BASE_OUTPUT_DIR" ]; then
        BASE_OUTPUT_DIR="${DRAW_RESULT_ROOT}/rsa_files/topk"
    fi

    # 如果没有指定 CSV_FILE 但指定了 results_basename，这里也可以按需要拼接；当前略过或沿用你原逻辑

    # 检查输入文件是否存在
    if [ ! -f "$CURRENT_CSV_FILE" ]; then
        echo "错误: 找不到 CSV 文件: $CURRENT_CSV_FILE"
        echo "UST_ID=${UST_ID}, METHOD=${METHOD}，请先确保已生成 fig2_top_llm_parcels_per_human_human.csv"
        OVERALL_EXIT_CODE=1
        continue
    fi

    if [ ! -f "$LLM_PARCEL_JSON" ]; then
        echo "错误: 找不到 LLM Parcel JSON 文件: $LLM_PARCEL_JSON"
        OVERALL_EXIT_CODE=1
        continue
    fi

    # 循环 top_k = 1..10
    for TOP_K in $(seq 3 3); do
        CURRENT_OUTPUT_FILE="${BASE_OUTPUT_DIR}/top${TOP_K}"
    mkdir -p "$CURRENT_OUTPUT_FILE"

        # 构建命令（先基础参数，再追加可选参数）
    CMD="python \"$PYTHON_SCRIPT\" \
        --csv_file \"$CURRENT_CSV_FILE\" \
        --llm_parcel_json \"$LLM_PARCEL_JSON\" \
        --parcel_desc_json \"$PARCEL_DESC_JSON\" \
        --top_k $TOP_K \
        --model_name \"$MODEL_NAME\" \
        --batch_size $BATCH_SIZE \
        --device \"$DEVICE\" \
        --output_file \"$CURRENT_OUTPUT_FILE\""

        # 开启全脑 RSM 聚类，识别块结构
        CMD="$CMD --cluster_rsa --cluster_n 6 --cluster_method ward"

    if [ -n "$OVERWRITE" ]; then
        CMD="$CMD --overwrite"
    fi

    if [ -n "$SKIP_EXISTING" ]; then
        CMD="$CMD --skip_existing"
    fi

    if [ ${#FILTER_NETWORKS[@]} -gt 0 ]; then
        CMD="$CMD --filter_networks ${FILTER_NETWORKS[*]}"
    fi

    if [ -n "$USE_KTH" ]; then
        CMD="$CMD --use_kth"
    fi

    if [ -n "$COMPARE_COGNITION_TERMS" ]; then
        CMD="$CMD --compare_cognition_terms"
        if [ -n "$COGNITION_TERMS_CSV" ]; then
            CMD="$CMD --cognition_terms_csv \"$COGNITION_TERMS_CSV\""
        fi
        if [ -n "$SIMILARITY_METHOD" ]; then
            CMD="$CMD --similarity_method \"$SIMILARITY_METHOD\""
        fi
    fi

    if [ -n "$COMPARE_FUNCTION_DESCRIPTION" ]; then
        CMD="$CMD --compare_function_description"
        if [ -n "$SIMILARITY_METHOD" ]; then
            CMD="$CMD --similarity_method \"$SIMILARITY_METHOD\""
        fi
    fi

    if [ -n "$SPLIT_BY_NETWORK_PLOTS" ]; then
        CMD="$CMD --split_by_network_plots"
    fi

    if [ -n "$PLOT_CORRELATION_BAR" ]; then
        CMD="$CMD --plot_correlation_bar"
    fi

    if [ -n "$PLOT_ONLY" ]; then
        CMD="$CMD --plot_only"
    fi

    if [ -n "$COMPARE_ACTIVATION" ]; then
        CMD="$CMD --compare_activation"
        if [ -n "$STORY_NAME" ]; then
            CMD="$CMD --story_name \"$STORY_NAME\""
        fi
        if [ -n "$DATA_DIR" ]; then
            CMD="$CMD --data_dir \"$DATA_DIR\""
        fi
        if [ -n "$SIMILARITY_METHOD" ]; then
                CMD="$CMD --similarity_method \"$SIMILARITY_METHOD\""
        fi
    fi

    echo "========================================="
        echo "开始计算 RSA: UST_ID=${UST_ID}, STORY_NAME=${STORY_NAME}, METHOD=${METHOD}, TOP_K=${TOP_K}"
    echo "CSV 文件: $CURRENT_CSV_FILE"
    echo "输出目录: $CURRENT_OUTPUT_FILE"
    echo "========================================="
    echo ""

    eval $CMD
    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo ""
            echo "✓ 组合 UST_ID=${UST_ID}, METHOD=${METHOD}, TOP_K=${TOP_K} 完成！"
        echo "结果已保存到: $CURRENT_OUTPUT_FILE"
    else
        echo ""
            echo "✗ 组合 UST_ID=${UST_ID}, METHOD=${METHOD}, TOP_K=${TOP_K} 执行失败，退出码: $exit_code"
        OVERALL_EXIT_CODE=$exit_code
    fi
    done

  done
done

exit $OVERALL_EXIT_CODE