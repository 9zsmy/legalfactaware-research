#!/usr/bin/env python3
"""Print a compact, data-driven Chinese summary for an experiment talk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def pct(value: float) -> str:
    return f"{100 * value:.2f}"


def delta_pct(value: float) -> str:
    return f"{100 * value:+.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).with_name("results"))
    args = parser.parse_args()

    metrics = load_json(args.results_dir / "all_metrics.json")
    significance = load_json(args.results_dir / "paired_significance.json")
    labels = load_json(args.results_dir / "matched_count_label_audit.json")
    training = load_json(args.results_dir / "training_positive_diagnostics.json")
    length_close = load_json(args.results_dir / "length_close_comparison.json")
    expressions = load_json(args.results_dir / "expression_by_type.json")

    full = metrics["full_reference"]
    matched = metrics["matched_count_independent"]
    lexical = metrics["lexical_text_only"]
    position = metrics["position_marker_only"]
    drop = metrics["drop_only"]
    legacy = metrics["legacy_truncation"]

    print("LegalFactAware 补充实验：讲解用动态摘要")
    print("=" * 56)
    print("范围：冻结开发集 942 篇；测试集未传入实验脚本。")
    print()
    print("1. 正例数量匹配")
    print(f"   正例数：Oracle={labels['oracle_positives']}, matched={labels['matched_positives']}")
    print(f"   Full R1/R2/RL:    {pct(full['R1'])}/{pct(full['R2'])}/{pct(full['RL'])}")
    print(f"   Matched R1/R2/RL: {pct(matched['R1'])}/{pct(matched['R2'])}/{pct(matched['RL'])}")
    print(f"   差值:              {delta_pct(full['R1']-matched['R1'])}/{delta_pct(full['R2']-matched['R2'])}/{delta_pct(full['RL']-matched['RL'])}")
    length_delta = training["paired_per_document_oracle_minus_matched"]["mean_positive_length"]
    print(f"   Oracle 每篇正例平均长 {length_delta['mean_delta_left_minus_right']:.2f} 字。")
    primary = length_close["within_25_characters"]
    print(f"   25 字接近子集：n={primary['documents']}，ΔRL={delta_pct(primary['metrics']['rouge_l_f1']['mean_delta_left_minus_right'])}。")
    print()
    print("2. 词法与位置")
    print(f"   Raw lexical only: {pct(lexical['R1'])}/{pct(lexical['R2'])}/{pct(lexical['RL'])}")
    print(f"   Position only:    {pct(position['R1'])}/{pct(position['R2'])}/{pct(position['RL'])}")
    print("   结论：词汇内容是主要信号；位置和辅助特征是小增益。")
    print()
    print("3. 解码器机制")
    print(f"   Full:      RL={pct(full['RL'])}, Expression F1={pct(full['Expression_F1'])}, hard trunc={full['Hard_truncations']}")
    print(f"   Drop-only: RL={pct(drop['RL'])}, Expression F1={pct(drop['Expression_F1'])}, hard trunc={drop['Hard_truncations']}")
    print(f"   Legacy:    RL={pct(legacy['RL'])}, Expression F1={pct(legacy['Expression_F1'])}, hard trunc={legacy['Hard_truncations']}")
    full_drop = {row['metric']: row for row in significance['full_minus_drop_only']}
    print(f"   Full-drop ΔRL={delta_pct(full_drop['rouge_l_f1']['mean_delta_left_minus_right'])}, Holm p={full_drop['rouge_l_f1']['wilcoxon_p_holm']:.3g}")
    print()
    print("4. 分类型 Expression 微平均 F1")
    for name in ("amount", "date", "law", "decision"):
        support = expressions["full_reference"][name]["reference_nonempty_documents"]
        full_f1 = expressions["full_reference"][name]["micro_f1"]
        matched_f1 = expressions["matched_count_independent"][name]["micro_f1"]
        print(f"   {name:8s} support={support:3d}: full={pct(full_f1)}, matched={pct(matched_f1)}")
    print()
    print("结论边界：这些是开发集机制证据，不是法律事实人工正确率，也不包含 SENDE 配对检验。")


if __name__ == "__main__":
    main()
