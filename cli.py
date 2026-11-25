"""CLI entrypoint for ucai-sovereign-box v0.2.0."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import List

from ucai_sovereign_box import (
    SovereignBoxEvaluator,
    SovereignHybridIndex,
    build_sovereign_benchmark,
    generate_ro_crate_bundle,
    verify_ro_crate_bundle,
)
from ucai_sovereign_box.evaluator import mcnemar_exact_pvalue


def main() -> None:
    parser = argparse.ArgumentParser(description="ucai-sovereign-box v0.2.0 Single-GPU Research-in-a-Box CLI")
    parser.add_argument("--pdf-dir", action="append", default=[], help="Directory containing open-access PDFs")
    parser.add_argument("--out-dir", default="results_tier2", help="Output directory")
    parser.add_argument("--max-pages", type=int, default=20, help="Max pages per PDF")
    parser.add_argument("--max-chunks", type=int, default=120, help="Max chunks per PDF")
    parser.add_argument("--per-class", type=int, default=25, help="Benchmark items per class (default 25 -> N=100)")
    parser.add_argument("--run-local-llm", action="store_true", help="Also run local Qwen2.5-1.5B-Instruct on GPU/CPU")
    parser.add_argument("--local-llm-model", default="Qwen/Qwen2.5-1.5B-Instruct", help="HuggingFace open model ID")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index = SovereignHybridIndex()
    t0 = time.perf_counter()
    pdf_files: List[Path] = []
    for d in args.pdf_dir:
        dp = Path(d)
        if dp.exists():
            pdf_files.extend(sorted(dp.glob("*.pdf")))

    for pdf in pdf_files:
        m = re.search(r"(2[456]\d{2}\.\d{4,5})", pdf.name)
        doc_id = m.group(1) if m else None
        try:
            index.ingest_pdf(doc_id, pdf, max_pages=args.max_pages, max_chunks=args.max_chunks)
        except Exception:
            continue

    hw_profile = index.build_dense_index()
    hw_profile["index_build_sec"] = round(time.perf_counter() - t0, 2)
    (out_dir / "hardware_profile.json").write_text(json.dumps(hw_profile, indent=2), encoding="utf-8")

    items = build_sovereign_benchmark(out_dir / "UCAI-SovereignBox-Bench-v2.json", per_class=args.per_class)
    ev = SovereignBoxEvaluator(index)

    bm25_res = ev.evaluate_mode(items, mode="bm25_only")
    dense_res = ev.evaluate_mode(items, mode="dense_only")
    sov_res = ev.evaluate_mode(items, mode="sovereign_box")

    (out_dir / "bm25_only_results.json").write_text(json.dumps(bm25_res, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "dense_only_results.json").write_text(json.dumps(dense_res, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "sovereign_box_results.json").write_text(json.dumps(sov_res, indent=2, ensure_ascii=False), encoding="utf-8")

    results_list = [
        ("BM25-Only (Lexical)", bm25_res),
        ("Dense-Only (MiniLM)", dense_res),
        ("UCAI Sovereign Box v0.2.0 (Hybrid+Guard)", sov_res),
    ]

    if args.run_local_llm:
        llm_res = ev.evaluate_local_open_llm(items, model_id=args.local_llm_model)
        (out_dir / "sovereign_local_llm_results.json").write_text(
            json.dumps(llm_res, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        results_list.append((f"Sovereign + Local LLM ({args.local_llm_model.split('/')[-1]})", llm_res))

    y_true = [it.expected_flag for it in items]
    p_bm25 = [r["pred_flag"] for r in bm25_res["records"]]
    p_sov = [r["pred_flag"] for r in sov_res["records"]]
    stats = {"sov_vs_bm25": mcnemar_exact_pvalue(y_true, p_sov, p_bm25)}
    (out_dir / "paper3_tier2_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    csv_path = out_dir / "paper3_tier2_comparison_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Mode", "N", "Hit@3", "MRR@3", "Safety_Acc", "Precision", "Recall", "F1", "F1_CI95", "FPR", "ms_per_item", "Cloud_USD"])
        for lbl, res in results_list:
            m = res["metrics"]
            w.writerow([
                lbl, m["n"], m["retrieval_hit_at_3"], m["mrr_at_3"], m["safety_audit_acc"],
                m["precision"], m["recall"], m["f1"], m["f1_ci95"], m["fpr"], m["ms_per_item"], m["cloud_cost_usd"],
            ])

    crate_info = generate_ro_crate_bundle(
        output_dir=out_dir / "ro_crate_bundle",
        crate_name="UCAI-Flagship4-SovereignBox-Artifact-v2",
        artifact_files=[
            out_dir / "hardware_profile.json",
            out_dir / "UCAI-SovereignBox-Bench-v2.json",
            out_dir / "sovereign_box_results.json",
            out_dir / "paper3_tier2_comparison_summary.csv",
            out_dir / "paper3_tier2_stats.json",
        ],
        hardware_profile=hw_profile,
    )
    verify_info = verify_ro_crate_bundle(out_dir / "ro_crate_bundle", out_dir)
    print(json.dumps({"hw": hw_profile, "crate": crate_info, "verify": verify_info}, indent=2))


if __name__ == "__main__":
    main()
