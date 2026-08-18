#!/usr/bin/env python3
# This module compares paired benchmark predictions from two ReVA systems using
# a paired t-test on per-question correctness deltas and McNemar's test on the
# paired binary outcomes. McNemar is reported with the exact binomial p-value
# when the number of discordant pairs is small, and with the continuity-
# corrected chi-square approximation otherwise.
"""Paired tests: baseline vs final on benchmark JSONL predictions.
Reports paired t-test (H0: mean(d)=0) and McNemar test (H0: symmetric discordant pairs) on binary correctness d_i = score_final_i - score_baseline_i.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

POPE_SPLITS = ("random", "popular", "adversarial")
MCNEMAR_EXACT_DISCORDANT_THRESHOLD = 25
PERCENT_SCALE = 100.0


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows: List[dict] = []
    with path.open() as file_handle:
        for line in file_handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_summary_metric(output_dir: Path, summary_name: str, *metric_keys: str) -> Optional[float]:
    summary_path = output_dir / f"{summary_name}_summary.json"
    if not summary_path.exists():
        return None
    summary_data = json.loads(summary_path.read_text())
    for metric_key in metric_keys:
        if metric_key in summary_data and summary_data[metric_key] is not None:
            return float(summary_data[metric_key])
    return None


def pope_correctness_score(row: dict) -> Optional[float]:
    prediction = row.get("prediction")
    label = str(row.get("label", "")).lower()
    if prediction not in ("yes", "no") or label not in ("yes", "no"):
        return None
    return 1.0 if prediction == label else 0.0


def mmbench_or_seed_correctness_score(row: dict) -> Optional[float]:
    if "correct" in row and row["correct"] is not None:
        return 1.0 if row["correct"] else 0.0
    prediction = row.get("prediction")
    answer = str(row.get("answer", "")).strip().upper()
    if not answer or prediction is None:
        return None
    return 1.0 if str(prediction).strip().upper() == answer else 0.0


def normalize_index_key(raw_value) -> str:
    if raw_value is None or (isinstance(raw_value, float) and math.isnan(raw_value)):
        return ""
    if isinstance(raw_value, float) and raw_value.is_integer():
        return str(int(raw_value))
    return str(raw_value).strip()


def scores_by_key(rows: Sequence[dict], item_key: str, score_fn: Callable[[dict], Optional[float]]) -> dict[str, float]:
    score_map: dict[str, float] = {}
    for row in rows:
        if item_key == "index":
            item_id = normalize_index_key(row[item_key])
        else:
            item_id = str(row[item_key])
        score = score_fn(row)
        if score is None:
            continue
        score_map[item_id] = score
    return score_map


def merge_paired(baseline_scores: dict[str, float], final_scores: dict[str, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    shared_item_ids = sorted(set(baseline_scores) & set(final_scores))
    if not shared_item_ids:
        empty = np.array([], dtype=float)
        return empty, empty, empty, 0
    baseline_array = np.array([baseline_scores[item_id] for item_id in shared_item_ids], dtype=float)
    final_array = np.array([final_scores[item_id] for item_id in shared_item_ids], dtype=float)
    score_delta = final_array - baseline_array
    return baseline_array, final_array, score_delta, len(shared_item_ids)


@dataclass
class McNemarResult:
    chi2: float
    p_value: float
    significant: bool
    baseline_ok_final_fail: int
    baseline_fail_final_ok: int
    ci_low_pp: float
    ci_high_pp: float
    used_exact_p_value: bool


@dataclass
class TTestResult:
    t_stat: float
    df: int
    p_value: float
    significant: bool
    ci_low_fraction: float
    ci_high_fraction: float


@dataclass
class PairedResult:
    benchmark: str
    metric_label: str
    n: int
    baseline_acc_pct: float
    final_acc_pct: float
    delta_pp: float
    ci_low_pp: float
    ci_high_pp: float
    t_stat: float
    df: int
    p_value: float
    significant: bool
    mcnemar_stat: float
    mcnemar_chi2: float
    mcnemar_p: float
    mcnemar_significant: bool
    mcnemar_ci_low_pp: float
    mcnemar_ci_high_pp: float
    mcnemar_exact: bool
    discordant_base_ok_final_fail: int
    discordant_base_fail_final_ok: int
    summary_baseline: Optional[float]
    summary_final: Optional[float]
    notes: str = ""
    b: Optional[np.ndarray] = field(default=None, repr=False)
    f: Optional[np.ndarray] = field(default=None, repr=False)


def mcnemar_chi2_continuity_corrected(baseline_ok_final_fail: int, baseline_fail_final_ok: int) -> float:
    discordant_pair_count = baseline_ok_final_fail + baseline_fail_final_ok
    if discordant_pair_count == 0:
        return float("nan")
    numerator = abs(baseline_ok_final_fail - baseline_fail_final_ok) - 1
    return float(numerator ** 2 / discordant_pair_count)


def mcnemar_proportion_diff_ci_pp(baseline_ok_final_fail: int, baseline_fail_final_ok: int, sample_size: int, *, alpha: float) -> Tuple[float, float]:
    """Asymptotic CI for paired proportion difference (final - baseline), in percentage points."""
    if sample_size <= 0:
        return float("nan"), float("nan")
    delta_pp = PERCENT_SCALE * (baseline_fail_final_ok - baseline_ok_final_fail) / sample_size
    discordant_pair_count = baseline_ok_final_fail + baseline_fail_final_ok
    if discordant_pair_count == 0:
        return delta_pp, delta_pp
    standard_error_pp = PERCENT_SCALE * math.sqrt(discordant_pair_count / (sample_size * sample_size))
    z_critical = float(stats.norm.ppf(1 - alpha / 2))
    return delta_pp - z_critical * standard_error_pp, delta_pp + z_critical * standard_error_pp


def _count_mcnemar_cells(baseline_scores: np.ndarray, final_scores: np.ndarray) -> Tuple[int, int, int, int, List[List[int]]]:
    baseline_correct = baseline_scores.astype(bool)
    final_correct = final_scores.astype(bool)
    both_correct = int(np.sum(baseline_correct & final_correct))
    baseline_ok_final_fail = int(np.sum(baseline_correct & ~final_correct))
    baseline_fail_final_ok = int(np.sum(~baseline_correct & final_correct))
    both_wrong = int(np.sum(~baseline_correct & ~final_correct))
    contingency_table = [[both_correct, baseline_ok_final_fail], [baseline_fail_final_ok, both_wrong]]
    return both_correct, baseline_ok_final_fail, baseline_fail_final_ok, both_wrong, contingency_table


def mcnemar_paired(baseline_scores: np.ndarray, final_scores: np.ndarray, *, alpha: float) -> McNemarResult:
    _, baseline_ok_final_fail, baseline_fail_final_ok, _, _ = _count_mcnemar_cells(baseline_scores, final_scores)
    sample_size = int(baseline_scores.size)
    chi2 = mcnemar_chi2_continuity_corrected(baseline_ok_final_fail, baseline_fail_final_ok)
    ci_low_pp, ci_high_pp = mcnemar_proportion_diff_ci_pp(baseline_ok_final_fail, baseline_fail_final_ok, sample_size, alpha=alpha)
    discordant_pair_count = baseline_ok_final_fail + baseline_fail_final_ok

    if discordant_pair_count == 0:
        return McNemarResult(chi2=chi2, p_value=float("nan"), significant=False, baseline_ok_final_fail=baseline_ok_final_fail, baseline_fail_final_ok=baseline_fail_final_ok, ci_low_pp=ci_low_pp, ci_high_pp=ci_high_pp, used_exact_p_value=False)

    use_exact_p_value = discordant_pair_count < MCNEMAR_EXACT_DISCORDANT_THRESHOLD
    if use_exact_p_value:
        p_value = float(stats.binomtest(baseline_fail_final_ok, n=discordant_pair_count, p=0.5, alternative="two-sided").pvalue)
    else:
        p_value = float(stats.chi2.sf(chi2, df=1))

    return McNemarResult(chi2=chi2, p_value=p_value, significant=bool(p_value < alpha), baseline_ok_final_fail=baseline_ok_final_fail, baseline_fail_final_ok=baseline_fail_final_ok, ci_low_pp=ci_low_pp, ci_high_pp=ci_high_pp, used_exact_p_value=use_exact_p_value)


def run_paired_t_test(score_delta: np.ndarray, *, alpha: float) -> TTestResult:
    sample_size = int(score_delta.size)
    mean_delta = float(np.mean(score_delta))
    standard_error = float(stats.sem(score_delta, nan_policy="omit"))
    t_stat, p_value = stats.ttest_1samp(score_delta, popmean=0.0, alternative="two-sided")
    degrees_of_freedom = sample_size - 1
    t_critical = float(stats.t.ppf(1 - alpha / 2, df=degrees_of_freedom))
    ci_low_fraction = mean_delta - t_critical * standard_error
    ci_high_fraction = mean_delta + t_critical * standard_error
    return TTestResult(t_stat=float(t_stat), df=degrees_of_freedom, p_value=float(p_value), significant=bool(p_value < alpha), ci_low_fraction=ci_low_fraction, ci_high_fraction=ci_high_fraction)


def _empty_mcnemar_result() -> McNemarResult:
    return McNemarResult(chi2=float("nan"), p_value=float("nan"), significant=False, baseline_ok_final_fail=0, baseline_fail_final_ok=0, ci_low_pp=float("nan"), ci_high_pp=float("nan"), used_exact_p_value=False)


def _build_paired_result(*, benchmark: str, metric_label: str, baseline_scores: np.ndarray, final_scores: np.ndarray, score_delta: np.ndarray, t_test: TTestResult, mcnemar: McNemarResult, summary_baseline: Optional[float], summary_final: Optional[float], notes: str) -> PairedResult:
    sample_size = int(score_delta.size)
    return PairedResult(
        benchmark=benchmark,
        metric_label=metric_label,
        n=sample_size,
        baseline_acc_pct=PERCENT_SCALE * float(np.mean(baseline_scores)) if sample_size else float("nan"),
        final_acc_pct=PERCENT_SCALE * float(np.mean(final_scores)) if sample_size else float("nan"),
        delta_pp=PERCENT_SCALE * float(np.mean(score_delta)) if sample_size else float("nan"),
        ci_low_pp=PERCENT_SCALE * t_test.ci_low_fraction,
        ci_high_pp=PERCENT_SCALE * t_test.ci_high_fraction,
        t_stat=t_test.t_stat,
        df=t_test.df,
        p_value=t_test.p_value,
        significant=t_test.significant,
        mcnemar_stat=mcnemar.chi2,
        mcnemar_chi2=mcnemar.chi2,
        mcnemar_p=mcnemar.p_value,
        mcnemar_significant=mcnemar.significant,
        mcnemar_ci_low_pp=mcnemar.ci_low_pp,
        mcnemar_ci_high_pp=mcnemar.ci_high_pp,
        mcnemar_exact=mcnemar.used_exact_p_value,
        discordant_base_ok_final_fail=mcnemar.baseline_ok_final_fail,
        discordant_base_fail_final_ok=mcnemar.baseline_fail_final_ok,
        summary_baseline=summary_baseline,
        summary_final=summary_final,
        notes=notes,
        b=baseline_scores,
        f=final_scores,
    )


def paired_ttest(*, benchmark: str, metric_label: str, b: np.ndarray, f: np.ndarray, d: np.ndarray, summary_baseline: Optional[float], summary_final: Optional[float], alpha: float, notes: str = "") -> PairedResult:
    sample_size = int(d.size)
    if sample_size < 2:
        empty_t_test = TTestResult(t_stat=float("nan"), df=max(0, sample_size - 1), p_value=float("nan"), significant=False, ci_low_fraction=float("nan"), ci_high_fraction=float("nan"))
        return _build_paired_result(benchmark=benchmark, metric_label=metric_label, baseline_scores=b, final_scores=f, score_delta=d, t_test=empty_t_test, mcnemar=_empty_mcnemar_result(), summary_baseline=summary_baseline, summary_final=summary_final, notes=notes or "Insufficient paired items (need n >= 2).")

    t_test = run_paired_t_test(d, alpha=alpha)
    mcnemar = mcnemar_paired(b, f, alpha=alpha)
    return _build_paired_result(benchmark=benchmark, metric_label=metric_label, baseline_scores=b, final_scores=f, score_delta=d, t_test=t_test, mcnemar=mcnemar, summary_baseline=summary_baseline, summary_final=summary_final, notes=notes)


def load_pope_pooled(dir_path: Path) -> dict[str, float]:
    pooled_scores: dict[str, float] = {}
    for split_name in POPE_SPLITS:
        predictions_path = dir_path / f"pope_{split_name}_predictions.jsonl"
        for row in read_jsonl(predictions_path):
            question_id = str(row.get("question_id", ""))
            if not question_id:
                continue
            score = pope_correctness_score(row)
            if score is None:
                continue
            pooled_scores[f"{split_name}/{question_id}"] = score
    return pooled_scores


def analyze_pope(baseline_dir: Path, final_dir: Path, alpha: float) -> PairedResult:
    baseline_scores = load_pope_pooled(baseline_dir)
    final_scores = load_pope_pooled(final_dir)
    baseline_array, final_array, score_delta, _ = merge_paired(baseline_scores, final_scores)
    summary_baseline = load_summary_metric(baseline_dir, "pope", "macro_f1_pct", "macro_accuracy_pct")
    summary_final = load_summary_metric(final_dir, "pope", "macro_f1_pct", "macro_accuracy_pct")
    return paired_ttest(benchmark="POPE", metric_label="Per-question accuracy (paired t-test + McNemar); summary F1 from pope_summary.json", b=baseline_array, f=final_array, d=score_delta, summary_baseline=summary_baseline, summary_final=summary_final, alpha=alpha, notes="POPE pooled across random/popular/adversarial; keys split/question_id.")


def analyze_mmbench(baseline_dir: Path, final_dir: Path, alpha: float) -> PairedResult:
    predictions_file = "mmbench_dev_en_predictions.jsonl"
    baseline_scores = scores_by_key(read_jsonl(baseline_dir / predictions_file), "index", mmbench_or_seed_correctness_score)
    final_scores = scores_by_key(read_jsonl(final_dir / predictions_file), "index", mmbench_or_seed_correctness_score)
    baseline_array, final_array, score_delta, _ = merge_paired(baseline_scores, final_scores)
    summary_baseline = load_summary_metric(baseline_dir, "mmbench_dev_en", "local_accuracy_pct")
    summary_final = load_summary_metric(final_dir, "mmbench_dev_en", "local_accuracy_pct")
    return paired_ttest(benchmark="MMBench (DEV EN)", metric_label="Per-question MCQ accuracy (paired t-test + McNemar)", b=baseline_array, f=final_array, d=score_delta, summary_baseline=summary_baseline, summary_final=summary_final, alpha=alpha, notes="Report VLMEvalKit CircularEval overall in thesis; t-test uses JSONL correctness.")


def analyze_seed(baseline_dir: Path, final_dir: Path, alpha: float) -> PairedResult:
    predictions_file = "seed_image_predictions.jsonl"
    baseline_scores = scores_by_key(read_jsonl(baseline_dir / predictions_file), "question_id", mmbench_or_seed_correctness_score)
    final_scores = scores_by_key(read_jsonl(final_dir / predictions_file), "question_id", mmbench_or_seed_correctness_score)
    baseline_array, final_array, score_delta, _ = merge_paired(baseline_scores, final_scores)
    summary_baseline = load_summary_metric(baseline_dir, "seed_image", "accuracy_pct")
    summary_final = load_summary_metric(final_dir, "seed_image", "accuracy_pct")
    return paired_ttest(benchmark="SEED-Image", metric_label="Per-question accuracy (paired t-test + McNemar)", b=baseline_array, f=final_array, d=score_delta, summary_baseline=summary_baseline, summary_final=summary_final, alpha=alpha, notes="SEED-Image dims 1-9 (same filter as evaluation.py).")


def analyze_vqav2_labeled(baseline_dir: Path, final_dir: Path, alpha: float) -> Optional[PairedResult]:
    """Only if JSONL rows include ground-truth ``correct`` (unusual on test2015)."""
    predictions_file = "vqav2_testdev_predictions.jsonl"
    baseline_rows = read_jsonl(baseline_dir / predictions_file)
    final_rows = read_jsonl(final_dir / predictions_file)
    if not baseline_rows or not final_rows:
        return None
    if not any("correct" in row for row in baseline_rows[:50]):
        return None

    def vqav2_correctness_score(row: dict) -> Optional[float]:
        if row.get("correct") is None:
            return None
        return 1.0 if row["correct"] else 0.0

    baseline_scores = scores_by_key(baseline_rows, "question_id", vqav2_correctness_score)
    final_scores = scores_by_key(final_rows, "question_id", vqav2_correctness_score)
    baseline_array, final_array, score_delta, _ = merge_paired(baseline_scores, final_scores)
    return paired_ttest(benchmark="VQAv2", metric_label="Per-question accuracy (paired t-test + McNemar)", b=baseline_array, f=final_array, d=score_delta, summary_baseline=load_summary_metric(baseline_dir, "vqav2_testdev", "score"), summary_final=load_summary_metric(final_dir, "vqav2_testdev", "score"), alpha=alpha, notes="Local labels present in JSONL.")


def confidence_level_percent(alpha: float) -> int:
    return int(round((1 - alpha) * PERCENT_SCALE))


def print_results(results: Sequence[PairedResult], alpha: float) -> None:
    ci_label = confidence_level_percent(alpha)
    print(
        f"Paired tests on binary correctness (two-sided, alpha = {alpha})\n"
        f"  t-test H0: mean(d) = 0  |  McNemar H0: P(b01) = P(b10) on discordant pairs\n"
        f"  McNemar chi-square uses continuity correction; p exact when discordant < {MCNEMAR_EXACT_DISCORDANT_THRESHOLD} else chi-square\n"
    )
    header = (
        f"{'Benchmark':<18} {'n':>8} {'Base%':>8} {'Final%':>8} {'Delta pp':>8} "
        f"{f't CI {ci_label}%':>22} {'t':>8} {'p_t':>10} {'Sig_t':>5} "
        f"{'chi2':>8} {f'McN CI {ci_label}%':>22} {'p_McN':>10} {'Sig_M':>5} "
        f"{'b01':>6} {'b10':>6}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        t_test_ci = f"[{result.ci_low_pp:+.2f}, {result.ci_high_pp:+.2f}]"
        mcnemar_ci = f"[{result.mcnemar_ci_low_pp:+.2f}, {result.mcnemar_ci_high_pp:+.2f}]"
        t_test_significant = "Yes" if result.significant else "No"
        mcnemar_significant = "Yes" if result.mcnemar_significant else "No"
        chi2_text = f"{result.mcnemar_chi2:.3f}" if not math.isnan(result.mcnemar_chi2) else "nan"
        print(
            f"{result.benchmark:<18} {result.n:>8} {result.baseline_acc_pct:>8.2f} {result.final_acc_pct:>8.2f} "
            f"{result.delta_pp:>+8.2f} {t_test_ci:>22} {result.t_stat:>8.3f} {result.p_value:>10.4g} {t_test_significant:>5} "
            f"{chi2_text:>8} {mcnemar_ci:>22} {result.mcnemar_p:>10.4g} {mcnemar_significant:>5} "
            f"{result.discordant_base_ok_final_fail:>6} {result.discordant_base_fail_final_ok:>6}"
        )
        if result.mcnemar_exact:
            print(f"  McNemar p-value: exact binomial (discordant < {MCNEMAR_EXACT_DISCORDANT_THRESHOLD})")
        if result.summary_baseline is not None or result.summary_final is not None:
            print(f"  summary.json headline: baseline={result.summary_baseline} final={result.summary_final}")
        if result.notes:
            print(f"  note: {result.notes}")
        print()
    print("b01 = baseline correct, final wrong; b10 = baseline wrong, final correct (discordant pairs)")


def collect_paired_results(baseline_dir: Path, final_dir: Path, alpha: float = 0.05) -> Tuple[List[PairedResult], Optional[str]]:
    """Run POPE / MMBench / SEED paired t-tests and McNemar tests; optional VQAv2 if labeled."""
    baseline_dir = baseline_dir.expanduser().resolve()
    final_dir = final_dir.expanduser().resolve()
    if not baseline_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {baseline_dir}")
    if not final_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {final_dir}")

    results: List[PairedResult] = [
        analyze_pope(baseline_dir, final_dir, alpha),
        analyze_mmbench(baseline_dir, final_dir, alpha),
        analyze_seed(baseline_dir, final_dir, alpha),
    ]
    vqav2_note: Optional[str] = None
    vqav2_result = analyze_vqav2_labeled(baseline_dir, final_dir, alpha)
    if vqav2_result is not None:
        results.append(vqav2_result)
    else:
        vqav2_note = "VQAv2: skipped paired t-test (no labeled JSONL or missing files). Use EvalAI overall scores in the table."
    return results, vqav2_note


def write_paired_export_csv(baseline_dir: Path, final_dir: Path, csv_path: Path) -> Path:
    baseline_dir = baseline_dir.expanduser().resolve()
    final_dir = final_dir.expanduser().resolve()
    csv_path = csv_path.expanduser().resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w") as csv_file:
        csv_file.write("benchmark,item_id,score_baseline,score_final,delta\n")
        baseline_pope = load_pope_pooled(baseline_dir)
        final_pope = load_pope_pooled(final_dir)
        for item_id in sorted(set(baseline_pope) & set(final_pope)):
            baseline_score, final_score = baseline_pope[item_id], final_pope[item_id]
            csv_file.write(f"POPE,{item_id},{baseline_score:.0f},{final_score:.0f},{final_score - baseline_score:.0f}\n")
        for benchmark_name, predictions_file, item_key in (
            ("MMBench", "mmbench_dev_en_predictions.jsonl", "index"),
            ("SEED-Image", "seed_image_predictions.jsonl", "question_id"),
        ):
            baseline_scores = scores_by_key(read_jsonl(baseline_dir / predictions_file), item_key, mmbench_or_seed_correctness_score)
            final_scores = scores_by_key(read_jsonl(final_dir / predictions_file), item_key, mmbench_or_seed_correctness_score)
            for item_id in sorted(set(baseline_scores) & set(final_scores)):
                baseline_score, final_score = baseline_scores[item_id], final_scores[item_id]
                csv_file.write(f"{benchmark_name},{item_id},{baseline_score:.0f},{final_score:.0f},{final_score - baseline_score:.0f}\n")
    return csv_path


def results_to_dataframe(results: Sequence[PairedResult]):
    """Return a pandas DataFrame for notebook display (requires pandas)."""
    import pandas as pd

    table_rows = []
    for result in results:
        table_rows.append(
            {
                "benchmark": result.benchmark,
                "n": result.n,
                "baseline_acc_pct": result.baseline_acc_pct,
                "final_acc_pct": result.final_acc_pct,
                "delta_pp": result.delta_pp,
                "ci_low_pp": result.ci_low_pp,
                "ci_high_pp": result.ci_high_pp,
                "t": result.t_stat,
                "df": result.df,
                "p_value_t": result.p_value,
                "significant_t": result.significant,
                "mcnemar_chi2": result.mcnemar_chi2,
                "mcnemar_stat": result.mcnemar_stat,
                "mcnemar_ci_low_pp": result.mcnemar_ci_low_pp,
                "mcnemar_ci_high_pp": result.mcnemar_ci_high_pp,
                "mcnemar_exact": result.mcnemar_exact,
                "p_value_mcnemar": result.mcnemar_p,
                "significant_mcnemar": result.mcnemar_significant,
                "discordant_base_ok_final_fail": result.discordant_base_ok_final_fail,
                "discordant_base_fail_final_ok": result.discordant_base_fail_final_ok,
                "summary_baseline": result.summary_baseline,
                "summary_final": result.summary_final,
            }
        )
    return pd.DataFrame(table_rows)


def benchmark_slug(benchmark_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", benchmark_name.lower()).strip("-")
    return slug or "benchmark"


def format_ttest_plot_annotation(result: PairedResult, alpha: float) -> str:
    ci_label = confidence_level_percent(alpha)
    significance_label = "significant" if result.significant else "not significant"
    return (
        f"Paired t-test (alpha={alpha})\n"
        f"t({result.df}) = {result.t_stat:.3f}\n"
        f"p = {result.p_value:.4g}\n"
        f"{significance_label}\n"
        f"Delta = {result.delta_pp:+.2f} pp\n"
        f"{ci_label}% CI: [{result.ci_low_pp:+.2f}, {result.ci_high_pp:+.2f}] pp"
    )


def format_mcnemar_plot_annotation(result: PairedResult, alpha: float) -> str:
    ci_label = confidence_level_percent(alpha)
    significance_label = "significant" if result.mcnemar_significant else "not significant"
    p_value_kind = "exact binomial" if result.mcnemar_exact else "chi2 approximation"
    chi2_line = f"chi2 = {result.mcnemar_chi2:.3f}\n" if not math.isnan(result.mcnemar_chi2) else "chi2 = n/a\n"
    return (
        f"McNemar ({p_value_kind})\n"
        f"{chi2_line}"
        f"p = {result.mcnemar_p:.4g}\n"
        f"{significance_label}\n"
        f"b01 = {result.discordant_base_ok_final_fail}, b10 = {result.discordant_base_fail_final_ok}\n"
        f"{ci_label}% CI: [{result.mcnemar_ci_low_pp:+.2f}, {result.mcnemar_ci_high_pp:+.2f}] pp"
    )


def plot_paired_significance(result: PairedResult, *, alpha: float = 0.05, out_path: Optional[Path] = None, show: bool = False, dpi: int = 150):
    """Two-panel figure: t-test null curve and McNemar null distribution."""
    import matplotlib.pyplot as plt

    if result.b is None or result.f is None:
        raise ValueError(f"Missing paired arrays for {result.benchmark!r}")

    sample_size = int(result.n)
    baseline_ok_final_fail = result.discordant_base_ok_final_fail
    baseline_fail_final_ok = result.discordant_base_fail_final_ok
    discordant_pair_count = baseline_ok_final_fail + baseline_fail_final_ok

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    figure.suptitle(f"{result.benchmark} - significance test distributions (n={sample_size:,})", fontsize=13, fontweight="bold")

    ttest_axis = axes[0]
    ttest_axis.set_title("Paired t-test null distribution")
    if result.df > 0 and not math.isnan(result.t_stat):
        t_critical = float(stats.t.ppf(1 - alpha / 2, df=result.df))
        x_limit = max(4.0, abs(result.t_stat) + 1.0, abs(t_critical) + 0.5)
        x_values = np.linspace(-x_limit, x_limit, 1000)
        y_values = stats.t.pdf(x_values, df=result.df)
        tail_mask = np.abs(x_values) >= t_critical

        ttest_axis.plot(x_values, y_values, color="#1f77b4", linewidth=2.0, label=f"t(df={result.df}) null")
        ttest_axis.fill_between(x_values, 0, y_values, where=tail_mask, color="#fdd0a2", alpha=0.45, label=f"two-sided alpha={alpha}")
        ttest_axis.axvline(result.t_stat, color="#d62728", linestyle="--", linewidth=2, label=f"observed t = {result.t_stat:.3f}")
        ttest_axis.axvline(-t_critical, color="#555555", linestyle=":", linewidth=1.5)
        ttest_axis.axvline(t_critical, color="#555555", linestyle=":", linewidth=1.5, label=f"critical +/-{t_critical:.3f}")
        ttest_axis.set_xlabel("t statistic under H0")
        ttest_axis.set_ylabel("Density")
        ttest_axis.legend(loc="upper left", fontsize=8, frameon=True)
    else:
        ttest_axis.text(0.5, 0.5, "Insufficient paired items for t-test curve", ha="center", va="center", transform=ttest_axis.transAxes)
        ttest_axis.set_axis_off()

    if ttest_axis.axison:
        ttest_axis.text(0.98, 0.98, format_ttest_plot_annotation(result, alpha), transform=ttest_axis.transAxes, ha="right", va="top", fontsize=9, family="monospace", bbox=dict(boxstyle="round", facecolor="white", alpha=0.92, edgecolor="#cccccc"))

    mcnemar_axis = axes[1]
    if discordant_pair_count == 0 or math.isnan(result.mcnemar_p):
        mcnemar_axis.set_title("McNemar null distribution")
        mcnemar_axis.text(0.5, 0.5, "No discordant pairs, so McNemar is undefined", ha="center", va="center", transform=mcnemar_axis.transAxes)
        mcnemar_axis.set_axis_off()
    elif result.mcnemar_exact:
        observed_successes = baseline_fail_final_ok
        k_values = np.arange(discordant_pair_count + 1)
        pmf_values = stats.binom.pmf(k_values, discordant_pair_count, 0.5)
        distance_from_center = abs(observed_successes - discordant_pair_count / 2)
        extreme_mask = np.abs(k_values - discordant_pair_count / 2) >= distance_from_center

        mcnemar_axis.set_title(f"McNemar exact null: Binomial(n={discordant_pair_count}, p=0.5)")
        mcnemar_axis.plot(k_values, pmf_values, color="#2ca02c", linewidth=1.8, marker="o", markersize=4, label="null PMF")
        mcnemar_axis.fill_between(k_values, 0, pmf_values, where=extreme_mask, step="mid", color="#c7e9c0", alpha=0.7, label="two-sided tail")
        mcnemar_axis.axvline(observed_successes, color="#d62728", linestyle="--", linewidth=2, label=f"observed b10 = {observed_successes}")
        mcnemar_axis.set_xlabel("b10 count under H0")
        mcnemar_axis.set_ylabel("Probability mass")
        mcnemar_axis.legend(loc="lower right", fontsize=8, frameon=True)
    else:
        chi2_critical = float(stats.chi2.ppf(1 - alpha, df=1))
        x_limit = max(6.0, result.mcnemar_chi2 + 1.0, chi2_critical + 0.5)
        x_values = np.linspace(0, x_limit, 1000)
        y_values = stats.chi2.pdf(x_values, df=1)
        tail_mask = x_values >= chi2_critical

        mcnemar_axis.set_title("McNemar asymptotic null: chi2(df=1)")
        mcnemar_axis.plot(x_values, y_values, color="#2ca02c", linewidth=2.0, label="chi2(df=1) null")
        mcnemar_axis.fill_between(x_values, 0, y_values, where=tail_mask, color="#c7e9c0", alpha=0.7, label=f"right tail alpha={alpha}")
        mcnemar_axis.axvline(result.mcnemar_chi2, color="#d62728", linestyle="--", linewidth=2, label=f"observed chi2 = {result.mcnemar_chi2:.3f}")
        mcnemar_axis.axvline(chi2_critical, color="#555555", linestyle=":", linewidth=1.5, label=f"critical {chi2_critical:.3f}")
        mcnemar_axis.set_xlabel("McNemar test statistic under H0")
        mcnemar_axis.set_ylabel("Density")
        mcnemar_axis.legend(loc="lower right", fontsize=8, frameon=True)

    if mcnemar_axis.axison:
        mcnemar_axis.text(0.98, 0.98, format_mcnemar_plot_annotation(result, alpha), transform=mcnemar_axis.transAxes, ha="right", va="top", fontsize=9, family="monospace", bbox=dict(boxstyle="round", facecolor="white", alpha=0.92, edgecolor="#cccccc"))

    figure.tight_layout(rect=[0, 0, 1, 0.97])
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(figure)
    return figure


def plot_all_significance(results: Sequence[PairedResult], *, alpha: float = 0.05, plot_dir: Optional[Path] = None, show: bool = False, dpi: int = 150) -> List[Path]:
    """Save (or show) one significance figure per benchmark."""
    saved_plot_paths: List[Path] = []
    output_directory = Path(plot_dir) if plot_dir is not None else None
    for result in results:
        plot_output_path = None
        if output_directory is not None:
            plot_output_path = output_directory / f"significance_{benchmark_slug(result.benchmark)}.png"
            saved_plot_paths.append(plot_output_path)
        plot_paired_significance(result, alpha=alpha, out_path=plot_output_path, show=show, dpi=dpi)
    return saved_plot_paths


pope_correct = pope_correctness_score
mmbench_or_seed_correct = mmbench_or_seed_correctness_score
index_key = normalize_index_key
mcnemar_chi2_cc = mcnemar_chi2_continuity_corrected
_benchmark_slug = benchmark_slug
_format_ttest_annotation = format_ttest_plot_annotation
_format_mcnemar_annotation = format_mcnemar_plot_annotation


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired t-test and McNemar test on ReVA benchmark JSONLs.")
    parser.add_argument("--baseline", type=Path, required=True, help="Output dir for ReVA (global only) / Stage 1")
    parser.add_argument("--final", type=Path, required=True, help="Output dir for ReVA (global + regional) / Stage 3")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    parser.add_argument("--export-csv", type=Path, default=None, help="If set, write long-format paired rows (appends per benchmark)")
    parser.add_argument("--plot-dir", type=Path, default=None, help="If set, save per-benchmark significance plots (PNG) here")
    parser.add_argument("--show-plots", action="store_true", help="Display plots interactively (in addition to --plot-dir saves)")
    args = parser.parse_args()

    results, vqav2_note = collect_paired_results(args.baseline, args.final, args.alpha)
    if vqav2_note:
        print(vqav2_note + "\n")

    print_results(results, args.alpha)

    if args.export_csv:
        export_path = write_paired_export_csv(args.baseline, args.final, args.export_csv)
        print(f"Wrote paired rows to {export_path}")

    if args.plot_dir is not None or args.show_plots:
        plot_paths = plot_all_significance(results, alpha=args.alpha, plot_dir=args.plot_dir, show=args.show_plots)
        if plot_paths:
            print("Wrote significance plots:")
            for plot_path in plot_paths:
                print(f"  {plot_path}")


if __name__ == "__main__":
    main()
