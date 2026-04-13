#!/usr/bin/env python3
"""Evaluate onset detection with Precision/Recall/F1 at various tolerances.

Standard AMT onset evaluation: match predicted onsets to reference onsets
using a time tolerance window, requiring exact pitch match.

Usage:
    python local/evaluate_onsets.py \
        --pred <pred_text_file> --ref <ref_text_file> \
        [--tolerances 10,20,50,100,200,500]
"""

import argparse
import re
from collections import defaultdict

import numpy as np


def parse_pitch(note_token):
    """Extract MIDI pitch number from NOTE_ON_INS_000_PITCH_073 or NOTE_ON_073."""
    m = re.search(r"(\d+)$", note_token)
    return int(m.group(1)) if m else None


def parse_events(text):
    """Parse token sequence into list of (time, pitch) tuples."""
    tokens = text.split()
    if not tokens:
        return []
    # Skip task token
    tokens = tokens[1:]
    events = []
    for i in range(0, len(tokens) - 1, 2):
        t_tok, n_tok = tokens[i], tokens[i + 1]
        if t_tok.startswith("T") and "." in t_tok:
            try:
                t = float(t_tok[1:])
                p = parse_pitch(n_tok)
                if p is not None:
                    events.append((t, p))
            except ValueError:
                pass
    return events


def match_events(pred_events, ref_events, tolerance_sec, pitch_agnostic=False):
    """Greedy matching: for each ref event, find closest unmatched pred within tolerance.

    Returns (TP, FP, FN).
    """
    pred_available = list(range(len(pred_events)))
    tp = 0

    for t_ref, p_ref in sorted(ref_events, key=lambda x: x[0]):
        best_idx = None
        best_dist = float("inf")
        for idx in pred_available:
            t_pred, p_pred = pred_events[idx]
            if not pitch_agnostic and p_pred != p_ref:
                continue
            dist = abs(t_pred - t_ref)
            if dist <= tolerance_sec and dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx is not None:
            pred_available.remove(best_idx)
            tp += 1

    fp = len(pred_available)
    fn = len(ref_events) - tp
    return tp, fp, fn


def prf(tp, fp, fn):
    """Compute precision, recall, F1."""
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def classify_density(n_ref_events):
    """Classify segment by note density."""
    if n_ref_events <= 10:
        return "sparse"
    elif n_ref_events <= 50:
        return "medium"
    else:
        return "dense"


def main():
    parser = argparse.ArgumentParser(description="Evaluate onset P/R/F1")
    parser.add_argument("--pred", required=True, help="Prediction text file")
    parser.add_argument("--ref", required=True, help="Reference text file")
    parser.add_argument(
        "--tolerances",
        default="10,20,50,100,200,500",
        help="Comma-separated tolerance values in ms (default: 10,20,50,100,200,500)",
    )
    args = parser.parse_args()

    tolerances_ms = [int(t) for t in args.tolerances.split(",")]
    tolerances_sec = [t / 1000.0 for t in tolerances_ms]

    # Load data
    preds, refs = {}, {}
    with open(args.pred) as f:
        for line in f:
            p = line.strip().split(maxsplit=1)
            preds[p[0]] = p[1] if len(p) > 1 else ""
    with open(args.ref) as f:
        for line in f:
            p = line.strip().split(maxsplit=1)
            refs[p[0]] = p[1] if len(p) > 1 else ""

    # ---- Per-segment results ----
    # Structure: {tolerance_ms: {utt_id: (tp, fp, fn)}}
    seg_results = {t: {} for t in tolerances_ms}
    seg_results_agnostic = {t: {} for t in tolerances_ms}
    seg_density = {}

    for utt_id in sorted(refs.keys()):
        ref_ev = parse_events(refs[utt_id])
        pred_ev = parse_events(preds.get(utt_id, ""))
        seg_density[utt_id] = classify_density(len(ref_ev))

        for tol_ms, tol_sec in zip(tolerances_ms, tolerances_sec):
            seg_results[tol_ms][utt_id] = match_events(pred_ev, ref_ev, tol_sec)
            seg_results_agnostic[tol_ms][utt_id] = match_events(
                pred_ev, ref_ev, tol_sec, pitch_agnostic=True
            )

    # ---- Print per-segment table at 50ms ----
    print("=" * 100)
    print("PER-SEGMENT RESULTS (onset + pitch match)")
    print("=" * 100)
    header_tols = [f"{t}ms" for t in tolerances_ms]
    print(f"{'SEGMENT':<55s} | {'#P':>4s} | {'#R':>4s} | {'DEN':>6s} | " + " | ".join(f"F1@{t:>4s}" for t in header_tols))
    print("-" * 100)

    for utt_id in sorted(refs.keys()):
        ref_ev = parse_events(refs[utt_id])
        pred_ev = parse_events(preds.get(utt_id, ""))
        den = seg_density[utt_id]
        f1_strs = []
        for tol_ms in tolerances_ms:
            tp, fp, fn = seg_results[tol_ms][utt_id]
            _, _, f1 = prf(tp, fp, fn)
            f1_strs.append(f"{f1*100:7.1f}%")
        print(f"{utt_id:<55s} | {len(pred_ev):4d} | {len(ref_ev):4d} | {den:>6s} | " + " | ".join(f1_strs))

    # ---- Aggregate by density ----
    print("\n" + "=" * 100)
    print("AGGREGATE BY DENSITY (onset + pitch match)")
    print("=" * 100)
    print(f"{'DENSITY':<10s} | {'#seg':>5s} | " + " | ".join(f"{'P@'+t:>9s} {'R@'+t:>9s} {'F1@'+t:>9s}" for t in header_tols))
    print("-" * 100)

    for cat in ["sparse", "medium", "dense", "ALL"]:
        utt_ids = [u for u in refs if (cat == "ALL" or seg_density[u] == cat)]
        if not utt_ids:
            continue
        row = f"{cat:<10s} | {len(utt_ids):5d} | "
        cells = []
        for tol_ms in tolerances_ms:
            tp_sum = sum(seg_results[tol_ms][u][0] for u in utt_ids)
            fp_sum = sum(seg_results[tol_ms][u][1] for u in utt_ids)
            fn_sum = sum(seg_results[tol_ms][u][2] for u in utt_ids)
            p, r, f1 = prf(tp_sum, fp_sum, fn_sum)
            cells.append(f"{p*100:8.1f}% {r*100:8.1f}% {f1*100:8.1f}%")
        print(row + " | ".join(cells))

    # ---- Aggregate pitch-agnostic ----
    print("\n" + "=" * 100)
    print("AGGREGATE BY DENSITY (onset only, pitch-agnostic)")
    print("=" * 100)
    print(f"{'DENSITY':<10s} | {'#seg':>5s} | " + " | ".join(f"{'P@'+t:>9s} {'R@'+t:>9s} {'F1@'+t:>9s}" for t in header_tols))
    print("-" * 100)

    for cat in ["sparse", "medium", "dense", "ALL"]:
        utt_ids = [u for u in refs if (cat == "ALL" or seg_density[u] == cat)]
        if not utt_ids:
            continue
        row = f"{cat:<10s} | {len(utt_ids):5d} | "
        cells = []
        for tol_ms in tolerances_ms:
            tp_sum = sum(seg_results_agnostic[tol_ms][u][0] for u in utt_ids)
            fp_sum = sum(seg_results_agnostic[tol_ms][u][1] for u in utt_ids)
            fn_sum = sum(seg_results_agnostic[tol_ms][u][2] for u in utt_ids)
            p, r, f1 = prf(tp_sum, fp_sum, fn_sum)
            cells.append(f"{p*100:8.1f}% {r*100:8.1f}% {f1*100:8.1f}%")
        print(row + " | ".join(cells))


if __name__ == "__main__":
    main()
