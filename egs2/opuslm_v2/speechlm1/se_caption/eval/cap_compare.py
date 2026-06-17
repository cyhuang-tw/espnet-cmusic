#!/usr/bin/env python3
"""Subset-restricted metric comparison for caption-SE trend analysis.

Usage:
  python cap_compare.py <uid_list_file> <label>:<score_dir> [<label>:<score_dir> ...]

uid_list_file: any file whose first whitespace column is a uid (e.g. an inf.scp).
Each score_dir is the .../score/ dir holding {dnsmos,nisqa,utmos,scoreq,speechbert,
lps,spk_sim,lid}.scp (scalar, "<uid> <val>") and wer/{CER,WER}.scp (JSON edit counts).
Means are computed ONLY over the given uids, so all systems are compared on the same set.
"""
import os, sys, json

SCALAR = [("dnsmos/DNSMOS_OVRL", "DNSMOS↑"), ("nisqa/NISQA_MOS", "NISQA↑"),
          ("utmos/UTMOS", "UTMOS↑"), ("scoreq/Scoreq", "SCOREQ↑"),
          ("speechbert/SpeechBERTScore", "SpeechBERT↑"), ("lps/PhonemeSimilarity", "Phoneme-sim↑"),
          ("spk_sim/SpeakerSimilarity", "SpkSim↑"), ("lid/LAcc", "LID↑")]
EDIT = [("wer/CER", "CER%↓"), ("wer/WER", "WER%↓")]


def load_scalar(p, uids):
    if not os.path.exists(p):
        return None, 0
    vals = []
    for ln in open(p):
        x = ln.split()
        if len(x) >= 2 and x[0] in uids:
            try:
                f = float(x[1])
                if f == f:
                    vals.append(f)
            except ValueError:
                pass
    return (sum(vals) / len(vals) if vals else None), len(vals)


def load_edit(p, uids):
    if not os.path.exists(p):
        return None, 0
    err = ref = n = 0
    for ln in open(p):
        parts = ln.split(None, 1)
        if len(parts) < 2 or parts[0] not in uids:
            continue
        try:
            d = json.loads(parts[1])
            err += d["delete"] + d["insert"] + d["replace"]
            ref += d["delete"] + d["replace"] + d["equal"]
            n += 1
        except (ValueError, KeyError):
            pass
    return (100 * err / ref if ref else None), n


def main():
    uid_file = sys.argv[1]
    systems = [a.split(":", 1) for a in sys.argv[2:]]  # [label, score_dir]
    uids = set()
    for ln in open(uid_file):
        x = ln.split()
        if x:
            uids.add(x[0])
    print(f"# {len(uids)} uids from {uid_file}\n")

    labels = [s[0] for s in systems]
    hdr = f"| {'metric':<13} |" + "".join(f" {l:>12} |" for l in labels)
    sep = "|" + "-" * 15 + "|" + "".join("-" * 14 + "|" for _ in labels)
    print(hdr)
    print(sep)
    for mf, lab in SCALAR:
        cells = []
        for _, d in systems:
            m, n = load_scalar(f"{d}/{mf}.scp", uids)
            cells.append(f"{m:.3f}({n})" if m is not None else "—")
        print(f"| {lab:<13} |" + "".join(f" {c:>12} |" for c in cells))
    for mf, lab in EDIT:
        cells = []
        for _, d in systems:
            m, n = load_edit(f"{d}/{mf}.scp", uids)
            cells.append(f"{m:.2f}({n})" if m is not None else "—")
        print(f"| {lab:<13} |" + "".join(f" {c:>12} |" for c in cells))


if __name__ == "__main__":
    main()
