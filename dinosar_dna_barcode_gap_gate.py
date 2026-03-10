#!/usr/bin/env python3
"""
DNA reliability gate for COI clusters (barcode gap first-pass)

This is a conservative, fast gate you can run *before* mPTP/bPTP.

Inputs
- --coi-fasta: sequences keyed by specimen_id (FASTA header token)
- --clusters-tsv: at minimum has specimen_id and cluster_id

Outputs
- barcode_gap_gate.tsv with per-cluster within/between stats and PASS/FAIL
- optional: writes per-cluster FASTA for downstream mPTP/bPTP (if requested)

Notes
- This script assumes sequences are already aligned or at least comparable in length.
  If you want MAFFT alignment, easiest is to align once externally and pass the aligned FASTA.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Tuple, Any

import numpy as np

def read_fasta(path: str) -> Dict[str, str]:
    seqs: Dict[str, str] = {}
    cur_id = None
    buf = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    seqs[cur_id] = "".join(buf)
                cur_id = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
        if cur_id is not None:
            seqs[cur_id] = "".join(buf)
    return seqs

def p_distance(a: str, b: str) -> float:
    a = a.upper()
    b = b.upper()
    L = min(len(a), len(b))
    if L == 0:
        return float("nan")
    dif = 0
    den = 0
    for i in range(L):
        ca = a[i]
        cb = b[i]
        if ca in {"N","-","?"} or cb in {"N","-","?"}:
            continue
        den += 1
        if ca != cb:
            dif += 1
    if den == 0:
        return float("nan")
    return dif / den

def read_clusters_tsv(path: str, specimen_col: str, cluster_col: str) -> Dict[str, str]:
    out = {}
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            sid = str(row.get(specimen_col, "")).strip()
            cid = str(row.get(cluster_col, "")).strip()
            if sid and cid:
                out[sid] = cid
    return out

def main():
    ap = argparse.ArgumentParser(description="COI barcode-gap reliability gate for clusters")
    ap.add_argument("--coi-fasta", required=True)
    ap.add_argument("--clusters-tsv", required=True)
    ap.add_argument("--specimen-id-col", default="specimen_id")
    ap.add_argument("--cluster-id-col", default="novel_cluster_id")
    ap.add_argument("--out-dir", required=True)

    # conservative defaults; tweak for Tetramorium once you see distributions
    ap.add_argument("--max-within", type=float, default=0.02, help="Max within-cluster p-distance allowed for PASS")
    ap.add_argument("--min-between", type=float, default=0.02, help="Min between-cluster nearest-neighbor distance for PASS")
    ap.add_argument("--min-gap", type=float, default=0.01, help="Require between_min - within_max >= min_gap for PASS")

    ap.add_argument("--export-cluster-fastas", action="store_true", help="Write per-cluster FASTA in out-dir/fastas/")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    coi = read_fasta(args.coi_fasta)
    sid_to_cluster = read_clusters_tsv(args.clusters_tsv, args.specimen_id_col, args.cluster_id_col)

    clusters: Dict[str, List[str]] = {}
    for sid, cid in sid_to_cluster.items():
        if sid in coi:
            clusters.setdefault(cid, []).append(sid)

    cluster_ids = sorted(clusters.keys())

    # precompute representative sequence per cluster (medoid by average distance)
    reps: Dict[str, str] = {}
    for cid in cluster_ids:
        sids = clusters[cid]
        if len(sids) == 1:
            reps[cid] = sids[0]
            continue
        # pairwise matrix
        D = np.full((len(sids), len(sids)), np.nan, dtype=float)
        for i,a in enumerate(sids):
            for j,b in enumerate(sids):
                if j <= i:
                    continue
                d = p_distance(coi[a], coi[b])
                D[i,j] = d
                D[j,i] = d
        avg = np.nanmean(D, axis=1)
        reps[cid] = sids[int(np.nanargmin(avg))]

    # compute stats
    rows: List[Dict[str, Any]] = []
    for cid in cluster_ids:
        sids = clusters[cid]
        # within
        within = []
        for i,a in enumerate(sids):
            for b in sids[i+1:]:
                within.append(p_distance(coi[a], coi[b]))
        within = [d for d in within if np.isfinite(d)]
        within_max = float(np.max(within)) if within else 0.0
        within_mean = float(np.mean(within)) if within else 0.0

        # between: nearest neighbor distance from cluster rep to other cluster reps
        rep_sid = reps[cid]
        between = []
        for other in cluster_ids:
            if other == cid:
                continue
            d = p_distance(coi[rep_sid], coi[reps[other]])
            if np.isfinite(d):
                between.append(d)
        between_min = float(np.min(between)) if between else float("inf")

        gap = between_min - within_max if np.isfinite(between_min) else float("nan")
        passed = (within_max <= args.max_within) and (between_min >= args.min_between) and (gap >= args.min_gap)

        rows.append({
            "cluster_id": cid,
            "n_with_dna": len(sids),
            "rep_specimen_id": rep_sid,
            "within_mean": round(within_mean, 6),
            "within_max": round(within_max, 6),
            "between_min": ("" if not np.isfinite(between_min) else round(between_min, 6)),
            "gap": ("" if not np.isfinite(gap) else round(gap, 6)),
            "PASS": bool(passed),
        })

    out_path = os.path.join(args.out_dir, "barcode_gap_gate.tsv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=list(rows[0].keys()) if rows else ["cluster_id"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"✓ wrote: {out_path}")

    if args.export_cluster_fastas:
        fasta_dir = os.path.join(args.out_dir, "fastas")
        os.makedirs(fasta_dir, exist_ok=True)
        for cid, sids in clusters.items():
            fp = os.path.join(fasta_dir, f"{cid}.fasta")
            with open(fp, "w") as f:
                for sid in sids:
                    f.write(f">{sid}\n{coi[sid]}\n")
        print(f"✓ wrote per-cluster FASTA: {fasta_dir}")

if __name__ == "__main__":
    main()
