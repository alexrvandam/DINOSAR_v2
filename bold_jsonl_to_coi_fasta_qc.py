#!/usr/bin/env python3
"""
bold_jsonl_to_coi_fasta_qc.py

Parse BOLD Systems JSON *lines* (one JSON object per line) and export:
  1) COI FASTA (headers keyed by specimen_id by default)
  2) metadata TSV (specimen_id, processid, bin_uri, genus, species, basecount, qc metrics, etc.)
  3) count tables (per BIN, per species)

This is designed for the situation you showed:
- Some sequences are perfect 658bp
- Some have trailing '-' or many 'N's
- Some are shorter (e.g. 612bp) and/or have alignment gaps

We do *light* QC here. For modeling, you can:
- train on BINs (recommended) or species names
- keep ambiguous bases as 'N' and gaps as '-'
- optionally trim trailing gap/N runs that are obviously pad artifacts

Example:
  python bold_jsonl_to_coi_fasta_qc.py \
    --bold-jsonl Tetramorium.jsonl \
    --out-prefix tetramorium_bold_qc \
    --id-field specimenid \
    --label-field bin_uri \
    --min-effective-len 600 \
    --max-ambig-frac 0.02 \
    --trim-trailing-ambig
"""

import argparse, json, os, re
from collections import Counter, defaultdict
from typing import Dict, Any, Tuple

DNA_VALID = set("ACGT")
AMBIG = set("N-")  # treat other IUPAC as ambiguous too

def clean_seq(seq: str) -> str:
    seq = (seq or "").strip().upper()
    seq = re.sub(r"\s+", "", seq)
    return seq

def trim_trailing_ambig(seq: str, max_trim: int = 120) -> str:
    """
    Trim trailing runs of '-'/'N'/'?' (often padding artifacts).
    max_trim prevents accidental removal of real data in extreme cases.
    """
    if not seq:
        return seq
    i = len(seq)
    trimmed = 0
    while i > 0 and trimmed < max_trim:
        c = seq[i-1]
        if c in "N-?":
            i -= 1
            trimmed += 1
        else:
            break
    return seq[:i]

def qc_metrics(seq: str) -> Tuple[int, int, int, float]:
    """
    Returns:
      effective_len = count of A/C/G/T
      ambig_len     = count of non-ACGT (including N, -, other letters)
      total_len
      ambig_frac = ambig_len / max(total_len,1)
    """
    if not seq:
        return 0, 0, 0, 1.0
    total = len(seq)
    eff = sum(1 for c in seq if c in DNA_VALID)
    amb = total - eff
    amb_frac = amb / total if total else 1.0
    return eff, amb, total, amb_frac

def safe_get(d: Dict[str, Any], key: str, default=None):
    return d.get(key, default)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bold-jsonl", required=True, help="BOLD export with one JSON record per line.")
    ap.add_argument("--out-prefix", required=True, help="Prefix for outputs (no extension).")

    ap.add_argument("--id-field", default="specimenid",
                    help="JSON field to use as specimen id key (e.g. specimenid, sampleid, processid).")
    ap.add_argument("--label-field", default="bin_uri",
                    help="JSON field to use as label (bin_uri recommended; or species).")

    # QC thresholds
    ap.add_argument("--min-effective-len", type=int, default=600,
                    help="Min count of A/C/G/T required to keep a record.")
    ap.add_argument("--max-ambig-frac", type=float, default=0.02,
                    help="Max fraction of ambiguous characters allowed (non-ACGT).")
    ap.add_argument("--trim-trailing-ambig", action="store_true",
                    help="Trim trailing runs of N/-/? (padding artifacts) before QC.")
    ap.add_argument("--require-coi-5p", action="store_true",
                    help="If set, only keep marker_code == COI-5P.")

    args = ap.parse_args()

    out_fasta = args.out_prefix + ".fasta"
    out_tsv   = args.out_prefix + ".tsv"
    out_bin_counts = args.out_prefix + ".counts_per_bin.tsv"
    out_label_counts = args.out_prefix + ".counts_per_label.tsv"

    os.makedirs(os.path.dirname(os.path.abspath(out_fasta)), exist_ok=True)

    n_total = 0
    n_kept = 0

    bin_counter = Counter()
    label_counter = Counter()

    with open(out_fasta, "w") as f_fa, open(out_tsv, "w") as f_tsv:
        header = [
            "specimen_id","processid","record_id","bin_uri","genus","species",
            "label","marker_code","nuc_basecount",
            "effective_len","ambig_len","total_len","ambig_frac",
            "country_iso","inst","sequence_upload_date","insdc_acs",
            "qc_pass"
        ]
        f_tsv.write("\t".join(header) + "\n")

        with open(args.bold_jsonl, "r", encoding="utf-8") as f_in:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                n_total += 1
                try:
                    rec = json.loads(line)
                except Exception:
                    # Some exports contain commas or junk; skip safely
                    continue

                if args.require_coi_5p and safe_get(rec, "marker_code") != "COI-5P":
                    continue

                specimen_id = str(safe_get(rec, args.id_field, "")).strip()
                if not specimen_id:
                    continue

                seq = clean_seq(safe_get(rec, "nuc", ""))
                if args.trim_trailing_ambig:
                    seq = trim_trailing_ambig(seq)

                eff, amb, total, amb_frac = qc_metrics(seq)

                qc_pass = (eff >= args.min_effective_len) and (amb_frac <= args.max_ambig_frac)

                label = str(safe_get(rec, args.label_field, "")).strip()
                if not label:
                    # fallback to species if label_field missing
                    label = str(safe_get(rec, "species", "")).strip()

                row = [
                    specimen_id,
                    str(safe_get(rec, "processid", "")),
                    str(safe_get(rec, "record_id", "")),
                    str(safe_get(rec, "bin_uri", "")),
                    str(safe_get(rec, "genus", "")),
                    str(safe_get(rec, "species", "")),
                    label,
                    str(safe_get(rec, "marker_code", "")),
                    str(safe_get(rec, "nuc_basecount", "")),
                    str(eff), str(amb), str(total), f"{amb_frac:.6f}",
                    str(safe_get(rec, "country_iso", "")),
                    str(safe_get(rec, "inst", "")),
                    str(safe_get(rec, "sequence_upload_date", "")),
                    str(safe_get(rec, "insdc_acs", "")),
                    "1" if qc_pass else "0",
                ]
                f_tsv.write("\t".join(row) + "\n")

                if not qc_pass:
                    continue

                # FASTA: header includes label/taxon for debugging, but first token is specimen_id for easy parsing
                genus = str(safe_get(rec, "genus", "")).strip()
                species = str(safe_get(rec, "species", "")).strip()
                bin_uri = str(safe_get(rec, "bin_uri", "")).strip()

                fa_header = f"{specimen_id}|label={label}|bin={bin_uri}|genus={genus}|species={species}"
                f_fa.write(f">{fa_header}\n{seq}\n")

                n_kept += 1
                bin_counter[bin_uri or "NA"] += 1
                label_counter[label or "NA"] += 1

    # counts
    def write_counts(path: str, counter: Counter, key_name: str):
        with open(path, "w") as f:
            f.write(f"{key_name}\tcount\n")
            for k, v in counter.most_common():
                f.write(f"{k}\t{v}\n")

    write_counts(out_bin_counts, bin_counter, "bin_uri")
    write_counts(out_label_counts, label_counter, "label")

    print(f"✓ wrote: {out_fasta}")
    print(f"✓ wrote: {out_tsv}")
    print(f"✓ wrote: {out_bin_counts}")
    print(f"✓ wrote: {out_label_counts}")
    print(f"kept {n_kept}/{n_total} records after QC")

if __name__ == "__main__":
    main()
