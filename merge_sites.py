"""
merge_sites.py — merge CAM, MHA, RSH, UKA unilateral datasets into one manifest.

Run AFTER preprocessing each site with preprocess.py.

Usage:
  python merge_sites.py --roots CAM MHA RSH UKA \
      --processed_roots data/processed_CAM data/processed_MHA data/processed_RSH data/processed_UKA \
      --outdir manifests_all
"""

import os, argparse, pandas as pd, json

def load_site_metadata(site_root, processed_root, site_name):
    ann_path = os.path.join(site_root, "metadata_unilateral", "annotation.csv")
    split_path = os.path.join(site_root, "metadata_unilateral", "split.csv")
    ann = pd.read_csv(ann_path)
    split = pd.read_csv(split_path)
    merged = ann.merge(split, on="UID", how="left")
    merged["center"] = site_name
    merged["processed_root"] = processed_root
    return merged

def merge_sites(sites, processed_roots, outdir):
    os.makedirs(outdir, exist_ok=True)
    all_meta = []
    for site, root in zip(sites, processed_roots):
        site = os.path.join("ODELIA2025/data",site)
        if not os.path.exists(site):
            print(f"[WARN] missing site {site}")
            continue
        df = load_site_metadata(site, root, os.path.basename(site))
        all_meta.append(df)
        print(f"\033[1;34m[OK] loaded {len(df)} entries from {site}\033[00m", )
    merged = pd.concat(all_meta, ignore_index=True)

    items = []
    for _,r in merged.iterrows():
        npz_path = os.path.join(r["processed_root"], f"{r['UID']}.npz")
        if not os.path.exists(npz_path):
            continue
        items.append({
            "npz": npz_path,
            "label": int(r["Lesion"]),
            "uid": r["UID"],
            "center": r["center"],
            "split": r["Split"]
        })

    out_path = os.path.join(outdir, "manifest_all.json")
    json.dump(items, open(out_path, "w"), indent=2)
    print(f"[OK] wrote {len(items)} entries to {out_path}")

    # optionally split into train/val/test manifests for convenience
    for subset in ["train","val","test"]:
        subset_items = [x for x in items if x.get("split","train") == subset]
        if subset_items:
            json.dump(subset_items, open(os.path.join(outdir, f"{subset}_items.json"), "w"), indent=2)
            print(f"  {subset}: {len(subset_items)} items")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, help="site folders (CAM MHA RSH UKA)")
    ap.add_argument("--processed_roots", nargs="+", required=True, help="matching processed roots")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    merge_sites(args.roots, args.processed_roots, args.outdir)
