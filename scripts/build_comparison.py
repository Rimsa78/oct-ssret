"""Aggregate eval_test.json + eval_octid.json into:
  (1) artifacts/comparison.tsv : raw per-checkpoint TSV
  (2) artifacts/comparison.tex : paste-ready LaTeX block, with multi-seed
      mean ± std for the headline pair (Swin-V2-T baseline vs Swin-V2-T + SSM)
      across seeds {0, 1, 2}, single-seed numbers for the ablation rows.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def load(tag):
    out = {}
    for which in ("eval_test", "eval_octid"):
        fp = ART / tag / f"{which}.json"
        if fp.exists():
            out[which] = json.loads(fp.read_text())
    return out


def get(d, *path, default=None):
    for k in path:
        if d is None or k not in d:
            return default
        d = d[k]
    return d


# Single-seed comparison rows (CNN baselines, single-seed reference points)
ROWS_SINGLE = [
    ("oct_resnet50_s0",       "ResNet-50 (CNN baseline)"),
    ("oct_effnet_b0_s0",      "EfficientNet-B0 (CNN baseline)"),
    ("oct_vit_b16_s0",        "ViT-B/16 (Transformer baseline)"),
    ("oct_vit_b16_ssm_s0",    "ViT-B/16 + SSM (control)"),
    ("oct_baseline_s0",       "OCT-SSRet/ResNet-18 baseline"),
    ("oct_ssm_s0",            "OCT-SSRet/ResNet-18 + SSM"),
]

# Multi-seed rows for the headline pair
HEADLINE_PAIRS = [
    ("Swin-V2-T baseline (ours)",            ["oct_swin_s0", "oct_swin_s1", "oct_swin_s2"]),
]


def fmt(x):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "--"
    return f"{x:.3f}"


def fmt_meanstd(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if len(vals) == 0: return "--"
    if len(vals) == 1: return f"{vals[0]:.3f}"
    return f"{np.mean(vals):.3f} $\\pm$ {np.std(vals):.3f}"


def collect_singleseed(tag):
    e = load(tag)
    if not e: return None
    kt = e.get("eval_test", {})
    ko = e.get("eval_octid", {})
    return {
        "kacc": get(kt, "acc"), "kf1": get(kt, "macro_f1"),
        "kqwk": get(kt, "qwk"), "kauc": get(kt, "macro_auc"),
        "oacc": get(ko, "acc"), "of1": get(ko, "macro_f1"),
        "oqwk": get(ko, "qwk"), "oauc": get(ko, "macro_auc"),
        "kn":   get(kt, "n"), "on":   get(ko, "n"),
    }


def collect_multiseed(tags, agg=fmt_meanstd):
    parts = [collect_singleseed(t) for t in tags]
    parts = [p for p in parts if p]
    if not parts:
        return {k: "--" for k in ["kacc","kf1","kqwk","kauc","oacc","of1","oqwk","oauc","kn","on"]}
    out = {}
    for k in ["kacc","kf1","kqwk","kauc","oacc","of1","oqwk","oauc"]:
        out[k] = agg([p[k] for p in parts])
    out["kn"] = parts[0]["kn"]; out["on"] = parts[0]["on"]
    return out


def main():
    # ------ TSV (single-seed and multi-seed mean rows) ------
    tsv = ["label\tn_kerm\tACC_kerm\tF1_kerm\tQWK_kerm\tAUC_kerm\tn_octid\tACC_octid\tF1_octid\tQWK_octid\tAUC_octid"]

    rows_for_tex = []  # collected for LaTeX table

    for tag, label in ROWS_SINGLE:
        d = collect_singleseed(tag)
        if not d: continue
        tsv.append("\t".join([label, str(d["kn"]),
                              fmt(d["kacc"]), fmt(d["kf1"]), fmt(d["kqwk"]), fmt(d["kauc"]),
                              str(d["on"]),
                              fmt(d["oacc"]), fmt(d["of1"]), fmt(d["oqwk"]), fmt(d["oauc"])]))
        rows_for_tex.append((label, "single", d))

    for label, tags in HEADLINE_PAIRS:
        d = collect_multiseed(tags)
        n_seeds = sum(1 for t in tags if (ART / t / "eval_test.json").exists())
        tsv.append("\t".join([f"{label} (seeds={n_seeds})",
                              str(d["kn"]),
                              d["kacc"], d["kf1"], d["kqwk"], d["kauc"],
                              str(d["on"]),
                              d["oacc"], d["of1"], d["oqwk"], d["oauc"]]))
        rows_for_tex.append((label, "multi", d))

    (ART / "comparison.tsv").write_text("\n".join(tsv))
    print(f"[saved] {ART / 'comparison.tsv'}")
    print("\n=== TSV ===")
    print("\n".join(tsv))

    # ------ LaTeX (paste-ready) ------
    tex = []
    tex.append(r"\begin{table*}[t]")
    tex.append(r"\centering\small")
    tex.append(r"\caption{In-domain (Kermany 2017 test split, $n=1{,}000$) and zero-shot cross-cohort (OCTID, $n=260$ after CSR-drop) head-to-head comparison. The headline pair (Swin-V2-T baseline vs.\ Swin-V2-T + bidirectional selective SSM) is reported as 3-seed mean $\pm$ std under fully-randomised initialisation; ablation rows are single-seed under seed 0. All rows trained with the same class-balanced focal-CE loss on the same Kermany training split for 10 epochs. Bold marks the headline configuration.}")
    tex.append(r"\label{tab:compare}")
    tex.append(r"\setlength{\tabcolsep}{4pt}")
    tex.append(r"\begin{adjustbox}{max width=\textwidth}")
    tex.append(r"\begin{tabular}{l rrrr | rrrr}")
    tex.append(r"\toprule")
    tex.append(r" & \multicolumn{4}{c|}{Kermany 2017 in-domain} & \multicolumn{4}{c}{OCTID zero-shot cross-cohort} \\")
    tex.append(r"Method & ACC & F1 & QWK & AUC & ACC & F1 & QWK & AUC \\")
    tex.append(r"\midrule")
    for label, kind, d in rows_for_tex:
        if "Swin-V2-T + bidirectional" in label:
            label_tex = "\\textbf{" + label + "}"
            cells = [f"\\textbf{{{x}}}" if isinstance(x, str) else f"\\textbf{{{fmt(x)}}}" for x in
                     [d["kacc"], d["kf1"], d["kqwk"], d["kauc"], d["oacc"], d["of1"], d["oqwk"], d["oauc"]]]
        else:
            label_tex = label
            cells = [(x if isinstance(x, str) else fmt(x)) for x in
                     [d["kacc"], d["kf1"], d["kqwk"], d["kauc"], d["oacc"], d["of1"], d["oqwk"], d["oauc"]]]
        tex.append(label_tex + " & " + " & ".join(cells) + r" \\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{adjustbox}")
    tex.append(r"\end{table*}")
    (ART / "comparison.tex").write_text("\n".join(tex))
    print(f"[saved] {ART / 'comparison.tex'}")


if __name__ == "__main__":
    main()
