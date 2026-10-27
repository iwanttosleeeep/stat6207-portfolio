#!/usr/bin/env python3
"""
04_evaluate.py
==============
Aggregate every experiment into one comparison, plot learning curves and a
comparison chart, and run an *evidence-based* error analysis on the validation
predictions.

Outputs (experiments/results/)
    comparison.json / comparison.csv  leaderboard + metrics
    best_model.json                   model recommended for final inference
    plots/learning_curves.png         loss & val-F1 per epoch, all models
    plots/model_comparison.png        accuracy / precision / recall / F1 bars
    plots/confusion_matrices.png      confusion matrix grid
    plots/error_gallery.png           validation faces the best model got wrong
    error_analysis.json               per-error records + group statistics

No numbers are invented: everything is recomputed from the saved predictions
and the dataset manifest.
"""
from __future__ import annotations

import csv
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (CLASS_NAMES, LOGS, PLOTS, PREDICTIONS, RESULTS,
                    load_json, manifest_index, face_path, save_json)

from PIL import Image


def collect_results():
    results = []
    for f in ("facenet_results.json", "cnn_results.json"):
        p = os.path.join(RESULTS, f)
        if os.path.exists(p):
            results.extend(load_json(p))
    for r in results:
        r["name"] = r.get("config") or r.get("arch")
    return results


def predictions(result):
    name = result["name"]
    prefix = "facenet_" if str(result.get("model", "")).startswith("FaceNet") else ""
    return load_json(os.path.join(PREDICTIONS, f"{prefix}{name}_val.json"))


# --------------------------------------------------------------------------- #
def plot_learning_curves(results):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for r in results:
        h = r["history"]
        ep = [x["epoch"] for x in h]
        axes[0].plot(ep, [x["train_loss"] for x in h], "--", label=f"{r['name']} train")
        axes[0].plot(ep, [x["val_loss"] for x in h], "-", label=f"{r['name']} val")
        axes[1].plot(ep, [x["val_f1"] for x in h], "-", label=f"{r['name']} val F1")
        axes[1].plot(ep, [x["train_acc"] for x in h], ":", label=f"{r['name']} train acc")
    axes[0].set(title="Loss vs epoch", xlabel="epoch", ylabel="loss")
    axes[1].set(title="Validation F1 / train accuracy vs epoch",
                xlabel="epoch", ylabel="score", ylim=(0, 1.02))
    for a in axes:
        a.grid(alpha=0.3)
        a.legend(fontsize=7, ncol=1)
    fig.suptitle("STAT6207 A3 - Learning curves (same 80/20 split)")
    fig.tight_layout()
    out = os.path.join(PLOTS, "learning_curves.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_comparison(results):
    names = [r["name"] for r in results]
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(names))
    w = 0.2
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(metrics):
        vals = [r["final_val"][m] for r in results]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=m)
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=1)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set(ylim=(0, 1.08), ylabel="score",
           title="Model comparison - validation metrics")
    ax.axhline(0.8, color="grey", ls="--", lw=1, label="F1 target 0.8")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(ncol=5, fontsize=8)
    fig.tight_layout()
    out = os.path.join(PLOTS, "model_comparison.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_confusion_matrices(results):
    n = len(results)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, r in zip(axes, results):
        cm = np.array(r["final_val"]["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=13)
        ax.set(title=f"{r['name']}  (acc={r['final_val']['accuracy']:.2f}, "
                     f"F1={r['final_val']['f1']:.2f})",
               xticks=[0, 1], yticks=[0, 1],
               xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
               xlabel="predicted", ylabel="true")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Confusion matrices (validation)")
    fig.tight_layout()
    out = os.path.join(PLOTS, "confusion_matrices.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
def laplacian_var(img):
    import cv2
    g = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def error_analysis(best_result, index):
    best_name = best_result["name"]
    preds = predictions(best_result)
    rows = []
    for p in preds:
        r = index[p["file"]]
        img = Image.open(face_path(r)).convert("RGB")
        arr = np.asarray(img, dtype=np.float64)
        rows.append({
            **p,
            "age_label": r["fairface_age_label"],
            "mtcnn_prob": float(r["mtcnn_prob"]),
            "blur_laplacian_var": laplacian_var(img),
            "brightness": float(arr.mean()),
        })
    wrong = [r for r in rows if not r["correct"]]
    right = [r for r in rows if r["correct"]]

    def stats(group, key):
        v = [g[key] for g in group]
        return {"mean": float(np.mean(v)), "std": float(np.std(v)),
                "min": float(np.min(v)), "max": float(np.max(v))} if v else {}

    def age_hist(group):
        h = {}
        for g in group:
            h[g["age_label"]] = h.get(g["age_label"], 0) + 1
        return h

    analysis = {
        "model": best_name,
        "n_val": len(rows),
        "n_correct": len(right),
        "n_wrong": len(wrong),
        "error_rate": len(wrong) / len(rows),
        "wrong_files": [r["file"] for r in wrong],
        "group_stats": {
            "wrong": {"blur": stats(wrong, "blur_laplacian_var"),
                      "brightness": stats(wrong, "brightness"),
                      "mtcnn_prob": stats(wrong, "mtcnn_prob"),
                      "age_hist": age_hist(wrong)},
            "correct": {"blur": stats(right, "blur_laplacian_var"),
                        "brightness": stats(right, "brightness"),
                        "mtcnn_prob": stats(right, "mtcnn_prob"),
                        "age_hist": age_hist(right)},
        },
        "records": rows,
        "note": ("Group statistics are descriptive only; with 40 validation "
                 "images differences are not statistically conclusive."),
    }
    save_json(analysis, os.path.join(RESULTS, "error_analysis.json"))

    # error gallery
    if wrong:
        cols = min(5, len(wrong))
        rr = int(np.ceil(len(wrong) / cols))
        fig, axes = plt.subplots(rr, cols, figsize=(2.4 * cols, 3.0 * rr))
        axes = np.atleast_1d(axes).ravel()
        for ax, g in zip(axes, wrong):
            img = Image.open(face_path(index[g["file"]])).convert("RGB")
            ax.imshow(img)
            ax.set_title(f"true={CLASS_NAMES[g['true']]}\n"
                         f"pred={CLASS_NAMES[g['pred']]} "
                         f"p_f={g['prob_female']:.2f}", fontsize=8)
            ax.axis("off")
        for ax in axes[len(wrong):]:
            ax.axis("off")
        fig.suptitle(f"Validation errors - {best_name}")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS, "error_gallery.png"), dpi=150)
        plt.close(fig)
    return analysis


# --------------------------------------------------------------------------- #
def main():
    results = collect_results()
    if not results:
        raise SystemExit("No result JSONs found; run the training scripts first.")

    # leaderboard (selection rule: highest validation F1)
    results_sorted = sorted(results, key=lambda r: (-r["final_val"]["f1"],
                                                    r["final_val"].get("loss", 1e9)))
    best = results_sorted[0]

    plot_learning_curves(results)
    plot_comparison(results)
    plot_confusion_matrices(results)

    index = manifest_index()
    ea = error_analysis(best, index)

    comparison = {
        "selection_rule": "highest validation F1, tie-break lower val loss",
        "best_model": best["name"],
        "models": [{
            "name": r["name"],
            "family": r.get("model"),
            "params": r.get("n_params") or r.get("freeze_info", {}).get("total_params"),
            "trainable_params": r.get("freeze_info", {}).get("trainable_params"),
            "best_epoch": r["best_epoch"],
            "train_time_sec": round(r["train_time_sec"], 1),
            "val": r["final_val"],
        } for r in results_sorted],
    }
    save_json(comparison, os.path.join(RESULTS, "comparison.json"))

    with open(os.path.join(RESULTS, "comparison.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "model", "val_accuracy", "val_precision",
                    "val_recall", "val_f1", "best_epoch", "train_time_sec"])
        for i, r in enumerate(results_sorted, 1):
            v = r["final_val"]
            w.writerow([i, r["name"], f"{v['accuracy']:.4f}",
                        f"{v['precision']:.4f}", f"{v['recall']:.4f}",
                        f"{v['f1']:.4f}", r["best_epoch"],
                        f"{r['train_time_sec']:.1f}"])

    save_json({
        "best_model": best["name"],
        "selection_rule": "highest validation F1, tie-break lower val loss",
        "val_metrics": best["final_val"],
        "val_f1": best["final_val"]["f1"],
        "checkpoint": best.get("checkpoint"),
        "note": "Chosen on the internal 80/20 validation split only.",
    }, os.path.join(RESULTS, "best_model.json"))

    print("Leaderboard (validation):")
    for i, r in enumerate(results_sorted, 1):
        v = r["final_val"]
        print(f"  {i}. {r['name']:14s} acc={v['accuracy']:.3f} "
              f"P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} "
              f"ep={r['best_epoch']} time={r['train_time_sec']:.1f}s")
    print(f"\nBest model: {best['name']} (val F1={best['final_val']['f1']:.3f})")
    print(f"Error analysis: {ea['n_wrong']}/{ea['n_val']} val errors")
    print(f"[out] {RESULTS}")


if __name__ == "__main__":
    main()
