"""
Learning with 40 Labels: Semi-Supervised and Active Learning from Scratch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Learning with 40 Labels: Semi-Supervised and Active Learning from Scratch.

MNIST with four labels per digit, a 10,000-image unlabeled pool and 2,000 test images.
One label draw (seed 0) is run end to end; the bracketed ranges are the same numbers
over five label draws (seeds 0-4) measured on this image when the project was calibrated.
"""
import time

RANGES = {
    "supervised": "0.55-0.67, mean 0.63", "supervised_aug": "0.58-0.71, mean 0.66",
    "labelprop_transductive": "0.64-0.78, mean 0.73", "labelprop_harmonic": "0.81-0.90, mean 0.85",
    "labelprop_mlp": "0.80-0.87, mean 0.83", "entropy": "0.60-0.77, mean 0.68", "mutual_information": "0.68-0.84, mean 0.75",
    "vat": "0.61-0.78, mean 0.71", "vat_mi": "0.72-0.88, mean 0.81", "simclr_finetune": "0.58-0.74, mean 0.69",
    "simclr_vat_mi": "0.71-0.86, mean 0.81", "student_distilled": "0.72-0.83, mean 0.77",
    "student_40_labels": "0.59-0.70, mean 0.66",
}


def line(name, label, a):
    print(f"   {label:46s} {a[name]:.3f}   [{RANGES[name]}]")


def main() -> None:
    t0 = time.time()
    rep = forty_labels_report(seed=0)
    a = rep["accuracy"]

    print("1. Forty labels and a baseline (MLP 784-256-128-10, test accuracy)")
    line("supervised", "supervised, 40 labels", a)
    line("supervised_aug", "+ shift/noise augmentation (VRM, eq 19.3)", a)

    print("\n2. Label propagation on a PCA-50 kNN graph (eq 19.26)")
    line("labelprop_transductive", "book iteration, pool accuracy", a)
    line("labelprop_harmonic", "closed form (I - T_UU)^-1 T_UL Y_L, pool acc", a)
    c = rep["labelprop_class_counts"]
    print(f"   pool images per class, iteration:   {c['iteration']}")
    print(f"   pool images per class, closed form: {c['closed_form']}")
    print(f"   the two agree on {rep['harmonic_agreement']:.3f} of the pool; per-step row renormalisation lets one class flood")
    line("labelprop_mlp", "MLP trained on the closed-form labels (test)", a)

    print("\n3. Offline self-training, threshold 0.95 (confirmation bias)")
    for h in rep["self_training"]:
        pa = "-" if h["pseudo_acc"] is None else f"{h['pseudo_acc']:.3f}"
        print(f"   round {h['round']}: pseudo-labels {h['n_selected']:5d}  precision {pa:>5s}  test accuracy {h['test_acc']:.3f}")
    print("   Over five draws round 2 is below round 1 every time, and the pseudo-label precision falls with it.")

    print("\n4. Unlabeled losses in one training loop (600 steps, ramped weight)")
    names = {"entropy": "entropy minimization (eq 19.17)", "mutual_information": "mutual information (eq 19.25)",
             "vat": "VAT (eq 19.28-19.29), xi 1e-3, eps 2", "vat_mi": "VAT + mutual information"}
    for k, label in names.items():
        print(f"   {label:46s} {a[k]:.3f}   [{RANGES[k]}]   smallest class on the pool: {rep['pool_min_class'][k]}")
    print("   Without the marginal-entropy term some classes nearly vanish from the pool predictions;")
    print("   over five draws the smallest class had 38-623 images under entropy minimization and 854-899 under MI.")

    print("\n5. SimCLR pretraining on the pool (eq 19.7), then fine-tuning")
    print("   contrastive loss: " + " -> ".join(f"{l:.2f}" for _, l in rep["simclr_loss"][::2]))
    line("simclr_finetune", "fine-tune on 40 labels", a)
    line("simclr_vat_mi", "fine-tune with VAT + MI", a)
    print("   Contrastive features help a plain fine-tune a little; on top of VAT + MI they add nothing at this scale.")

    print("\n6. Distillation into a 784-64-10 student (eq 19.45, tau 2, teacher: VAT + MI)")
    line("student_40_labels", "student trained on the 40 labels", a)
    line("student_distilled", "student distilled on the pool", a)
    print(f"   student agrees with the teacher on {rep['distill_agreement']:.3f} of the pool")

    print("\n7. Active learning from 20 to 120 labels (5-member ensemble, batches of 20)")
    for s, curve in rep["active_learning"].items():
        pts = "  ".join(f"{n}:{acc:.3f}" for n, acc in curve)
        print(f"   {s:8s} AULC {rep['aulc'][s]:.3f}   {pts}")
    print("   Over five draws: AULC random 0.720, max-entropy 0.700, BALD 0.727; at 120 labels 0.805, 0.790, 0.826.")
    print("   BALD beat random on four of five draws; max-entropy fell below random on four of five.")

    print("\nseconds per block: " + ", ".join(f"{k} {v:.0f}" for k, v in rep["seconds"].items()))
    print(f"study finished in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

