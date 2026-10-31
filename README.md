# Learning with 40 Labels: Semi-Supervised and Active Learning from Scratch

You get MNIST with four labels per digit (40 in total), 10,000 unlabeled images and a held-out test set. How far can you get? This project works through chapter 19 of Murphy's Probabilistic Machine Learning (Learning with Fewer Labeled Examples) on that problem, in tinygrad, with every method written by you. First you build the tools: a seeded split from the dml_data registry, an MLP, a hand-written Adam, data augmentation as vicinal risk minimization (eq 19.3), and one TinyJit training loop that adds any unlabeled loss with a ramped weight. Then you try the chapter's methods one at a time. Label propagation on a kNN graph (eq 19.26), solved by the book's iteration and in closed form, then distilled into an MLP. Offline self-training with a confidence threshold, where you watch confirmation bias happen. Entropy minimization (eq 19.17) next to the input-output mutual information objective (eq 19.25), whose marginal-entropy term stops the predictions on the pool from collapsing onto a few classes. Consistency regularization and virtual adversarial training (eq 19.27-19.29), including why VAT's xi must respect float32 resolution. SimCLR contrastive pretraining on the pool (eq 19.7), fine-tuning, and distillation into a smaller student (eq 19.45). Last comes pool-based active learning, where you compare random sampling, max-entropy and BALD (eq 19.51) acquisition with a five-member ensemble. Headline numbers on the prod CPU image, mean test accuracy over five label draws: supervised MLP on 40 labels 0.63 (0.66 with augmentation); label propagation 0.73 on the pool with the book's per-step renormalisation but 0.85 with the closed-form fixed point, whose mass-conserving transition matrix keeps the classes balanced, and 0.83 once distilled into an MLP; offline self-training gets worse in its second round on every draw; entropy minimization 0.68 against 0.75 for mutual information; VAT 0.71 and VAT plus the mutual-information term 0.81 (0.72 to 0.88 across draws); SimCLR then fine-tuning 0.69; a 784-64-10 student distilled from the VAT teacher 0.77, against 0.66 for the same student trained on the 40 labels. In active learning from 20 to 120 labels, BALD edges past random sampling (area under the learning curve 0.727 vs 0.720, 0.826 vs 0.805 at 120 labels) and max-entropy sampling falls below it (0.700). The assembled script runs the whole study for one label draw in about four minutes.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** load_forty_labels
- [x] **2.** MLP
- [x] **3.** Adam
- [x] **4.** predict_proba
- [x] **5.** augment_batch
- [x] **6.** train_ssl
- [x] **7.** pca_fit
- [x] **8.** knn_graph
- [x] **9.** label_propagation
- [x] **10.** harmonic_solution
- [x] **11.** train_soft
- [x] **12.** select_pseudo_labels
- [x] **13.** self_training_rounds
- [x] **14.** unlabeled_losses
- [x] **15.** mutual_information_loss
- [x] **16.** consistency_loss
- [x] **17.** vat_direction
- [x] **18.** make_vat_loss
- [x] **19.** nt_xent_loss
- [x] **20.** pretrain_simclr
- [x] **21.** with_new_head
- [x] **22.** distill
- [x] **23.** Ensemble
- [x] **24.** fit_ensemble
- [x] **25.** acquisition_scores
- [x] **26.** active_learning_curve

---

Built on Deep-ML.
