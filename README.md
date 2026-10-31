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
- [x] **27.** forty_labels_report

## Results

```
1. Forty labels and a baseline (MLP 784-256-128-10, test accuracy)
   supervised, 40 labels                          0.642   [0.55-0.67, mean 0.63]
   + shift/noise augmentation (VRM, eq 19.3)      0.707   [0.58-0.71, mean 0.66]

2. Label propagation on a PCA-50 kNN graph (eq 19.26)
   book iteration, pool accuracy                  0.738   [0.64-0.78, mean 0.73]
   closed form (I - T_UU)^-1 T_UL Y_L, pool acc   0.898   [0.81-0.90, mean 0.85]
   pool images per class, iteration:   [997, 766, 2833, 900, 1074, 527, 1004, 580, 823, 496]
   pool images per class, closed form: [1011, 1200, 971, 1004, 1065, 851, 1130, 968, 806, 994]
   the two agree on 0.777 of the pool; per-step row renormalisation lets one class flood
   MLP trained on the closed-form labels (test)   0.880   [0.80-0.87, mean 0.83]

3. Offline self-training, threshold 0.95 (confirmation bias)
   round 0: pseudo-labels     0  precision     -  test accuracy 0.707
   round 1: pseudo-labels  5295  precision 0.886  test accuracy 0.700
   round 2: pseudo-labels  5422  precision 0.871  test accuracy 0.667
   Over five draws round 2 is below round 1 every time, and the pseudo-label precision falls with it.

4. Unlabeled losses in one training loop (600 steps, ramped weight)
   entropy minimization (eq 19.17)                0.733   [0.60-0.77, mean 0.68]   smallest class on the pool: 541
   mutual information (eq 19.25)                  0.750   [0.68-0.84, mean 0.75]   smallest class on the pool: 749
   VAT (eq 19.28-19.29), xi 1e-3, eps 2           0.783   [0.61-0.78, mean 0.71]   smallest class on the pool: 205
   VAT + mutual information                       0.894   [0.72-0.88, mean 0.81]   smallest class on the pool: 700
   Without the marginal-entropy term some classes nearly vanish from the pool predictions;
   over five draws the smallest class had 38-623 images under entropy minimization and 854-899 under MI.

5. SimCLR pretraining on the pool (eq 19.7), then fine-tuning
   contrastive loss: 6.10 -> 3.51 -> 2.98 -> 2.68 -> 2.55 -> 2.33
   fine-tune on 40 labels                         0.727   [0.58-0.74, mean 0.69]
   fine-tune with VAT + MI                        0.860   [0.71-0.86, mean 0.81]
   Contrastive features help a plain fine-tune a little; on top of VAT + MI they add nothing at this scale.

6. Distillation into a 784-64-10 student (eq 19.45, tau 2, teacher: VAT + MI)
   student trained on the 40 labels               0.701   [0.59-0.70, mean 0.66]
   student distilled on the pool                  0.857   [0.72-0.83, mean 0.77]
   student agrees with the teacher on 0.919 of the pool

7. Active learning from 20 to 120 labels (5-member ensemble, batches of 20)
   random   AULC 0.706   20:0.554  40:0.663  60:0.678  80:0.745  100:0.785  120:0.811
   entropy  AULC 0.715   20:0.554  40:0.670  60:0.708  80:0.739  100:0.770  120:0.848
   bald     AULC 0.714   20:0.554  40:0.629  60:0.729  80:0.759  100:0.791  120:0.820
   Over five draws: AULC random 0.720, max-entropy 0.700, BALD 0.727; at 120 labels 0.805, 0.790, 0.826.
   BALD beat random on four of five draws; max-entropy fell below random on four of five.

seconds per block: baselines 8, label_propagation 17, self_training 14, ssl_losses 45, simclr 23, distillation 9, active_learning 51
study finished in 169 s
```
