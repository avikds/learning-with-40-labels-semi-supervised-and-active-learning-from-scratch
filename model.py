"""
Learning with 40 Labels: Semi-Supervised and Active Learning from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_forty_labels
import numpy as np
import dml_data

_MNIST = {}


def load_forty_labels(n_per_class=4, n_unlabeled=10000, n_test=2000, seed=0):
    # Load and cache MNIST so repeated calls do not reload the dataset.
    global _MNIST

    if not _MNIST:
        _MNIST.update(dml_data.load("mnist"))

    x_train = _MNIST["x_train"]
    y_train = _MNIST["y_train"]
    x_test = _MNIST["x_test"]
    y_test = _MNIST["y_test"]

    # Create a reproducible random number generator.
    rng = np.random.default_rng(seed)

    # Step 1: Select n_per_class labeled samples from each digit class.
    # Concatenation preserves class-major ordering (0 through 9).
    labeled_indices = []

    for c in range(10):
        class_indices = np.flatnonzero(y_train == c)

        selected = rng.choice(
            class_indices,
            n_per_class,
            replace=False
        )

        labeled_indices.append(selected)

    labeled = np.concatenate(labeled_indices)

    # Step 2: Select the unlabeled pool from the remaining training samples.
    rest = np.setdiff1d(np.arange(60000), labeled)

    unlabeled = rng.choice(
        rest,
        n_unlabeled,
        replace=False
    )

    # Step 3: Select test samples using an independent, fixed random seed.
    # This ensures every experiment uses exactly the same test subset.
    test_rng = np.random.default_rng(12345)

    test_indices = test_rng.choice(
        10000,
        n_test,
        replace=False
    )

    # Step 4: Flatten images into 784 features, convert to float32,
    # and normalize pixel values to the range [0, 1].
    X_lab = x_train[labeled].reshape(-1, 784).astype(np.float32) / 255.0
    X_unlab = x_train[unlabeled].reshape(-1, 784).astype(np.float32) / 255.0
    X_test = x_test[test_indices].reshape(-1, 784).astype(np.float32) / 255.0

    # Step 5: Return the labeled, unlabeled, and test datasets.
    return {
        "X_lab": X_lab,
        "y_lab": y_train[labeled].astype(np.int64),

        "X_unlab": X_unlab,
        "y_unlab": y_train[unlabeled].astype(np.int64),

        "X_test": X_test,
        "y_test": y_test[test_indices].astype(np.int64)
    }

