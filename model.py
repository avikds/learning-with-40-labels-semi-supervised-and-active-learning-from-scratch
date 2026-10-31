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

# Step 2 - MLP
import math
from tinygrad import Tensor


class MLP:
    def __init__(self, sizes, seed=0):
        # Store the architecture as a list of layer widths.
        self.sizes = list(sizes)

        # Initialize a seeded random number generator.
        rng = np.random.default_rng(seed)

        # Store (weight, bias) pairs for every layer.
        self.layers = []

        # Construct the layers in order.
        for i, (a, b) in enumerate(zip(self.sizes[:-1], self.sizes[1:])):

            # Use He initialization for hidden layers and 1/a
            # variance initialization for the final output layer.
            if i == len(self.sizes) - 2:
                std = math.sqrt(1.0 / a)
            else:
                std = math.sqrt(2.0 / a)

            # Initialize weights from a normal distribution.
            W = rng.normal(
                loc=0.0,
                scale=std,
                size=(a, b)
            ).astype(np.float32)

            # Convert weights into realized, contiguous tinygrad tensors.
            W = Tensor(W).contiguous().realize()

            # Initialize biases to zero without consuming random numbers.
            b = Tensor.zeros(b).contiguous().realize()

            # Save the weight and bias tensors.
            self.layers.append((W, b))

    def parameters(self):
        # Return all weights and biases in alternating order.
        # Example: [W0, b0, W1, b1, W2, b2]
        return [
            param
            for W, b in self.layers
            for param in (W, b)
        ]

    def embed(self, x):
        # Apply every hidden layer followed by a ReLU activation.
        for W, b in self.layers[:-1]:
            x = (x @ W + b).relu()

        # Return the final hidden representation.
        return x

    def __call__(self, x):
        # Obtain the final layer's weights and biases.
        W_last, b_last = self.layers[-1]

        # Compute the output logits without applying softmax.
        return self.embed(x) @ W_last + b_last

