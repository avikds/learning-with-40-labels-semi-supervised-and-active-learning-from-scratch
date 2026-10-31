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

# Step 3 - Adam
class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999,
                 eps=1e-8, weight_decay=0.0):

        # Store the parameters and optimizer hyperparameters.
        self.params = list(params)
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.weight_decay = weight_decay

        # Initialize first and second moment estimates with zeros.
        # Each moment tensor has the same shape as its corresponding parameter.
        self.m = [
            Tensor.zeros(*p.shape).contiguous().realize()
            for p in self.params
        ]

        self.v = [
            Tensor.zeros(*p.shape).contiguous().realize()
            for p in self.params
        ]

        # Initialize the optimization step counter.
        self.t = Tensor([0.0]).contiguous().realize()

    def step(self, grads):
        # Ensure that every parameter has a corresponding gradient.
        if len(grads) != len(self.params):
            raise ValueError(
                "Number of gradients must match the number of parameters."
            )

        # Increment the optimization step counter.
        t = self.t.assign(self.t + 1.0)

        # Compute bias-correction factors.
        bc1 = 1.0 - self.b1 ** t
        bc2 = 1.0 - self.b2 ** t

        # Update each parameter using its corresponding gradient.
        for p, g, m, v in zip(self.params, grads, self.m, self.v):

            # Apply L2 weight decay when enabled.
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p

            # Update the first moment estimate.
            m.assign(
                self.b1 * m + (1.0 - self.b1) * g
            )

            # Update the second moment estimate.
            v.assign(
                self.b2 * v + (1.0 - self.b2) * g * g
            )

            # Apply the bias-corrected Adam parameter update.
            p.assign(
                p - self.lr * (m / bc1) / ((v / bc2).sqrt() + self.eps)
            )

        # Return all tensors that the caller must realize.
        return [self.t, *self.params, *self.m, *self.v]

# Step 4 - predict_proba
def predict_proba(model, X, batch=1000):
    # Validate the batch size.
    if batch <= 0:
        raise ValueError("batch must be a positive integer.")

    # Store probability predictions for each batch.
    probabilities = []

    # Process the input data in consecutive chunks.
    for start in range(0, len(X), batch):
        end = start + batch

        # Extract the current batch and convert it into a tinygrad Tensor.
        x_batch = Tensor(X[start:end])

        # Generate logits, apply softmax, and convert predictions to NumPy.
        probs = model(x_batch).softmax(-1).numpy()

        # Store the probabilities for the current batch.
        probabilities.append(probs)

    # Concatenate all batch predictions into one NumPy array.
    if not probabilities:
        raise ValueError("X must contain at least one sample.")

    return np.concatenate(probabilities, axis=0)


def accuracy(model, X, y):
    # Generate class probabilities for the input dataset.
    probabilities = predict_proba(model, X)

    # Check that the number of predictions matches the number of labels.
    if len(probabilities) != len(y):
        raise ValueError("X and y must contain the same number of samples.")

    # Predict the class with the highest probability.
    predictions = np.argmax(probabilities, axis=1)

    # Calculate and return the fraction of correct predictions.
    return float(np.mean(predictions == y))

