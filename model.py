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

# Step 5 - augment_batch
def augment_batch(X, rng, max_shift=2, flip_prob=0.0, noise_std=0.0, cutout=0):
    # Validate the input dimensions.
    X = np.asarray(X, dtype=np.float32)

    if X.ndim != 2 or X.shape[1] != 784:
        raise ValueError("X must have shape (n, 784).")

    if max_shift < 0:
        raise ValueError("max_shift must be non-negative.")

    if not 0.0 <= flip_prob <= 1.0:
        raise ValueError("flip_prob must be between 0 and 1.")

    if cutout < 0 or cutout > 28:
        raise ValueError("cutout must be between 0 and 28.")

    if noise_std < 0:
        raise ValueError("noise_std must be non-negative.")

    n = len(X)

    # Reshape flattened images into 28x28 format.
    # Create a copy so the original input X is never modified.
    images = X.reshape(n, 28, 28).copy()

    # ---------------------------------------------------------
    # 1. Random translation (zero-padded shifting)
    # ---------------------------------------------------------
    if max_shift > 0:
        s = max_shift

        # Add zero-padding on all four sides.
        padded = np.pad(
            images,
            ((0, 0), (s, s), (s, s)),
            mode="constant"
        )

        # Draw random vertical and horizontal crop offsets.
        oy = rng.integers(0, 2 * s + 1, size=n)
        ox = rng.integers(0, 2 * s + 1, size=n)

        # Extract a 28x28 window for every image.
        shifted = np.empty_like(images)

        for i in range(n):
            shifted[i] = padded[
                i,
                oy[i]:oy[i] + 28,
                ox[i]:ox[i] + 28
            ]

        images = shifted

    # ---------------------------------------------------------
    # 2. Random horizontal flipping
    # ---------------------------------------------------------

    # Randomly select images to flip from left to right.
    flip_mask = rng.random(n) < flip_prob

    images[flip_mask] = images[flip_mask, :, ::-1]

    # ---------------------------------------------------------
    # 3. Random cutout augmentation
    # ---------------------------------------------------------
    if cutout > 0:

        # Draw the top-left corner coordinates of each cutout.
        cy = rng.integers(0, 28 - cutout + 1, size=n)
        cx = rng.integers(0, 28 - cutout + 1, size=n)

        # Set the selected square regions to zero.
        for i in range(n):
            images[
                i,
                cy[i]:cy[i] + cutout,
                cx[i]:cx[i] + cutout
            ] = 0.0

    # ---------------------------------------------------------
    # 4. Add Gaussian noise
    # ---------------------------------------------------------
    if noise_std > 0.0:

        noise = rng.normal(
            loc=0.0,
            scale=noise_std,
            size=images.shape
        )

        images += noise

    # ---------------------------------------------------------
    # 5. Clip pixel values and restore flattened dimensions
    # ---------------------------------------------------------

    # Keep all pixel values within [0, 1].
    images = np.clip(images, 0.0, 1.0)

    # Return flattened float32 images.
    return images.reshape(n, 784).astype(np.float32)

# Step 6 - train_ssl
from tinygrad import TinyJit, dtypes


def ramp_weight(t, ramp_steps, lam):
    # Calculate the ramped weight for the unlabeled loss.
    if ramp_steps > 0:
        progress = min(1.0, t / ramp_steps)

        return float(
            lam * math.exp(-5.0 * (1.0 - progress) ** 2)
        )

    # Use the full weight when ramping is disabled.
    return float(lam)


def train_ssl(model, X_lab, y_lab, X_unlab=None, unlabeled_loss=None, lam=1.0,
              ramp_steps=0, steps=300, batch=64, ubatch=256, lr=1e-3,
              augment=None, uaugment=None, weight_decay=0.0, seed=0):

    # Validate the training configuration.
    if steps < 0:
        raise ValueError("steps must be non-negative.")

    if batch <= 0:
        raise ValueError("batch must be positive.")

    if len(X_lab) == 0:
        raise ValueError("X_lab must contain at least one sample.")

    if len(X_lab) != len(y_lab):
        raise ValueError("X_lab and y_lab must have the same length.")

    if unlabeled_loss is not None:
        if X_unlab is None or len(X_unlab) == 0:
            raise ValueError(
                "X_unlab must be provided when unlabeled_loss is used."
            )

        if ubatch <= 0:
            raise ValueError("ubatch must be positive.")

    # Ensure consistent input data types.
    X_lab = np.asarray(X_lab, dtype=np.float32)
    y_lab = np.asarray(y_lab, dtype=np.int32)

    if X_lab.ndim != 2:
        raise ValueError("X_lab must be a two-dimensional array.")

    feature_dim = X_lab.shape[1]

    if unlabeled_loss is not None:
        X_unlab = np.asarray(X_unlab, dtype=np.float32)

        if X_unlab.ndim != 2 or X_unlab.shape[1] != feature_dim:
            raise ValueError("X_unlab must have the same feature dimension as X_lab.")

    # ---------------------------------------------------------
    # 1. Initialize the random generator and Adam optimizer.
    # ---------------------------------------------------------

    rng = np.random.default_rng(seed)

    params = model.parameters()

    opt = Adam(
        params,
        lr=lr,
        weight_decay=weight_decay
    )

    # ---------------------------------------------------------
    # 2. Create the TinyJit training step.
    # ---------------------------------------------------------

    @TinyJit
    def train_step(xl, yl, xu, xu_aug, noise, ramp):

        # Supervised cross-entropy loss.
        sup_loss = model(xl).sparse_categorical_crossentropy(yl)

        # Calculate the unlabeled loss when provided.
        if unlabeled_loss is not None:

            unsup_loss = unlabeled_loss(
                model,
                xu,
                xu_aug,
                noise
            )

        else:
            # No unlabeled objective in supervised-only training.
            unsup_loss = sup_loss * 0.0

        # Combine supervised and ramp-weighted unlabeled losses.
        # Sum ensures a scalar loss for gradient computation.
        loss = (sup_loss + ramp * unsup_loss).sum()

        # Compute gradients with respect to all model parameters.
        grads = loss.gradient(*params)

        # Update model parameters using the custom Adam optimizer.
        opt_tensors = opt.step(grads)

        # Realize both loss terms and all optimizer tensors together.
        Tensor.realize(
            sup_loss,
            unsup_loss,
            *opt_tensors
        )

        # Return the two loss components for recording history.
        return sup_loss, unsup_loss

    # ---------------------------------------------------------
    # 3. Training loop.
    # ---------------------------------------------------------

    history = []

    for t in range(steps):

        # -----------------------------------------------------
        # A. Draw a labeled minibatch.
        # -----------------------------------------------------

        i = rng.integers(
            0,
            len(X_lab),
            size=batch
        )

        xl = X_lab[i]

        # Apply labeled data augmentation when provided.
        if augment is not None:
            xl = augment(xl, rng)

        # -----------------------------------------------------
        # B. Draw unlabeled minibatches and create two views.
        # -----------------------------------------------------

        if unlabeled_loss is not None:

            j = rng.integers(
                0,
                len(X_unlab),
                size=ubatch
            )

            xu_base = X_unlab[j]

            # First unlabeled view.
            if augment is not None:
                xu = augment(xu_base, rng)
            else:
                xu = xu_base

            # Second unlabeled view.
            if uaugment is not None:
                xu_aug = uaugment(xu_base, rng)
            else:
                xu_aug = xu_base

            # Generate standard Gaussian noise for the unlabeled loss.
            noise = rng.normal(
                loc=0.0,
                scale=1.0,
                size=xu.shape
            ).astype(np.float32)

        else:
            # Supply three dummy zero arrays for supervised-only training.
            # Their shapes are (1, D), as required by the specification.
            xu = np.zeros((1, feature_dim), dtype=np.float32)
            xu_aug = np.zeros((1, feature_dim), dtype=np.float32)
            noise = np.zeros((1, feature_dim), dtype=np.float32)

        # -----------------------------------------------------
        # C. Construct the dynamic ramp-weight tensor.
        # -----------------------------------------------------

        ramp = Tensor(
            np.asarray(
                [ramp_weight(t, ramp_steps, lam)],
                dtype=np.float32
            ),
            dtype=dtypes.float32
        )

        # -----------------------------------------------------
        # D. Convert minibatches into tinygrad tensors.
        # -----------------------------------------------------

        xl_t = Tensor(
            np.asarray(xl, dtype=np.float32),
            dtype=dtypes.float32
        )

        yl_t = Tensor(
            np.asarray(y_lab[i], dtype=np.int32),
            dtype=dtypes.int32
        )

        xu_t = Tensor(
            np.asarray(xu, dtype=np.float32),
            dtype=dtypes.float32
        )

        xu_aug_t = Tensor(
            np.asarray(xu_aug, dtype=np.float32),
            dtype=dtypes.float32
        )

        noise_t = Tensor(
            np.asarray(noise, dtype=np.float32),
            dtype=dtypes.float32
        )

        # -----------------------------------------------------
        # E. Execute the TinyJit training step.
        # -----------------------------------------------------

        sup_loss, unsup_loss = train_step(
            xl_t,
            yl_t,
            xu_t,
            xu_aug_t,
            noise_t,
            ramp
        )

        # -----------------------------------------------------
        # F. Record training history.
        # -----------------------------------------------------

        if t % 50 == 0 or t == steps - 1:

            history.append(
                (
                    int(t),
                    float(sup_loss.item()),
                    float(unsup_loss.item())
                )
            )

    # Return the recorded training history.
    return history

# Step 7 - pca_fit
def pca_fit(X, k):
    # Convert input to a NumPy array.
    X = np.asarray(X, dtype=np.float64)

    # Validate the input dimensions.
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    n, d = X.shape

    if n == 0 or d == 0:
        raise ValueError("X must not be empty.")

    if not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_)):
        raise ValueError("k must be an integer.")

    if k <= 0 or k > min(n, d):
        raise ValueError("k must be between 1 and min(n, d).")

    # Step 1: Calculate the mean of each feature.
    mean = X.mean(axis=0)

    # Step 2: Center the data by subtracting the feature means.
    X_centered = X - mean

    # Step 3: Perform Singular Value Decomposition.
    # Vt contains the right singular vectors in descending
    # order of their corresponding singular values.
    _, _, Vt = np.linalg.svd(
        X_centered,
        full_matrices=False
    )

    # Step 4: Select the first k principal components.
    components = Vt[:k].copy()

    # Step 5: Fix the sign ambiguity of each component.
    # Make the entry with the largest absolute magnitude positive.
    for i in range(k):
        max_idx = np.argmax(np.abs(components[i]))

        if components[i, max_idx] < 0:
            components[i] *= -1.0

    # Return the feature mean and principal components.
    return mean, components


def pca_transform(X, mean, components):
    # Convert inputs to NumPy arrays.
    X = np.asarray(X)
    mean = np.asarray(mean)
    components = np.asarray(components)

    # Center the input data using the fitted mean.
    X_centered = X - mean

    # Project the centered data onto the principal components.
    return X_centered @ components.T

# Step 8 - knn_graph
import scipy.sparse as sp


def knn_graph(Z, k=10, sigma=None):
    # Convert input data to float64.
    Z = np.asarray(Z, dtype=np.float64)

    # Validate the input.
    if Z.ndim != 2:
        raise ValueError("Z must be a 2D array.")

    n, d = Z.shape

    if n < 2 or d == 0:
        raise ValueError("Z must contain at least two non-empty samples.")

    if not np.all(np.isfinite(Z)):
        raise ValueError("Z must contain only finite values.")

    if not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_)):
        raise ValueError("k must be an integer.")

    if k < 1 or k >= n:
        raise ValueError("k must be between 1 and n-1.")

    # Store nearest-neighbor indices and distances.
    all_indices = []
    all_distances = []

    # Precompute squared norms of all feature vectors.
    norms = np.sum(Z * Z, axis=1)

    # ---------------------------------------------------------
    # 1. Compute k nearest neighbors in blocks of 2000 rows.
    # ---------------------------------------------------------

    block_size = 2000

    for start in range(0, n, block_size):

        end = min(start + block_size, n)

        # Extract the current block.
        block = Z[start:end]

        # Compute squared Euclidean distances using:
        # ||x-y||² = ||x||² + ||y||² - 2(x @ y)
        d2 = (
            norms[start:end, None]
            + norms[None, :]
            - 2.0 * (block @ Z.T)
        )

        # Remove small negative values caused by floating-point errors.
        np.maximum(d2, 0.0, out=d2)

        # Exclude each point from its own neighbor candidates.
        d2[
            np.arange(end - start),
            np.arange(start, end)
        ] = np.inf

        # Find the indices of the k nearest neighbors.
        neighbor_idx = np.argpartition(
            d2,
            kth=k - 1,
            axis=1
        )[:, :k]

        # Extract the corresponding squared distances.
        neighbor_d2 = np.take_along_axis(
            d2,
            neighbor_idx,
            axis=1
        )

        # Convert squared distances into Euclidean distances.
        neighbor_dist = np.sqrt(neighbor_d2)

        # Store the results for the current block.
        all_indices.append(neighbor_idx)
        all_distances.append(neighbor_dist)

    # Combine results from all blocks.
    indices = np.concatenate(all_indices, axis=0)
    distances = np.concatenate(all_distances, axis=0)

    # ---------------------------------------------------------
    # 2. Determine the Gaussian kernel bandwidth.
    # ---------------------------------------------------------

    if sigma is None:
        sigma = float(np.median(distances))

    else:
        sigma = float(sigma)

        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be a positive finite number.")

    # ---------------------------------------------------------
    # 3. Compute Gaussian similarity weights.
    # ---------------------------------------------------------

    # Flatten nearest-neighbor distances.
    dist = distances.reshape(-1)

    if sigma == 0.0:
        # Handle the degenerate case where all median neighbor
        # distances are zero, avoiding division by zero.
        weights = (dist == 0.0).astype(np.float64)

    else:
        weights = np.exp(
            -(dist ** 2) / (2.0 * sigma ** 2)
        )

    # ---------------------------------------------------------
    # 4. Construct the directed sparse kNN graph.
    # ---------------------------------------------------------

    # Each point contributes exactly k outgoing edges.
    rows = np.repeat(np.arange(n), k)
    cols = indices.reshape(-1)

    W = sp.csr_matrix(
        (weights, (rows, cols)),
        shape=(n, n),
        dtype=np.float64
    )

    # ---------------------------------------------------------
    # 5. Symmetrize the graph.
    # ---------------------------------------------------------

    # An edge exists if either point selects the other
    # as one of its k nearest neighbors.
    W = W.maximum(W.T).tocsr()

    # Ensure the diagonal contains only zeros.
    W.setdiag(0.0)

    # Remove explicitly stored zero entries.
    W.eliminate_zeros()

    # Return a float64 CSR sparse matrix.
    return W.astype(np.float64)

# Step 9 - label_propagation
def label_propagation(W, labeled_idx, y_labeled, n_classes,
                      n_iter=200, renormalize=True):

    # Convert the graph into a float64 CSR sparse matrix.
    W = sp.csr_matrix(W, dtype=np.float64)

    # Validate the graph dimensions.
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be a square adjacency matrix.")

    n = W.shape[0]

    # Validate the number of classes and iterations.
    if not isinstance(n_classes, (int, np.integer)) or n_classes <= 0:
        raise ValueError("n_classes must be a positive integer.")

    if not isinstance(n_iter, (int, np.integer)) or n_iter < 0:
        raise ValueError("n_iter must be a non-negative integer.")

    # Convert labeled indices and labels into NumPy arrays.
    labeled_idx = np.asarray(labeled_idx, dtype=np.int64).reshape(-1)
    y_labeled = np.asarray(y_labeled, dtype=np.int64).reshape(-1)

    if len(labeled_idx) != len(y_labeled):
        raise ValueError("labeled_idx and y_labeled must have equal lengths.")

    if np.any(labeled_idx < 0) or np.any(labeled_idx >= n):
        raise ValueError("labeled_idx contains invalid node indices.")

    if np.any(y_labeled < 0) or np.any(y_labeled >= n_classes):
        raise ValueError("y_labeled contains invalid class indices.")

    # ---------------------------------------------------------
    # 1. Construct the column-normalized transition matrix.
    # ---------------------------------------------------------

    # Calculate the sum of each column.
    col_sum = np.asarray(W.sum(axis=0)).ravel()

    # Calculate the reciprocal of each column sum.
    # Isolated columns receive an inverse of zero.
    inv_col_sum = np.zeros_like(col_sum, dtype=np.float64)

    np.divide(
        1.0,
        col_sum,
        out=inv_col_sum,
        where=col_sum != 0
    )

    # Construct the transition matrix:
    # T[i, j] = W[i, j] / sum_k W[k, j]
    T = W @ sp.diags(
        inv_col_sum,
        format="csr"
    )

    T = T.tocsr()

    # ---------------------------------------------------------
    # 2. Initialize the label distribution matrix.
    # ---------------------------------------------------------

    # Each row represents a node and each column a class.
    # Labeled nodes receive one-hot distributions.
    # Unlabeled nodes initially receive all-zero distributions.
    Y0 = np.zeros(
        (n, n_classes),
        dtype=np.float64
    )

    # Assign the known labels to their corresponding nodes.
    Y0[labeled_idx, y_labeled] = 1.0

    # Keep a working copy of the initial label distributions.
    Y = Y0.copy()

    # ---------------------------------------------------------
    # 3. Iteratively propagate labels through the graph.
    # ---------------------------------------------------------

    for _ in range(n_iter):

        # Propagate class distributions using the transition matrix.
        Y = T @ Y

        # -----------------------------------------------------
        # 4. Renormalize each row, if requested.
        # -----------------------------------------------------

        if renormalize:

            # Calculate the sum of each row.
            row_sum = Y.sum(axis=1, keepdims=True)

            # Normalize nonzero rows so their class probabilities
            # sum to one. All-zero rows remain unchanged.
            Y_normalized = np.zeros_like(Y)

            np.divide(
                Y,
                row_sum,
                out=Y_normalized,
                where=row_sum != 0
            )

            Y = Y_normalized

        # -----------------------------------------------------
        # 5. Clamp labeled nodes to their original labels.
        # -----------------------------------------------------

        # Prevent propagation from changing known labels.
        Y[labeled_idx] = Y0[labeled_idx]

    # Return the propagated class distributions.
    return Y.astype(np.float64)

# Step 10 - harmonic_solution
from scipy.sparse.linalg import spsolve


def harmonic_solution(W, labeled_idx, y_labeled, n_classes):

    # Convert the graph into a float64 CSR sparse matrix.
    W = sp.csr_matrix(W, dtype=np.float64)

    # Validate the graph.
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be a square adjacency matrix.")

    if not np.all(np.isfinite(W.data)):
        raise ValueError("W must contain only finite values.")

    if np.any(W.data < 0):
        raise ValueError("W must contain non-negative weights.")

    n = W.shape[0]

    # Validate the number of classes.
    if not isinstance(n_classes, (int, np.integer)) or isinstance(n_classes, (bool, np.bool_)) or n_classes <= 0:
        raise ValueError("n_classes must be a positive integer.")

    # Convert labeled indices and labels to NumPy arrays.
    labeled_idx = np.asarray(labeled_idx, dtype=np.int64).reshape(-1)
    y_labeled = np.asarray(y_labeled, dtype=np.int64).reshape(-1)

    if len(labeled_idx) != len(y_labeled):
        raise ValueError("labeled_idx and y_labeled must have equal lengths.")

    if len(labeled_idx) == 0:
        raise ValueError("At least one labeled node is required.")

    if np.any(labeled_idx < 0) or np.any(labeled_idx >= n):
        raise ValueError("labeled_idx contains invalid node indices.")

    if len(np.unique(labeled_idx)) != len(labeled_idx):
        raise ValueError("labeled_idx must not contain duplicates.")

    if np.any(y_labeled < 0) or np.any(y_labeled >= n_classes):
        raise ValueError("y_labeled contains invalid class indices.")

    # ---------------------------------------------------------
    # 1. Construct the column-normalized transition matrix.
    # ---------------------------------------------------------

    # Calculate the sum of each column of W.
    col_sum = np.asarray(W.sum(axis=0)).ravel()

    # Compute the reciprocal of each column sum.
    inv_col_sum = np.zeros_like(col_sum, dtype=np.float64)

    np.divide(
        1.0,
        col_sum,
        out=inv_col_sum,
        where=col_sum != 0
    )

    # Construct the transition matrix:
    # T[i, j] = W[i, j] / sum_k W[k, j]
    T = (
        W @ sp.diags(inv_col_sum, format="csr")
    ).tocsr()

    # ---------------------------------------------------------
    # 2. Identify labeled and unlabeled nodes.
    # ---------------------------------------------------------

    L = labeled_idx

    # All remaining nodes, in increasing index order.
    U = np.setdiff1d(
        np.arange(n),
        L
    )

    # ---------------------------------------------------------
    # 3. Initialize the full label distribution matrix.
    # ---------------------------------------------------------

    Y = np.zeros(
        (n, n_classes),
        dtype=np.float64
    )

    # Assign one-hot labels to the labeled nodes.
    Y[L, y_labeled] = 1.0

    # ---------------------------------------------------------
    # 4. Solve the harmonic fixed-point equation.
    # ---------------------------------------------------------

    if len(U) > 0:

        # Extract the unlabeled-to-unlabeled transition matrix.
        T_UU = T[U, :][:, U].tocsc()

        # Extract the unlabeled-to-labeled transition matrix.
        T_UL = T[U, :][:, L].tocsr()

        # Construct the identity matrix.
        I = sp.eye(
            len(U),
            format="csc",
            dtype=np.float64
        )

        # Construct the sparse linear system:
        # (I - T_UU) Y_U = T_UL Y_L
        A = I - T_UU

        # Extract the one-hot labels of the labeled nodes.
        Y_L = Y[L]

        # Calculate the right-hand side.
        rhs = T_UL @ Y_L

        # Solve the sparse linear system without matrix inversion.
        Y_U = spsolve(A, rhs)

        # Ensure a consistent two-dimensional array shape.
        Y_U = np.asarray(
            Y_U,
            dtype=np.float64
        ).reshape(len(U), n_classes)

        # Remove tiny negative values caused by numerical precision.
        Y_U = np.maximum(Y_U, 0.0)

        # -----------------------------------------------------
        # 5. Normalize the unlabeled node distributions.
        # -----------------------------------------------------

        # Calculate the sum of each unlabeled row.
        row_sum = Y_U.sum(axis=1, keepdims=True)

        # Normalize each row so its class probabilities sum to one.
        # Zero-sum rows remain zero.
        np.divide(
            Y_U,
            row_sum,
            out=Y_U,
            where=row_sum > 0
        )

        # Store the normalized harmonic solutions.
        Y[U] = Y_U

    # Return the complete label distribution matrix.
    return Y.astype(np.float64)

# Step 11 - train_soft
def train_soft(model, X, P, steps=500, batch=256, lr=1e-3,
               tau=1.0, seed=0, augment=None):

    # Validate the training configuration.
    if steps <= 0:
        raise ValueError("steps must be positive.")

    if batch <= 0:
        raise ValueError("batch must be positive.")

    if tau <= 0:
        raise ValueError("tau must be positive.")

    # Convert input data and soft targets to float32.
    X = np.asarray(X, dtype=np.float32)
    P = np.asarray(P, dtype=np.float32)

    # Validate input dimensions.
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    if P.ndim != 2:
        raise ValueError("P must be a 2D array.")

    if len(X) == 0 or len(X) != len(P):
        raise ValueError("X and P must contain the same nonzero number of samples.")

    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(P)):
        raise ValueError("X and P must contain only finite values.")

    if np.any(P < 0):
        raise ValueError("Soft targets P must be non-negative.")

    if not np.allclose(P.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Each row of P must sum to one.")

    # Initialize a reproducible random number generator.
    rng = np.random.default_rng(seed)

    # Initialize the custom Adam optimizer.
    params = model.parameters()
    opt = Adam(params, lr=lr)

    # ---------------------------------------------------------
    # 1. Define the TinyJit training step.
    # ---------------------------------------------------------

    @TinyJit
    def train_step(x, p):

        # Calculate temperature-scaled model log probabilities.
        log_probs = (model(x) / tau).log_softmax(-1)

        # Soft-target cross-entropy loss:
        # L = -mean_i sum_c P[i,c] * log_probs[i,c]
        loss = -(p * log_probs).sum(axis=-1).mean()

        # Calculate gradients with respect to model parameters.
        grads = loss.gradient(*params)

        # Update parameters using the custom Adam optimizer.
        updated = opt.step(grads)

        # Realize the loss and all optimizer tensors.
        Tensor.realize(loss, *updated)

        return loss

    # ---------------------------------------------------------
    # 2. Training loop.
    # ---------------------------------------------------------

    last_loss = 0.0

    for _ in range(steps):

        # Randomly sample a minibatch of indices.
        j = rng.integers(
            0,
            len(X),
            size=batch
        )

        # Extract the corresponding input images.
        x_batch = X[j]

        # Apply augmentation if provided.
        if augment is not None:
            x_batch = augment(x_batch, rng)

        # Retrieve the corresponding soft-target distributions.
        p_batch = P[j]

        # Convert minibatches into tinygrad tensors.
        x_t = Tensor(
            np.asarray(x_batch, dtype=np.float32)
        )

        p_t = Tensor(
            np.asarray(p_batch, dtype=np.float32)
        )

        # Execute the TinyJit training step.
        loss = train_step(x_t, p_t)

        # Store the latest loss as a Python float.
        last_loss = float(loss.item())

    # Return the final training loss.
    return last_loss

# Step 12 - select_pseudo_labels
def select_pseudo_labels(probs, threshold, y_true=None):
    # Convert predicted probabilities into a NumPy array.
    probs = np.asarray(probs, dtype=np.float64)

    # Validate the probability array.
    if probs.ndim != 2:
        raise ValueError("probs must be a 2D array of shape (n, n_classes).")

    if probs.shape[1] == 0:
        raise ValueError("probs must contain at least one class.")

    if not np.all(np.isfinite(probs)):
        raise ValueError("probs must contain only finite values.")

    if np.any(probs < 0):
        raise ValueError("Probabilities must be non-negative.")

    if not np.isclose(probs.sum(axis=1), 1.0, atol=1e-5).all():
        raise ValueError("Each probability row must sum to one.")

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    n = len(probs)

    # ---------------------------------------------------------
    # 1. Find the predicted class and its confidence.
    # ---------------------------------------------------------

    # Predicted class for each sample.
    predicted_labels = np.argmax(probs, axis=1)

    # Maximum predicted probability for each sample.
    confidence = np.max(probs, axis=1)

    # ---------------------------------------------------------
    # 2. Select samples meeting the confidence threshold.
    # ---------------------------------------------------------

    # np.flatnonzero returns indices in increasing order.
    idx = np.flatnonzero(confidence >= threshold).astype(np.int64)

    # Obtain pseudo-labels for the selected samples.
    labels = predicted_labels[idx].astype(np.int64)

    # Number of selected pseudo-labels.
    n_selected = int(len(idx))

    # Fraction of the pool selected.
    coverage = float(n_selected / n) if n > 0 else 0.0

    # ---------------------------------------------------------
    # 3. Calculate pseudo-label accuracy for diagnostics.
    # ---------------------------------------------------------

    pseudo_accuracy = None

    if y_true is not None:
        y_true = np.asarray(y_true).reshape(-1)

        if len(y_true) != n:
            raise ValueError("y_true must have the same length as probs.")

        # Calculate accuracy only when at least one sample is selected.
        if n_selected > 0:
            pseudo_accuracy = float(
                np.mean(labels == y_true[idx])
            )

    # ---------------------------------------------------------
    # 4. Return the selection results.
    # ---------------------------------------------------------

    return {
        "idx": idx,
        "labels": labels,
        "n_selected": n_selected,
        "coverage": coverage,
        "accuracy": pseudo_accuracy
    }

# Step 13 - self_training_rounds
def self_training_rounds(
    data,
    make_model,
    rounds=2,
    threshold=0.95,
    init_steps=300,
    steps=400,
    augment=None,
    seed=0
):
    # Extract the labeled, unlabeled, and test datasets.
    X_lab = np.asarray(data["X_lab"], dtype=np.float32)
    y_lab = np.asarray(data["y_lab"], dtype=np.int64)

    X_unlab = np.asarray(data["X_unlab"], dtype=np.float32)

    y_unlab = np.asarray(data["y_unlab"], dtype=np.int64)

    X_test = np.asarray(data["X_test"], dtype=np.float32)
    y_test = np.asarray(data["y_test"], dtype=np.int64)

    # Validate the training configuration.
    if rounds < 0:
        raise ValueError("rounds must be non-negative.")

    if init_steps <= 0 or steps <= 0:
        raise ValueError("Training steps must be positive.")

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    if len(X_lab) != len(y_lab):
        raise ValueError("X_lab and y_lab must have equal lengths.")

    if len(X_unlab) != len(y_unlab):
        raise ValueError("X_unlab and y_unlab must have equal lengths.")

    # Store the diagnostics for every training round.
    history = []

    # ---------------------------------------------------------
    # 1. Round 0: Supervised training using genuine labels only.
    # ---------------------------------------------------------

    # Create a fresh, untrained model.
    model = make_model()

    # Train using only the original labeled dataset.
    train_ssl(
        model,
        X_lab,
        y_lab,
        steps=init_steps,
        augment=augment,
        seed=seed
    )

    # Record the initial model's performance.
    history.append({
        "round": 0,
        "n_selected": 0,
        "pseudo_acc": None,
        "test_acc": accuracy(model, X_test, y_test)
    })

    # ---------------------------------------------------------
    # 2. Perform offline self-training rounds.
    # ---------------------------------------------------------

    for r in range(1, rounds + 1):

        # Generate class probabilities for the entire unlabeled pool.
        probs = predict_proba(model, X_unlab)

        # Select high-confidence pseudo-labels.
        # y_unlab is used ONLY to calculate diagnostic accuracy.
        selected = select_pseudo_labels(
            probs,
            threshold,
            y_true=y_unlab
        )

        # Obtain the selected pool indices and pseudo-labels.
        idx = selected["idx"]
        pseudo_labels = selected["labels"]

        # -----------------------------------------------------
        # 3. Construct the expanded training dataset.
        # -----------------------------------------------------

        # Combine genuine labeled images with selected unlabeled images.
        X_combined = np.concatenate(
            [X_lab, X_unlab[idx]],
            axis=0
        )

        # Combine genuine labels with predicted pseudo-labels.
        y_combined = np.concatenate(
            [y_lab, pseudo_labels],
            axis=0
        )

        # -----------------------------------------------------
        # 4. Train a completely fresh model from scratch.
        # -----------------------------------------------------

        model = make_model()

        train_ssl(
            model,
            X_combined,
            y_combined,
            steps=steps,
            batch=128,
            augment=augment,
            seed=seed
        )

        # -----------------------------------------------------
        # 5. Record diagnostics for the current round.
        # -----------------------------------------------------

        history.append({
            "round": r,
            "n_selected": selected["n_selected"],
            "pseudo_acc": selected["accuracy"],
            "test_acc": accuracy(model, X_test, y_test)
        })

    # Return the complete training history and the final model.
    return history, model

# Step 14 - unlabeled_losses
def unlabeled_losses(logits, T=0.5):
    # Validate the temperature.
    if T <= 0:
        raise ValueError("T must be positive.")

    # Ensure the input contains batch-wise class logits.
    if len(logits.shape) != 2:
        raise ValueError("logits must have shape (batch_size, n_classes).")

    n_classes = logits.shape[-1]

    if n_classes == 0:
        raise ValueError("logits must contain at least one class.")

    # Calculate class probabilities and log probabilities.
    p = logits.softmax(-1)
    log_p = logits.log_softmax(-1)

    # ---------------------------------------------------------
    # 1. Self-training loss (Equation 19.16).
    # ---------------------------------------------------------

    # Obtain the predicted class for each sample.
    # Detach the pseudo-label so gradients do not flow through it.
    hard_targets = (
        logits.argmax(axis=-1)
        .detach()
        .one_hot(n_classes)
        .cast(logits.dtype)
    )

    # Cross-entropy against the detached hard pseudo-label.
    self_training = -(
        hard_targets * log_p
    ).sum(axis=-1).mean()

    # ---------------------------------------------------------
    # 2. Entropy minimization loss (Equation 19.17).
    # ---------------------------------------------------------

    # Minimize the entropy of the predicted class distribution.
    # Gradients flow through both p and log_p.
    entropy = -(
        p * log_p
    ).sum(axis=-1).mean()

    # ---------------------------------------------------------
    # 3. Sharpened self-training loss.
    # ---------------------------------------------------------

    # Construct temperature-sharpened probability targets.
    # Detaching prevents gradients from flowing through the targets.
    q = (
        log_p / T
    ).softmax(-1).detach()

    # Cross-entropy against the detached sharpened targets.
    sharpened = -(
        q * log_p
    ).sum(axis=-1).mean()

    # Return all three scalar loss tensors.
    return {
        "self_training": self_training,
        "entropy": entropy,
        "sharpened": sharpened
    }


def entropy_loss(model, xu, xu_aug, noise):
    # Generate logits for the unlabeled minibatch.
    logits = model(xu)

    # Return the entropy-minimization objective.
    # xu_aug and noise are intentionally unused.
    return unlabeled_losses(logits)["entropy"]

# Step 15 - mutual_information_loss
def mutual_information_loss(model, xu, xu_aug, noise):
    # Generate logits and calculate class probabilities.
    p = model(xu).softmax(-1)

    # Add a small epsilon for numerical stability.
    log_p = (p + 1e-8).log()

    # ---------------------------------------------------------
    # 1. Conditional entropy H(y|x).
    # ---------------------------------------------------------

    # Calculate the entropy of each individual prediction.
    # Then average across the unlabeled minibatch.
    H_cond = -(p * log_p).sum(axis=-1).mean()

    # ---------------------------------------------------------
    # 2. Marginal class probabilities.
    # ---------------------------------------------------------

    # Calculate the average predicted probability of each class.
    p_bar = p.mean(axis=0)

    # Calculate the logarithm of the marginal probabilities.
    log_p_bar = (p_bar + 1e-8).log()

    # ---------------------------------------------------------
    # 3. Marginal entropy H(y).
    # ---------------------------------------------------------

    H_marg = -(p_bar * log_p_bar).sum()

    # ---------------------------------------------------------
    # 4. Negative mutual information objective.
    # ---------------------------------------------------------

    # Minimizing H_cond - H_marg maximizes mutual information.
    return H_cond - H_marg

# Step 16 - consistency_loss
def consistency_loss(p, logits_prime, kind="mse"):
    # Treat clean-input probabilities as constant targets.
    p = p.detach()

    # Calculate probabilities for the perturbed inputs.
    p_prime = logits_prime.softmax(-1)

    # Validate the probability and logits dimensions.
    if p.shape != logits_prime.shape:
        raise ValueError(
            "p and logits_prime must have the same shape."
        )

    # ---------------------------------------------------------
    # 1. Mean squared error consistency loss.
    # ---------------------------------------------------------

    if kind == "mse":
        # Squared L2 distance between the two distributions.
        loss = (p - p_prime) ** 2

        # Sum over classes and average over the batch.
        return loss.sum(axis=-1).mean()

    # ---------------------------------------------------------
    # 2. KL divergence consistency loss.
    # ---------------------------------------------------------

    elif kind == "kl":
        # Calculate KL(p || p_prime).
        # A small epsilon ensures numerical stability.
        log_p = (p + 1e-8).log()

        log_p_prime = logits_prime.log_softmax(-1)

        loss = p * (log_p - log_p_prime)

        # Sum over classes and average over the batch.
        return loss.sum(axis=-1).mean()

    # Reject unsupported consistency loss types.
    else:
        raise ValueError(
            "kind must be either 'mse' or 'kl'."
        )


def make_consistency_loss(kind="mse"):
    # Validate the requested loss type.
    if kind not in ("mse", "kl"):
        raise ValueError(
            "kind must be either 'mse' or 'kl'."
        )

    # Return a trainer-compatible unlabeled loss function.
    def consistency_objective(model, xu, xu_aug, noise):

        # Generate probabilities for the clean unlabeled inputs.
        # Detach them so they serve as fixed reference targets.
        p = model(xu).softmax(-1).detach()

        # Generate logits for the perturbed unlabeled inputs.
        logits_prime = model(xu_aug)

        # Calculate the selected consistency loss.
        return consistency_loss(
            p,
            logits_prime,
            kind=kind
        )

    return consistency_objective

# Step 17 - vat_direction
def vat_direction(model, x, d, xi=1e-6):
    # Validate input dimensions and perturbation magnitude.
    if len(x.shape) != 2 or x.shape != d.shape:
        raise ValueError("x and d must have the same (B, D) shape.")

    if xi <= 0:
        raise ValueError("xi must be positive.")

    # ---------------------------------------------------------
    # 1. Obtain the reference predictions from clean inputs.
    # ---------------------------------------------------------

    # Compute clean-input class probabilities.
    # Detach them so they remain constant during VAT optimization.
    p = model(x).softmax(-1).detach()

    # ---------------------------------------------------------
    # 2. Normalize the random direction.
    # ---------------------------------------------------------

    # Calculate the L2 norm of each row.
    # Add epsilon to prevent division by zero.
    d_norm = (d * d).sum(axis=1, keepdim=True).sqrt() + 1e-12

    # Normalize each perturbation direction to unit L2 norm.
    d_unit = d / d_norm

    # Scale the normalized direction by xi.
    r = xi * d_unit

    # ---------------------------------------------------------
    # 3. Calculate the KL divergence for the perturbed inputs.
    # ---------------------------------------------------------

    # Obtain predictions for x + r.
    logits_prime = model(x + r)

    # Calculate the KL divergence:
    # KL(p || p_prime)
    #
    # Sum across classes and the entire batch to obtain
    # a scalar objective for gradient computation.
    k1 = (
        p * (
            (p + 1e-8).log()
            - logits_prime.log_softmax(-1)
        )
    ).sum()

    # ---------------------------------------------------------
    # 4. Calculate the gradient with respect to r.
    # ---------------------------------------------------------

    # Obtain the gradient of the KL divergence with respect
    # to the perturbation tensor.
    #
    # Return the detached gradient without normalization.
    direction = k1.gradient(r)[0].detach()

    return direction

# Step 18 - make_vat_loss
def make_vat_loss(eps=2.0, xi=1e-3, mi_weight=0.0):

    # Validate the VAT hyperparameters.
    if eps < 0:
        raise ValueError("eps must be non-negative.")

    if xi <= 0:
        raise ValueError("xi must be positive.")

    if mi_weight < 0:
        raise ValueError("mi_weight must be non-negative.")

    # Return a trainer-compatible unlabeled loss function.
    def vat_loss(model, xu, xu_aug, noise):

        # -----------------------------------------------------
        # 1. Obtain the clean reference predictions.
        # -----------------------------------------------------

        # Calculate class probabilities for the clean inputs.
        # Detach them so they remain fixed during VAT optimization.
        p = model(xu).softmax(-1).detach()

        # -----------------------------------------------------
        # 2. Calculate the virtual adversarial direction.
        # -----------------------------------------------------

        # Use the trainer-provided standard-normal noise
        # as the initial random perturbation direction.
        g = vat_direction(
            model,
            xu,
            noise,
            xi=xi
        )

        # -----------------------------------------------------
        # 3. Normalize the adversarial direction.
        # -----------------------------------------------------

        # Calculate the L2 norm independently for every sample.
        g_norm = (
            (g * g).sum(axis=1, keepdim=True).sqrt()
            + 1e-12
        )

        # Normalize the VAT direction.
        delta = g / g_norm

        # Scale the normalized perturbation by eps.
        # The resulting perturbation has approximately L2 norm eps.
        x_adv = xu + eps * delta

        # -----------------------------------------------------
        # 4. Calculate the VAT consistency loss.
        # -----------------------------------------------------

        # Minimize KL divergence between clean and adversarial
        # predictions.
        vat = consistency_loss(
            p,
            model(x_adv),
            kind="kl"
        )

        # -----------------------------------------------------
        # 5. Optionally add mutual-information regularization.
        # -----------------------------------------------------

        if mi_weight > 0.0:

            mi_loss = mutual_information_loss(
                model,
                xu,
                xu_aug,
                noise
            )

            vat = vat + mi_weight * mi_loss

        # Return the combined scalar unlabeled loss.
        return vat

    return vat_loss

# Step 19 - nt_xent_loss
def nt_xent_loss(z1, z2, tau=0.5):
    # Validate the inputs.
    if len(z1.shape) != 2 or len(z2.shape) != 2:
        raise ValueError("z1 and z2 must be 2D embedding tensors.")

    if z1.shape != z2.shape:
        raise ValueError("z1 and z2 must have identical shapes.")

    if z1.shape[0] == 0 or z1.shape[1] == 0:
        raise ValueError("Embedding tensors must not be empty.")

    if tau <= 0:
        raise ValueError("tau must be positive.")

    n = z1.shape[0]

    # ---------------------------------------------------------
    # 1. Concatenate the two augmented-view embeddings.
    # ---------------------------------------------------------

    # The first n rows correspond to z1.
    # The next n rows correspond to z2.
    z = Tensor.cat(z1, z2, dim=0)

    # ---------------------------------------------------------
    # 2. L2-normalize every embedding.
    # ---------------------------------------------------------

    # Calculate the L2 norm of each embedding.
    norms = (
        (z * z).sum(axis=1, keepdim=True).sqrt()
        + 1e-8
    )

    # Normalize each embedding.
    z = z / norms

    # ---------------------------------------------------------
    # 3. Calculate pairwise cosine similarities.
    # ---------------------------------------------------------

    # Since embeddings are normalized, the dot product
    # gives cosine similarity.
    similarity = (z @ z.T) / tau

    # Remove self-similarity by subtracting a large value
    # from the diagonal.
    diagonal_mask = Tensor(
        np.eye(2 * n, dtype=np.float32)
    )

    similarity = similarity - 1e9 * diagonal_mask

    # ---------------------------------------------------------
    # 4. Construct positive-pair targets.
    # ---------------------------------------------------------

    # For the first view, its positive pair is in the second view.
    # For the second view, its positive pair is in the first view.
    positive_indices = np.concatenate([
        np.arange(n, 2 * n),
        np.arange(0, n)
    ])

    # Construct one-hot targets for the positive pairs.
    targets = Tensor(
        np.eye(2 * n, dtype=np.float32)[positive_indices]
    )

    # ---------------------------------------------------------
    # 5. Calculate the NT-Xent cross-entropy loss.
    # ---------------------------------------------------------

    # Compute log-softmax across all non-self similarities.
    log_probs = similarity.log_softmax(-1)

    # Calculate cross-entropy against the positive-pair targets.
    # Sum over candidate embeddings and average over all 2n rows.
    loss = -(targets * log_probs).sum(axis=-1).mean()

    return loss

