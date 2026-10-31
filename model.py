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

