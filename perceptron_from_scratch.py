import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from itertools import product
from urllib.request import urlopen
import csv
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


class PerceptronFromScratch:
    def __init__(
        self,
        learning_rate=0.01,
        n_iters=200,
        activation="step",
        loss="perceptron",
        optimizer="sgd",
        regularization="none",
        reg_lambda=0.0,
        momentum=0.9,
        batch_size=None,
        shuffle=True,
        verbose=False,
        log_every=25,
    ):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.activation = activation
        self.loss = loss
        self.optimizer = optimizer
        self.regularization = regularization
        self.reg_lambda = reg_lambda
        self.momentum = momentum
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.verbose = verbose
        self.log_every = log_every
        self.weights = None
        self.bias = None
        self.velocity_w = None
        self.velocity_b = 0.0
        self.loss_history = []
        self.train_acc_history = []

        self._validate_hyperparameters()

    def _validate_hyperparameters(self):
        valid_activations = {"step", "sigmoid", "tanh"}
        valid_losses = {"perceptron", "log_loss", "mse"}
        valid_optimizers = {"sgd", "momentum"}
        valid_regularization = {"none", "l1", "l2"}

        if self.activation not in valid_activations:
            raise ValueError(f"activation must be one of {valid_activations}")
        if self.loss not in valid_losses:
            raise ValueError(f"loss must be one of {valid_losses}")
        if self.optimizer not in valid_optimizers:
            raise ValueError(f"optimizer must be one of {valid_optimizers}")
        if self.regularization not in valid_regularization:
            raise ValueError(f"regularization must be one of {valid_regularization}")
        if self.loss == "perceptron" and self.activation != "step":
            raise ValueError("perceptron loss currently supports activation='step' only")
        if self.loss == "log_loss" and self.activation != "sigmoid":
            raise ValueError("log_loss requires activation='sigmoid'")
        if self.reg_lambda < 0:
            raise ValueError("reg_lambda must be >= 0")
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be > 0 when provided")
        if self.log_every <= 0:
            raise ValueError("log_every must be > 0")

    @staticmethod
    def _step(values):
        return (values >= 0).astype(int)

    @staticmethod
    def _sigmoid(values):
        return 1.0 / (1.0 + np.exp(-values))

    @staticmethod
    def _tanh(values):
        return np.tanh(values)

    def _activation(self, values):
        if self.activation == "step":
            return self._step(values)
        if self.activation == "sigmoid":
            return self._sigmoid(values)
        return self._tanh(values)

    def _activation_derivative(self, activated_values):
        if self.activation == "sigmoid":
            return activated_values * (1.0 - activated_values)
        if self.activation == "tanh":
            return 1.0 - activated_values**2
        return np.ones_like(activated_values)

    def _regularization_gradient(self, n_samples):
        if self.regularization == "l2":
            return (self.reg_lambda / n_samples) * self.weights
        if self.regularization == "l1":
            return (self.reg_lambda / n_samples) * np.sign(self.weights)
        return np.zeros_like(self.weights)

    def _compute_loss(self, y_true, linear_output, activated_output):
        eps = 1e-12
        n_samples = y_true.shape[0]

        if self.loss == "perceptron":
            margins = y_true * linear_output
            base_loss = np.mean(np.maximum(0.0, -margins))
        elif self.loss == "log_loss":
            probs = np.clip(activated_output, eps, 1.0 - eps)
            base_loss = -np.mean(
                y_true * np.log(probs) + (1 - y_true) * np.log(1 - probs)
            )
        else:  # mse
            base_loss = np.mean((activated_output - y_true) ** 2)

        reg_term = 0.0
        if self.regularization == "l2":
            reg_term = (self.reg_lambda / (2.0 * n_samples)) * np.sum(self.weights**2)
        elif self.regularization == "l1":
            reg_term = (self.reg_lambda / n_samples) * np.sum(np.abs(self.weights))

        return base_loss + reg_term

    def _apply_optimizer(self, grad_w, grad_b):
        if self.optimizer == "sgd":
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b
            return

        # momentum
        self.velocity_w = self.momentum * self.velocity_w - self.learning_rate * grad_w
        self.velocity_b = self.momentum * self.velocity_b - self.learning_rate * grad_b
        self.weights += self.velocity_w
        self.bias += self.velocity_b

    def _forward(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        activated_output = self._activation(linear_output)
        return linear_output, activated_output

    def _backward(self, X_batch, y_batch, linear_output, activated_output):
        n_samples = X_batch.shape[0]

        if self.loss == "perceptron":
            # Perceptron-style update (sub-gradient style).
            predictions = activated_output
            error = y_batch - predictions
            grad_w = -(np.dot(X_batch.T, error) / n_samples)
            grad_b = -np.mean(error)
        elif self.loss == "log_loss":
            # Backprop for sigmoid + binary cross entropy:
            # dz = a - y
            dz = activated_output - y_batch
            grad_w = np.dot(X_batch.T, dz) / n_samples
            grad_b = np.mean(dz)
        else:  # mse
            # Backprop chain rule:
            # dz = dL/da * da/dz
            dz = (activated_output - y_batch) * self._activation_derivative(activated_output)
            grad_w = np.dot(X_batch.T, dz) / n_samples
            grad_b = np.mean(dz)

        grad_w += self._regularization_gradient(n_samples)
        return grad_w, grad_b

    def fit(self, X, y):
        """
        X shape: (n_samples, n_features)
        y shape: (n_samples,), expected labels in {0, 1}
        """
        n_samples, n_features = X.shape
        y = y.astype(float)
        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0
        self.velocity_w = np.zeros(n_features, dtype=float)
        self.velocity_b = 0.0
        self.loss_history = []
        self.train_acc_history = []
        batch_size = self.batch_size if self.batch_size is not None else n_samples

        for epoch in range(self.n_iters):
            # --- training loop with mini-batches ---
            if self.shuffle:
                indices = np.random.permutation(n_samples)
                X_epoch = X[indices]
                y_epoch = y[indices]
            else:
                X_epoch = X
                y_epoch = y

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_epoch[start:end]
                y_batch = y_epoch[start:end]

                linear_output, activated_output = self._forward(X_batch)
                grad_w, grad_b = self._backward(
                    X_batch,
                    y_batch,
                    linear_output,
                    activated_output,
                )
                self._apply_optimizer(grad_w, grad_b)

            # --- epoch metrics ---
            if self.loss == "perceptron":
                y_for_loss = np.where(y > 0, 1.0, -1.0)
                linear_for_loss, activated_for_loss = self._forward(X)
                loss_value = self._compute_loss(
                    y_for_loss,
                    linear_for_loss,
                    activated_for_loss,
                )
            else:
                linear_for_loss, activated_for_loss = self._forward(X)
                loss_value = self._compute_loss(
                    y,
                    linear_for_loss,
                    activated_for_loss,
                )

            train_acc = self.score(X, y.astype(int))
            self.loss_history.append(loss_value)
            self.train_acc_history.append(train_acc)

            if self.verbose and ((epoch + 1) % self.log_every == 0 or epoch == 0):
                print(
                    f"Epoch {epoch + 1:4d}/{self.n_iters} "
                    f"| loss={loss_value:.6f} | train_acc={train_acc:.4f}"
                )

        return self

    def predict_proba(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        activated_output = self._activation(linear_output)

        if self.activation == "tanh":
            return (activated_output + 1.0) / 2.0
        return activated_output

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y, y_pred),
        }
        return metrics


class MultiLayerPerceptronFromScratch:
    def __init__(
        self,
        input_dim,
        hidden_layers=(4, 4),
        learning_rate=0.1,
        n_epochs=3000,
        batch_size=None,
        hidden_activation="tanh",
        optimizer="sgd",
        momentum=0.9,
        regularization="none",
        reg_lambda=0.0,
        shuffle=True,
        verbose=False,
        log_every=200,
        random_state=42,
    ):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.hidden_activation = hidden_activation
        self.optimizer = optimizer
        self.momentum = momentum
        self.regularization = regularization
        self.reg_lambda = reg_lambda
        self.shuffle = shuffle
        self.verbose = verbose
        self.log_every = log_every
        self.random_state = random_state

        self.weights = []
        self.biases = []
        self.vel_w = []
        self.vel_b = []
        self.loss_history = []
        self.train_acc_history = []

        self._init_params()

    def _init_params(self):
        rng = np.random.default_rng(self.random_state)
        layer_dims = [self.input_dim, *self.hidden_layers, 1]

        self.weights = []
        self.biases = []
        self.vel_w = []
        self.vel_b = []

        for i in range(len(layer_dims) - 1):
            fan_in = layer_dims[i]
            fan_out = layer_dims[i + 1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            w = rng.uniform(-limit, limit, size=(fan_in, fan_out))
            b = np.zeros((1, fan_out))
            self.weights.append(w)
            self.biases.append(b)
            self.vel_w.append(np.zeros_like(w))
            self.vel_b.append(np.zeros_like(b))

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _sigmoid_derivative(a):
        return a * (1.0 - a)

    @staticmethod
    def _tanh(z):
        return np.tanh(z)

    @staticmethod
    def _tanh_derivative(a):
        return 1.0 - a**2

    @staticmethod
    def _relu(z):
        return np.maximum(0.0, z)

    @staticmethod
    def _relu_derivative(z):
        return (z > 0).astype(float)

    def _hidden_activate(self, z):
        if self.hidden_activation == "relu":
            return self._relu(z)
        if self.hidden_activation == "sigmoid":
            return self._sigmoid(z)
        return self._tanh(z)

    def _hidden_activate_derivative(self, a, z):
        if self.hidden_activation == "relu":
            return self._relu_derivative(z)
        if self.hidden_activation == "sigmoid":
            return self._sigmoid_derivative(a)
        return self._tanh_derivative(a)

    def _forward(self, X):
        activations = [X]
        zs = []

        for i in range(len(self.weights) - 1):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            a = self._hidden_activate(z)
            zs.append(z)
            activations.append(a)

        # Output layer: sigmoid for binary classification
        z_out = activations[-1] @ self.weights[-1] + self.biases[-1]
        a_out = self._sigmoid(z_out)
        zs.append(z_out)
        activations.append(a_out)

        return activations, zs

    def _compute_loss(self, y_true, y_prob):
        eps = 1e-12
        y_prob = np.clip(y_prob, eps, 1.0 - eps)
        bce = -np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob))

        reg_term = 0.0
        if self.regularization == "l2":
            reg_term = (self.reg_lambda / (2.0 * y_true.shape[0])) * sum(
                np.sum(w**2) for w in self.weights
            )
        elif self.regularization == "l1":
            reg_term = (self.reg_lambda / y_true.shape[0]) * sum(
                np.sum(np.abs(w)) for w in self.weights
            )

        return bce + reg_term

    def _regularization_grad(self, w, n_samples):
        if self.regularization == "l2":
            return (self.reg_lambda / n_samples) * w
        if self.regularization == "l1":
            return (self.reg_lambda / n_samples) * np.sign(w)
        return np.zeros_like(w)

    def _backward(self, y_true, activations, zs):
        n_samples = y_true.shape[0]
        y_true = y_true.reshape(-1, 1)
        grads_w = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)

        # Output layer gradient (sigmoid + BCE)
        delta = activations[-1] - y_true
        grads_w[-1] = (activations[-2].T @ delta) / n_samples + self._regularization_grad(
            self.weights[-1], n_samples
        )
        grads_b[-1] = np.mean(delta, axis=0, keepdims=True)

        # Hidden layers gradients
        for layer in range(len(self.weights) - 2, -1, -1):
            delta = (delta @ self.weights[layer + 1].T) * self._hidden_activate_derivative(
                activations[layer + 1], zs[layer]
            )
            grads_w[layer] = (
                activations[layer].T @ delta
            ) / n_samples + self._regularization_grad(self.weights[layer], n_samples)
            grads_b[layer] = np.mean(delta, axis=0, keepdims=True)

        return grads_w, grads_b

    def _step(self, grads_w, grads_b):
        if self.optimizer == "momentum":
            for i in range(len(self.weights)):
                self.vel_w[i] = self.momentum * self.vel_w[i] - self.learning_rate * grads_w[i]
                self.vel_b[i] = self.momentum * self.vel_b[i] - self.learning_rate * grads_b[i]
                self.weights[i] += self.vel_w[i]
                self.biases[i] += self.vel_b[i]
        else:
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * grads_w[i]
                self.biases[i] -= self.learning_rate * grads_b[i]

    def fit(self, X, y):
        X = X.astype(float)
        y = y.astype(float)
        n_samples = X.shape[0]
        batch_size = self.batch_size if self.batch_size is not None else n_samples
        self.loss_history = []
        self.train_acc_history = []

        for epoch in range(self.n_epochs):
            if self.shuffle:
                idx = np.random.permutation(n_samples)
                X_epoch = X[idx]
                y_epoch = y[idx]
            else:
                X_epoch = X
                y_epoch = y

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_epoch[start:end]
                y_batch = y_epoch[start:end]

                activations, zs = self._forward(X_batch)
                grads_w, grads_b = self._backward(y_batch, activations, zs)
                self._step(grads_w, grads_b)

            # epoch metrics
            probs = self.predict_proba(X)
            loss = self._compute_loss(y.reshape(-1, 1), probs.reshape(-1, 1))
            acc = accuracy_score(y, self.predict(X))
            self.loss_history.append(loss)
            self.train_acc_history.append(acc)

            if self.verbose and ((epoch + 1) % self.log_every == 0 or epoch == 0):
                print(
                    f"[MLP] Epoch {epoch + 1:4d}/{self.n_epochs} "
                    f"| loss={loss:.6f} | train_acc={acc:.4f}"
                )

        return self

    def predict_proba(self, X):
        activations, _ = self._forward(X.astype(float))
        return activations[-1].reshape(-1)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y, y_pred),
        }


def optimize_hyperparameters(X_train, y_train, X_val, y_val):
    candidate_configs = [
        {
            "learning_rate": 0.05,
            "n_iters": 300,
            "activation": "sigmoid",
            "loss": "log_loss",
            "optimizer": "momentum",
            "regularization": "l2",
            "reg_lambda": 0.001,
            "momentum": 0.9,
            "batch_size": 64,
        },
        {
            "learning_rate": 0.03,
            "n_iters": 350,
            "activation": "sigmoid",
            "loss": "log_loss",
            "optimizer": "sgd",
            "regularization": "l2",
            "reg_lambda": 0.0005,
            "momentum": 0.9,
            "batch_size": 64,
        },
        {
            "learning_rate": 0.05,
            "n_iters": 300,
            "activation": "sigmoid",
            "loss": "mse",
            "optimizer": "momentum",
            "regularization": "none",
            "reg_lambda": 0.0,
            "momentum": 0.9,
            "batch_size": 64,
        },
    ]

    best_model = None
    best_config = None
    best_score = -1.0

    for config in candidate_configs:
        model = PerceptronFromScratch(
            shuffle=True,
            verbose=False,
            log_every=50,
            **config,
        )
        model.fit(X_train, y_train)
        val_acc = model.score(X_val, y_val)

        if val_acc > best_score:
            best_score = val_acc
            best_model = model
            best_config = config

    return best_model, best_config, best_score


def run_demo():
    # Similar setup to the original notebook, but with our own class.
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=12,
        n_redundant=0,
        n_repeated=0,
        n_clusters_per_class=1,
        class_sep=2.0,
        flip_y=0.0,
        n_classes=2,
        random_state=42,
    )

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )
    model, best_config, best_val_acc = optimize_hyperparameters(
        X_train, y_train, X_val, y_val
    )
    # Final fine-tune training on full train split with best config.
    model = PerceptronFromScratch(
        shuffle=True,
        verbose=True,
        log_every=50,
        **best_config,
    )
    model.fit(X_train_full, y_train_full)

    train_metrics = model.evaluate(X_train_full, y_train_full)
    test_metrics = model.evaluate(X_test, y_test)

    print("Perceptron from scratch results")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best config: {best_config}")
    print("\nTrain metrics:")
    print(f"  Accuracy : {train_metrics['accuracy']:.4f}")
    print(f"  Precision: {train_metrics['precision']:.4f}")
    print(f"  Recall   : {train_metrics['recall']:.4f}")
    print(f"  F1-score : {train_metrics['f1']:.4f}")
    print(f"  Confusion matrix:\n{train_metrics['confusion_matrix']}")
    print("\nTest metrics:")
    print(f"  Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall   : {test_metrics['recall']:.4f}")
    print(f"  F1-score : {test_metrics['f1']:.4f}")
    print(f"  Confusion matrix:\n{test_metrics['confusion_matrix']}")
    print(f"Final loss    : {model.loss_history[-1]:.6f}")


def run_experiment_grid():
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=12,
        n_redundant=0,
        n_repeated=0,
        n_clusters_per_class=1,
        class_sep=2.0,
        flip_y=0.0,
        n_classes=2,
        random_state=42,
    )

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    activations = ["sigmoid", "tanh"]
    losses = ["log_loss", "mse"]
    optimizers = ["sgd", "momentum"]
    regularizations = ["none", "l1", "l2"]

    results = []

    for activation, loss, optimizer, regularization in product(
        activations, losses, optimizers, regularizations
    ):
        # Keep only valid combinations.
        if loss == "log_loss" and activation != "sigmoid":
            continue

        model = PerceptronFromScratch(
            learning_rate=0.05,
            n_iters=300,
            activation=activation,
            loss=loss,
            optimizer=optimizer,
            regularization=regularization,
            reg_lambda=0.001 if regularization != "none" else 0.0,
            momentum=0.9,
        )
        model.fit(X_train, y_train)

        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        final_loss = model.loss_history[-1]

        results.append(
            {
                "activation": activation,
                "loss": loss,
                "optimizer": optimizer,
                "regularization": regularization,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "final_loss": final_loss,
            }
        )

    results.sort(key=lambda item: item["test_acc"], reverse=True)

    print("\n=== Experiment Grid Results (sorted by test accuracy) ===")
    header = (
        f"{'activation':<10} {'loss':<10} {'optimizer':<9} "
        f"{'reg':<5} {'train_acc':>9} {'test_acc':>9} {'final_loss':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{row['activation']:<10} {row['loss']:<10} {row['optimizer']:<9} "
            f"{row['regularization']:<5} {row['train_acc']:>9.4f} "
            f"{row['test_acc']:>9.4f} {row['final_loss']:>12.6f}"
        )


def train_with_logic_gates():
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )

    gates = {
        "AND": np.array([0, 0, 0, 1]),
        "OR": np.array([0, 1, 1, 1]),
        "NAND": np.array([1, 1, 1, 0]),
        "NOR": np.array([1, 0, 0, 0]),
        "XOR": np.array([0, 1, 1, 0]),
    }

    print("\n=== Logic Gates Training ===")
    print("Note: A single-layer perceptron should fail on XOR (not linearly separable).")

    for gate_name, y in gates.items():
        model = PerceptronFromScratch(
            learning_rate=0.1,
            n_iters=500,
            activation="sigmoid",
            loss="log_loss",
            optimizer="momentum",
            regularization="none",
            reg_lambda=0.0,
            momentum=0.9,
            batch_size=4,
            shuffle=True,
            verbose=False,
        )
        model.fit(X, y)
        preds = model.predict(X)
        metrics = model.evaluate(X, y)

        print(f"\n{gate_name} gate")
        print(f"  Targets    : {y.tolist()}")
        print(f"  Predictions: {preds.tolist()}")
        print(f"  Accuracy   : {metrics['accuracy']:.4f}")
        print(f"  Final loss : {model.loss_history[-1]:.6f}")


def train_multilayer_on_logic_gates():
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    gates = {
        "AND": np.array([0, 0, 0, 1]),
        "OR": np.array([0, 1, 1, 1]),
        "NAND": np.array([1, 1, 1, 0]),
        "NOR": np.array([1, 0, 0, 0]),
        "XOR": np.array([0, 1, 1, 0]),
    }

    print("\n=== Multi-Layer Perceptron (MLP) Logic Gates Training ===")
    print("Using hidden layers to learn non-linear decision boundaries (including XOR).")

    for gate_name, y in gates.items():
        mlp = MultiLayerPerceptronFromScratch(
            input_dim=2,
            hidden_layers=(4, 4),
            learning_rate=0.1,
            n_epochs=5000,
            batch_size=4,
            hidden_activation="tanh",
            optimizer="momentum",
            momentum=0.9,
            regularization="none",
            reg_lambda=0.0,
            shuffle=True,
            verbose=False,
            random_state=42,
        )
        mlp.fit(X, y)
        preds = mlp.predict(X)
        metrics = mlp.evaluate(X, y)

        print(f"\n{gate_name} gate (MLP)")
        print(f"  Targets    : {y.tolist()}")
        print(f"  Predictions: {preds.tolist()}")
        print(f"  Accuracy   : {metrics['accuracy']:.4f}")
        print(f"  Final loss : {mlp.loss_history[-1]:.6f}")


def load_stooq_ohlcv(ticker="AAPL.US"):
    """
    Download daily OHLCV data from Stooq as CSV.
    """
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}&i=d"
    with urlopen(url, timeout=15) as response:
        content = response.read().decode("utf-8")

    reader = csv.DictReader(content.splitlines())
    rows = [row for row in reader if row.get("Close") not in (None, "", "null")]
    if len(rows) < 300:
        raise ValueError(f"Not enough rows downloaded for {ticker}.")

    # Sort by date ascending if needed
    rows = sorted(rows, key=lambda r: r["Date"])
    return rows


def build_financial_features(rows):
    """
    Build a next-day direction classification dataset from OHLCV.
    Features are based on returns, momentum, range, volume and rolling stats.
    """
    close = np.array([float(r["Close"]) for r in rows], dtype=float)
    open_ = np.array([float(r["Open"]) for r in rows], dtype=float)
    high = np.array([float(r["High"]) for r in rows], dtype=float)
    low = np.array([float(r["Low"]) for r in rows], dtype=float)
    volume = np.array([float(r["Volume"]) for r in rows], dtype=float)

    # Basic transforms
    ret_1 = (close[1:] - close[:-1]) / (close[:-1] + 1e-12)
    log_vol = np.log(volume + 1.0)
    intraday = (close - open_) / (open_ + 1e-12)
    hl_range = (high - low) / (close + 1e-12)

    # Rolling helper
    def rolling_mean(x, window):
        out = np.full_like(x, np.nan, dtype=float)
        for i in range(window - 1, len(x)):
            out[i] = np.mean(x[i - window + 1 : i + 1])
        return out

    def rolling_std(x, window):
        out = np.full_like(x, np.nan, dtype=float)
        for i in range(window - 1, len(x)):
            out[i] = np.std(x[i - window + 1 : i + 1])
        return out

    # Features aligned on day t (predict t+1 movement)
    close_ma5 = rolling_mean(close, 5)
    close_ma20 = rolling_mean(close, 20)
    vol_std10 = rolling_std(ret_1, 10)  # shorter array, handled below
    vol_std20 = rolling_std(ret_1, 20)
    vol_ma10 = rolling_mean(log_vol, 10)
    vol_ma20 = rolling_mean(log_vol, 20)

    # Align ret-based rolling arrays to close index by shifting +1
    vol_std10_aligned = np.full_like(close, np.nan, dtype=float)
    vol_std20_aligned = np.full_like(close, np.nan, dtype=float)
    vol_std10_aligned[1:] = vol_std10
    vol_std20_aligned[1:] = vol_std20

    ret_1_aligned = np.full_like(close, np.nan, dtype=float)
    ret_1_aligned[1:] = ret_1

    ret_5 = np.full_like(close, np.nan, dtype=float)
    ret_10 = np.full_like(close, np.nan, dtype=float)
    ret_5[5:] = (close[5:] - close[:-5]) / (close[:-5] + 1e-12)
    ret_10[10:] = (close[10:] - close[:-10]) / (close[:-10] + 1e-12)

    # Trend feature
    ma_ratio = close_ma5 / (close_ma20 + 1e-12) - 1.0

    features = np.column_stack(
        [
            ret_1_aligned,
            ret_5,
            ret_10,
            intraday,
            hl_range,
            close / (close_ma5 + 1e-12) - 1.0,
            ma_ratio,
            vol_std10_aligned,
            vol_std20_aligned,
            log_vol - vol_ma10,
            log_vol - vol_ma20,
        ]
    )

    # Label: next-day direction
    y = (close[1:] > close[:-1]).astype(int)
    X = features[:-1]

    valid = ~np.isnan(X).any(axis=1)
    X = X[valid]
    y = y[valid]
    return X, y


def train_mlp_on_financial_data(ticker="AAPL.US"):
    print("\n=== Financial Dataset Training (MLP) ===")
    print(f"Ticker: {ticker}")

    try:
        rows = load_stooq_ohlcv(ticker=ticker)
        X, y = build_financial_features(rows)
        source = "stooq"
    except Exception as exc:
        print(f"Financial data download failed ({exc}). Falling back to synthetic market-like data.")
        rng = np.random.default_rng(42)
        n = 2500
        # Synthetic market-like returns with drift + regime noise
        base = rng.normal(0.0004, 0.012, size=n)
        regime = np.where(np.arange(n) % 200 < 100, 0.0003, -0.0001)
        returns = base + regime
        price = 100 * np.cumprod(1 + returns)

        rows = []
        for i in range(n):
            c = price[i]
            o = c / (1 + rng.normal(0, 0.003))
            h = max(o, c) * (1 + abs(rng.normal(0, 0.002)))
            l = min(o, c) * (1 - abs(rng.normal(0, 0.002)))
            v = 1_000_000 + 250_000 * abs(rng.normal())
            rows.append(
                {
                    "Date": f"d{i}",
                    "Open": str(o),
                    "High": str(h),
                    "Low": str(l),
                    "Close": str(c),
                    "Volume": str(v),
                }
            )
        X, y = build_financial_features(rows)
        source = "synthetic"

    # Time-aware split (no shuffling)
    n_total = X.shape[0]
    train_end = int(0.7 * n_total)
    val_end = int(0.85 * n_total)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    candidate_configs = [
        {"hidden_layers": (8, 4), "learning_rate": 0.03, "optimizer": "momentum", "regularization": "l2", "reg_lambda": 0.0005},
        {"hidden_layers": (16, 8), "learning_rate": 0.02, "optimizer": "momentum", "regularization": "l2", "reg_lambda": 0.0010},
        {"hidden_layers": (12, 6), "learning_rate": 0.02, "optimizer": "sgd", "regularization": "none", "reg_lambda": 0.0},
    ]

    best_model = None
    best_config = None
    best_val_acc = -1.0

    for cfg in candidate_configs:
        mlp = MultiLayerPerceptronFromScratch(
            input_dim=X_train.shape[1],
            hidden_layers=cfg["hidden_layers"],
            learning_rate=cfg["learning_rate"],
            n_epochs=1500,
            batch_size=64,
            hidden_activation="tanh",
            optimizer=cfg["optimizer"],
            momentum=0.9,
            regularization=cfg["regularization"],
            reg_lambda=cfg["reg_lambda"],
            shuffle=False,  # preserve sequence in batches
            verbose=False,
            random_state=42,
        )
        mlp.fit(X_train, y_train)
        val_metrics = mlp.evaluate(X_val, y_val)
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_model = mlp
            best_config = cfg

    # Refit best model on train+val
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])
    best_model = MultiLayerPerceptronFromScratch(
        input_dim=X_trainval.shape[1],
        hidden_layers=best_config["hidden_layers"],
        learning_rate=best_config["learning_rate"],
        n_epochs=2000,
        batch_size=64,
        hidden_activation="tanh",
        optimizer=best_config["optimizer"],
        momentum=0.9,
        regularization=best_config["regularization"],
        reg_lambda=best_config["reg_lambda"],
        shuffle=False,
        verbose=True,
        log_every=400,
        random_state=42,
    )
    best_model.fit(X_trainval, y_trainval)

    train_metrics = best_model.evaluate(X_trainval, y_trainval)
    test_metrics = best_model.evaluate(X_test, y_test)

    print(f"Data source             : {source}")
    print(f"Samples (train/val/test): {len(y_train)}/{len(y_val)}/{len(y_test)}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best config             : {best_config}")
    print("\nFinancial train metrics:")
    print(f"  Accuracy : {train_metrics['accuracy']:.4f}")
    print(f"  Precision: {train_metrics['precision']:.4f}")
    print(f"  Recall   : {train_metrics['recall']:.4f}")
    print(f"  F1-score : {train_metrics['f1']:.4f}")
    print("\nFinancial test metrics:")
    print(f"  Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall   : {test_metrics['recall']:.4f}")
    print(f"  F1-score : {test_metrics['f1']:.4f}")
    print(f"  Confusion matrix:\n{test_metrics['confusion_matrix']}")


if __name__ == "__main__":
    run_demo()
    run_experiment_grid()
    train_with_logic_gates()
    train_multilayer_on_logic_gates()
    train_mlp_on_financial_data("AAPL.US")
