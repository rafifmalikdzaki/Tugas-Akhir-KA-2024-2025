import numpy as np
from numpy.linalg import pinv
from sklearn.preprocessing import OneHotEncoder

class FELM_AE:
    def __init__(self, n_hidden, fisher_lambda, activation='relu'):
        self.n_hidden = n_hidden
        self.fisher_lambda = fisher_lambda
        self.W = None
        self.b = None
        self.beta = None
        self.activation = activation

    def _activation(self, x):
        match(self.activation):
            case "relu":
                return np.maximum(0, x)
            case "sigmoid":
                return 1 / (1 + np.exp(-x))
            case "tanh":
                return np.tanh(x)
            case _:
                raise ValueError('Unsupported Activation Function')

    def _calculate_fisher_matrix(self, H, y):
        unique_class = np.unique(y)
        n_samples, n_hidden = H.shape

        h_mean_global = np.mean(H, axis=0)

        Sb = np.zeros((n_hidden, n_hidden))
        Sw = np.zeros((n_hidden, n_hidden))

        for cls in unique_class:
            class_idx = np.where(y == cls)[0]
            H_class = H[class_idx]
            n_class = len(class_idx)
            h_mean_class = np.mean(H_class, axis=0)

            # Between-class scatter
            cg_mean_diff = h_mean_class - h_mean_global
            Sb += n_class * np.outer(cg_mean_diff, cg_mean_diff)

            for h_i in H_class:
                hih_mean_diff = h_i - h_mean_class
                Sw += np.outer(hih_mean_diff, hih_mean_diff)


        epsilon = 1e-12
        D = np.diag(np.diag(Sb)) + epsilon * np.eye(Sb.shape[0])

        D_inv = np.linalg.inv(D)
        S = D_inv @ Sw
        return S

    def fit(self, X, y):

        n_samples, n_features = X.shape

        if y.ndim == 1:
            y = np.eye(len(np.unique(y)))[0]

        self.W = np.random.uniform(-1, 1, (n_features, self.n_hidden))
        self.b = np.random.uniform(-1, 1, self.n_hidden)

        # Compute hidden layer output H
        H = self._activation(X @ self.W + self.b)

        # Calculate Fisher regularization matrix S
        S = self._calculate_fisher_matrix(H, np.argmax(y, axis=1))

        if n_samples >= self.n_hidden:
            # Case N >= L
            self.beta = pinv(self.fisher_lambda * S + H.T @ H) @ H.T @ y
        else:
            # Case N < L
            self.beta = H.T @ pinv(self.fisher_lambda * S + H @ H.T) @ y

    def transform(self, X):
        H = self._activation(X @ self.W + self.b)
        return H @ self.beta, H

class ML_FELM:
    def __init__(self, layer_sizes, fisher_lambdas, activation="relu"):
        """
        Initialize the ML-FELM.

        Parameters:
        - layer_sizes: List of hidden nodes for each layer.
        - fisher_lambdas: List of Fisher regularization parameters for each layer.
        - activation: Activation function for hidden layers.
        """
        self.final_weights = None
        self.classes = None
        self.layer_sizes = layer_sizes
        self.fisher_lambdas = fisher_lambdas
        self.activation = activation
        self.layers = []
        self.one_hot_encoder = OneHotEncoder(sparse_output=False)

    def fit(self, X, y):
        """
        Train the ML-FELM.

        Parameters:
        - X: Input data matrix (N x d).
        - y: Target labels (N,).
        """
        # One-hot encode labels
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        y_onehot = np.zeros((n_samples, len(self.classes)))
        if len(np.unique(y)) > 2:  # Multi-class classification
            for idx, cls in enumerate(self.classes):
                y_onehot[y == cls, idx] = 1

        y = y_onehot

        H_new = X
        for i, (n_hidden, fisher_lambda) in enumerate(zip(self.layer_sizes, self.fisher_lambdas)):
            felm_ae = FELM_AE(n_hidden, activation=self.activation, fisher_lambda=fisher_lambda)
            felm_ae.fit(H_new, y)
            H_new, abstract_features = felm_ae.transform(H_new)
            self.layers.append(felm_ae)

        # Final classification weighggts
        self.final_weights = pinv(H_new) @ y

    def predict(self, X):
        """
        Predict class labels using the ML-FELM.

        Parameters:
        - X: Input data matrix (N x d).

        Returns:
        - Predicted class labels (N,).
        """
        H_new = X
        for felm_ae in self.layers:
            H_new = felm_ae._activation(H_new @ felm_ae.W + felm_ae.b) @ felm_ae.beta

        output = H_new @ self.final_weights

        if len(output.shape) > 1:  # Multi-class classification
            predictions = np.argmax(output, axis=1)
            return np.array([self.classes[pred] for pred in predictions])
        else:  # Binary classification
            return (output > 0.5).astype(int)

class ELMClassifier:
    def __init__(self, n_hidden, activation="sigmoid"):
        """
        Initialize the ELM Classifier.
        
        Parameters:
        - n_hidden: Number of hidden nodes.
        - activation: Activation function for the hidden layer.
                      Options: "sigmoid", "tanh", "relu".
        """
        self.classes = None
        self.n_hidden = n_hidden
        self.activation = activation
        self.W = None  # Input weights
        self.b = None  # Bias
        self.beta = None  # Output weights

    def _activation(self, X):
        """
        Compute the activation function for the hidden layer.

        Parameters:
        - X: Input matrix to the activation function.

        Returns:
        - Transformed matrix after applying the activation function.
        """
        if self.activation == "sigmoid":
            return 1 / (1 + np.exp(-X))
        elif self.activation == "tanh":
            return np.tanh(X)
        elif self.activation == "relu":
            return np.maximum(0, X)
        else:
            raise ValueError("Unsupported activation function: choose 'sigmoid', 'tanh', or 'relu'.")

    def fit(self, X, y):
        """
        Train the ELM Classifier.

        Parameters:
        - X: Training data of shape (n_samples, n_features).
        - y: Target labels of shape (n_samples,).
        """
        n_samples, n_features = X.shape

        # Randomly initialize weights and biases
        self.W = np.random.uniform(-1, 1, (n_features, self.n_hidden))  # Input weights
        self.b = np.random.uniform(-1, 1, self.n_hidden)               # Bias

        # Compute the hidden layer output (H)
        H = self._activation(X @ self.W + self.b)

        # Convert y to one-hot encoding if multi-class classification
        self.classes = np.unique(y)
        y_onehot = np.zeros((n_samples, len(self.classes)))
        if len(np.unique(y)) > 2:  # Multi-class classification
            for idx, cls in enumerate(self.classes):
                y_onehot[y == cls, idx] = 1
                
        y = y_onehot

        # Compute the output weights (beta) using least-squares
        self.beta = np.linalg.pinv(H) @ y  # Moore-Penrose pseudo-inverse

    def predict(self, X):
        """
        Predict class labels for the input data.

        Parameters:
        - X: Input data of shape (n_samples, n_features).

        Returns:
        - Predicted class labels of shape (n_samples,).
        """
        # Compute the hidden layer output (H)
        H = self._activation(X @ self.W + self.b)

        # Compute the network output
        output = H @ self.beta

        # Convert to class labels
        if len(output.shape) > 1:  # Multi-class classification
            predictions = np.argmax(output, axis=1)
            return np.array([self.classes[pred] for pred in predictions])
        else:  # Binary classification
            return (output > 0.5).astype(int)


if __name__ == "__main__":
    from sklearn.datasets import make_classification, load_iris, load_wine
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler

    iris = load_iris()
    X = iris.data
    y = iris.target

    wine = load_wine()
    X = wine.data
    y = wine.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("ELM")
    elm = ELMClassifier(n_hidden=100, activation="relu")
    elm.fit(X_train, y_train)

    y_pred = elm.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")


    X = StandardScaler().fit_transform(X)

    print("Data shape:", X.shape)
    print("Labels shape:", y.shape)


    print("ELM-Autoencoder")
    fisher_lambda = 0.1
    n_hidden = 50
    felm_ae = FELM_AE(n_hidden=n_hidden, fisher_lambda=fisher_lambda)
    felm_ae.fit(X, y)

    abstract_features = felm_ae.transform(X)[0]
    print("Abstract feature shape:", abstract_features.shape)

    print("ML-FELM")
        # Define ML-FELM parameters
    layer_sizes = [30, 20, 10]  # Number of hidden nodes in each layer
    fisher_lambdas = [0.1, 0.1, 0.1]  # Fisher regularization for each layer

# Train ML-FELM
    ml_felm = ML_FELM(layer_sizes, fisher_lambdas, activation="relu")
    ml_felm.fit(X_train, y_train)

    y_pred = ml_felm.predict(X_test)

# Evaluate accuracy
    print(y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.2f}")

