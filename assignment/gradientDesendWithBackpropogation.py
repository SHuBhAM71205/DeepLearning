import numpy as np

# ------------------- Global Constants -------------------
MAX_ITR = 100000
NO_OF_LAYERS = 3        # 2 hidden + 1 output
NO_OF_OUTPUT = 4
NO_OF_INPUT = 4
OUTPUT_FUNCTION = "SOFTMAX"
LEARNING_RATE = 0.01

# ------------------- Initialize Parameters -------------------
weights = [np.random.randn(NO_OF_INPUT, NO_OF_INPUT) * 0.01 for _ in range(NO_OF_LAYERS - 1)]
biases = [np.zeros((NO_OF_INPUT, 1)) for _ in range(NO_OF_LAYERS - 1)]

# Output layer weight & bias
weights.append(np.random.randn(NO_OF_OUTPUT, NO_OF_INPUT) * 0.01)
biases.append(np.zeros((NO_OF_OUTPUT, 1)))

# ------------------- Helper Functions -------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # stability trick
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    # y_true: one-hot vector, y_pred: predicted softmax
    return -np.sum(y_true * np.log(y_pred + 1e-9))

# ------------------- Forward Propagation -------------------
def forward_propagation(X):
    activations = [X]
    pre_activations = []

    for i in range(NO_OF_LAYERS):
        W = weights[i]
        b = biases[i]
        Z = np.dot(W, activations[-1]) + b  # pre-activation
        pre_activations.append(Z)

        if i == NO_OF_LAYERS - 1:
            A = softmax(Z)  # output layer
        else:
            A = sigmoid(Z)  # hidden layers

        activations.append(A)

    return activations, pre_activations

# ------------------- Backpropagation -------------------
def back_propagation(activations, pre_activations, Y):
    m = Y.shape[1]  # number of examples
    grads_w = [None] * NO_OF_LAYERS
    grads_b = [None] * NO_OF_LAYERS

    # Output layer gradient
    dZ = activations[-1] - Y  # derivative of loss wrt Z for softmax+crossentropy
    for i in reversed(range(NO_OF_LAYERS)):
        A_prev = activations[i]
        grads_w[i] = (1/m) * np.dot(dZ, A_prev.T)
        grads_b[i] = (1/m) * np.sum(dZ, axis=1, keepdims=True)

        if i > 0:  # propagate to previous layer
            W = weights[i]
            dA_prev = np.dot(W.T, dZ)
            dZ = dA_prev * sigmoid_derivative(pre_activations[i-1])

    return grads_w, grads_b

# ------------------- Update Parameters -------------------
def update_parameters(grads_w, grads_b):
    for i in range(NO_OF_LAYERS):
        weights[i] -= LEARNING_RATE * grads_w[i]
        biases[i] -= LEARNING_RATE * grads_b[i]

# ------------------- Training -------------------
INPUT = np.array([[0.5], [0.2], [0.1], [0.7]])  # shape (4,1)
Y = np.array([[1], [0], [0], [0]])  # one-hot target (class 0)

for count in range(MAX_ITR):
    # Forward pass
    activations, pre_activations = forward_propagation(INPUT)

    # Compute loss
    loss = cross_entropy_loss(Y, activations[-1])

    # Backward pass
    grads_w, grads_b = back_propagation(activations, pre_activations, Y)

    # Update weights
    update_parameters(grads_w, grads_b)

    if count % 100 == 0:
        print(f"Iteration {count}, Loss: {loss}")
