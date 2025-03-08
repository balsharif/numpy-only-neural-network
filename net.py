"""
Author: Bandar Alsharif
Date: 2025-03-07
License: MIT License

Description:
This script demonstrates how to implement and train a neural network from scratch using Numpy.
The model was trained to classify handwritten digits from the MNIST dataset, a classic problem in machine learning.
The implementation includes core components of neural networks, such as dense layers, activation functions,
loss functions, and backpropagation, providing a hands-on understanding of how neural networks work.

Key Features:
- Implementation of a fully connected (dense) layer.
- Implementation of cross-entropy loss for multi-class classification.
- Implementation of activation functions: softmax and sigmoid.
- Backpropagation algorithm using stochastic gradient descent (SGD) with mini-batches.
- Training, evaluation, and testing code included.

Dependencies:
- numpy
- matplotlib

Usage:
1. Install Dependencies:
    Ensure the required libraries are installed. You can install them using:
        pip install numpy matplotlib

2. Download the MNIST Dataset:
    - Download the MNIST dataset from a reliable source.
    - Update the dataset paths in the script to point to the location of the dataset on your device.

3. Run the Script:
    Execute the script using the following command:
        python fnn.py

4. Monitor Training:
    The script will train the neural network on the MNIST dataset and display:
        - Training progress (loss and accuracy per epoch).
        - Visualization of sample tests drown randomly from the test dataset.
"""
import numpy as np
import matplotlib.pyplot as plt

model_layers = []
layers_reversed = []
np.random.seed(1234)


def glorot_uniform(in_dim, out_dim):
    x = np.sqrt(6 / (in_dim + out_dim))
    return np.random.uniform(-x, x, (out_dim, in_dim))


def glorot_gaussian(in_dim, out_dim):
    x = np.sqrt(2 / (in_dim + out_dim))
    return np.random.normal(0, x, (out_dim, in_dim))


def softmax(input):
    out = np.exp(input - np.max(input, axis=-1, keepdims=True))
    return out / np.sum(out, axis=-1, keepdims=True)


def cross_entropy_loss_with_logits(y, t):
    y = softmax(y)
    y = y[np.arange(y.shape[0]), t.flatten()]
    y = np.clip(y, 1e-15, 1 - 1e-15)
    loss = -np.sum(np.log(y), axis=-1)
    return loss


def derivative_of_cross_entropy_loss_wrt_logits(y, t):
    y = softmax(y)
    dy = y.copy()
    dy[np.arange(y.shape[0]), t.flatten()] -= 1
    return dy


def normalize(x):
    return x / 255


class DenseLayer:
    def __init__(self, in_dim, out_dim, weight_init="glorot_uniform", activation=None, bias=True):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation
        if weight_init == "glorot_uniform":
            self.w = glorot_uniform(in_dim, out_dim)
        elif weight_init == "glorot_gaussian":
            self.w = glorot_gaussian(in_dim, out_dim)
        else:
            self.w = np.random.random(size=(out_dim, in_dim))

        self.b = np.zeros(shape=(1, out_dim)) if bias else None
        self.x = None
        self.y = None
        self.dw = None
        self.db = None

    def forward(self, x):
        self.x = x
        self.y = self.x @ self.w.T
        if self.b is not None:
            self.y += self.b

        if self.activation == "sigmoid":
            self.y = 1 / (1 + np.exp(-self.y))

        return self.y

    def backward(self, dy):
        if self.activation == "sigmoid":
            dy = self.y * (1 - self.y) * dy

        self.dw = dy.T @ self.x
        self.db = np.sum(dy, axis=0, keepdims=True)
        dx = dy @ self.w
        return dx

    def update(self, lr):
        self.w -= lr * self.dw
        if self.b is not None:
            self.b -= lr * self.db


def build():
    global model_layers, layers_reversed
    model_layers.append(DenseLayer(784, 32, activation="sigmoid"))
    model_layers.append(DenseLayer(32, 64, activation="sigmoid"))
    model_layers.append(DenseLayer(64, 10, activation=None))
    layers_reversed = model_layers.copy()
    layers_reversed.reverse()


def train():
    dataset = np.loadtxt("../data/MNIST/train.csv", delimiter=',', skiprows=1, dtype=int)
    train_ratio = 0.9
    train_data = dataset[:int(len(dataset) * train_ratio), :]
    eval_data = dataset[int(len(dataset) * train_ratio):, :]
    train_x = normalize(train_data[:, 1:])
    eval_x = normalize(eval_data[:, 1:])
    train_y = np.expand_dims(train_data[:, 0], axis=-1)
    eval_y = np.expand_dims(eval_data[:, 0], axis=-1)
    train_indices = np.arange(train_x.shape[0])

    num_epochs = 80
    batch_size = 8
    lr = 0.01
    train_losses = []
    test_losses = []
    accuracies = []

    def eval():
        test_loss = 0
        num_correct = 0
        for j in range(0, len(eval_x), batch_size):
            idx = np.arange(j, min(len(eval_x), j + batch_size))
            x = np.take(eval_x, idx, axis=0)
            t = np.take(eval_y, idx, axis=0)
            y = x
            for k in range(len(model_layers)):
                y = model_layers[k].forward(y)
            e = cross_entropy_loss_with_logits(y, t)
            labels = np.argmax(y, axis=-1)
            test_loss += e
            num_correct += len(np.where(labels == t.squeeze())[0])
        return test_loss / len(eval_x), num_correct

    for i in range(num_epochs):
        test_loss = 0
        num_correct = 0
        if i == 0:
            test_loss, num_correct = eval()
        else:
            np.random.shuffle(train_indices)

        train_loss = 0
        for j in range(0, len(train_x), batch_size):
            idx = np.arange(j, min(len(train_x), j + batch_size))
            x = np.take(train_x, train_indices[idx], axis=0)
            t = np.take(train_y, train_indices[idx], axis=0)

            if j == 60:
                lr /= 10
            y = x
            for k in range(len(model_layers)):
                y = model_layers[k].forward(y)

            e = cross_entropy_loss_with_logits(y, t)
            train_loss += e
            e /= batch_size
            dy = derivative_of_cross_entropy_loss_wrt_logits(y, t) / batch_size

            dx = dy
            for k in range(len(layers_reversed)):
                dx = layers_reversed[k].backward(dx)

            for k in range(len(model_layers)):
                model_layers[k].update(lr)

        train_loss /= len(train_x)
        print("\nepoch", i + 1, "---- train_loss =", train_loss, end="")
        train_losses.append(train_loss)

        if i > 0:
            test_loss, num_correct = eval()

        accuracy = (num_correct / len(eval_x)) * 100
        print(" ---- test_loss =", test_loss, "----- accuracy =", "{:.3f}%".format(accuracy), end="")
        test_losses.append(test_loss)
        accuracies.append(accuracy)

    epochs = range(num_epochs)
    fig, a1 = plt.subplots(figsize=(8, 6))
    a1.set_xlabel('Epochs')
    a1.set_ylabel('Loss')
    a1.plot(epochs, train_losses, label='Training Loss', color='blue')
    a1.plot(epochs, test_losses, label='Testing Loss', color='red')
    a1.tick_params(axis='y')
    a2 = a1.twinx()
    a2.set_ylabel('Accuracy')
    a2.plot(epochs, accuracies, label='Accuracy', color='green')
    a2.tick_params(axis='y')
    a1.legend(loc='upper left')
    a2.legend(loc='upper right')
    plt.show()

def test():
    num_samples = 10
    test_data = np.loadtxt("../data/MNIST/test.csv", delimiter=',', skiprows=1, dtype=int)
    test_idx = np.random.choice(test_data.shape[0], size=num_samples, replace=False)
    test_data = normalize(test_data[test_idx])
    labels = np.zeros(shape=(num_samples, 1))

    for i in range(len(test_data)):
        x = test_data[i][None, :]
        y = x
        for j in range(len(model_layers)):
            y = model_layers[j].forward(y)
        labels[i, 0] = np.argmax(y, axis=-1).item()

    fig, axes = plt.subplots(2, int(num_samples/2), figsize=(10, 5))
    axes = axes.ravel()
    for i, (img, label) in enumerate(zip(test_data, labels)):
        img = np.reshape(img, newshape=(28, 28))
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(label[0], fontsize=20)
        axes[i].axis('off')

    plt.subplots_adjust(hspace=0.25)
    plt.show()


if __name__ == "__main__":
    build()
    train()
    test()
