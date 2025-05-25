# ---------------------- Code 1 ----------------------
import numpy as np

class SimpleEmotionNet:
    def __init__(self, input_size=2304, hidden_size=64, output_size=7, activation="sigmoid", seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.eta = 0.01  # learning rate

        # Initialisation des poids et biais
        self.w1 = np.random.randn(hidden_size, input_size) * 0.01
        self.b1 = np.zeros((hidden_size, 1))
        self.w2 = np.random.randn(output_size, hidden_size) * 0.01
        self.b2 = np.zeros((output_size, 1))

        # Fonction d’activation
        self.set_activation(activation)

    def set_activation(self, name):
        if name == "sigmoid":
            self.f = lambda z: 1 / (1 + np.exp(-z))
            self.df = lambda z: self.f(z) * (1 - self.f(z))
        elif name == "tanh":
            self.f = lambda z: np.tanh(z)
            self.df = lambda z: 1 - np.tanh(z)**2
        elif name == "arctan":
            self.f = lambda z: np.arctan(z) / np.pi + 0.5
            self.df = lambda z: 1 / (np.pi * (1 + z**2))

    def forward(self, x):
        self.z1 = np.dot(self.w1, x) + self.b1
        self.a1 = self.f(self.z1)
        self.z2 = np.dot(self.w2, self.a1) + self.b2
        self.a2 = self.f(self.z2)
        return self.a2

    def backward(self, x, y):
        a2 = self.forward(x)
        delta2 = (a2 - y) * self.df(self.z2)
        delta1 = np.dot(self.w2.T, delta2) * self.df(self.z1)

        dw2 = np.dot(delta2, self.a1.T)
        db2 = delta2
        dw1 = np.dot(delta1, x.T)
        db1 = delta1

        # Mise à jour (équation du gradient)
        self.w2 -= self.eta * dw2
        self.b2 -= self.eta * db2
        self.w1 -= self.eta * dw1
        self.b1 -= self.eta * db1

        # Retourner le coût
        return 0.5 * np.linalg.norm(a2 - y)**2

    def predict(self, x):
        return np.argmax(self.forward(x))

# ---------------------- Code 2 ----------------------
import matplotlib.pyplot as plt
import cv2
import os
import random

IMG_SIZE = 48
INPUT_SIZE = IMG_SIZE * IMG_SIZE
OUTPUT_SIZE = 7

# Chargement de 100 images par classe
def load_dataset(path, max_per_class=100):
    data = []
    labels = sorted(os.listdir(path))
    for i, label in enumerate(labels):
        folder = os.path.join(path, label)
        for file in os.listdir(folder)[:max_per_class]:
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            x = img.flatten().reshape(INPUT_SIZE, 1) / 255.0
            y = np.zeros((OUTPUT_SIZE, 1))
            y[i] = 1
            data.append((x, y))
    return data

# Entraînement sur un réseau donné
def train_model(model, data, epochs=10):
    costs = []
    for epoch in range(epochs):
        total_cost = 0
        random.shuffle(data)
        for x, y in data:
            total_cost += model.backward(x, y)
        costs.append(total_cost)
        print(f"Époque {epoch+1} - Coût total : {total_cost:.4f}")
    return costs

# Test de plusieurs activations
def test_activations(train_data):
    activations = ["sigmoid", "tanh", "arctan"]
    for act in activations:
        print(f"\nTest activation : {act}")
        model = SimpleEmotionNet(activation=act)
        costs = train_model(model, train_data, epochs=10)
        plt.plot(costs, label=act)
    plt.title("Évolution du coût selon la fonction d’activation")
    plt.xlabel("Époques")
    plt.ylabel("Coût total")
    plt.legend()
    plt.show()
# ---------------------- Code 3 ----------------------
def evaluate_model(model, test_data):
    correct = 0
    for x, y in test_data:
        if model.predict(x) == np.argmax(y):
            correct += 1
    accuracy = correct / len(test_data)
    print(f"✅ Précision finale : {accuracy * 100:.2f}%")
    return accuracy
