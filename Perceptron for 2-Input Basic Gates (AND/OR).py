{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNtfxi929A54UEOia0U6yNU",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/sandip988/ANN_021_348/blob/main/Perceptron%20for%202-Input%20Basic%20Gates%20(AND/OR).py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 3,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "V8FJ152D945i",
        "outputId": "fbeaf84d-6ead-4e42-8423-5d38a8c27d59"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Training Perceptron for AND gate:\n",
            "\n",
            "Initial weights: [0.91786166 0.25262956], Initial bias: 0.7096\n",
            "Epoch 1: weights = [0.81786166 0.15262956], bias = 0.4096, errors = 3\n",
            "Epoch 2: weights = [0.71786166 0.05262956], bias = 0.1096, errors = 3\n",
            "Epoch 3: weights = [ 0.61786166 -0.04737044], bias = -0.1904, errors = 3\n",
            "Epoch 4: weights = [ 0.51786166 -0.04737044], bias = -0.2904, errors = 1\n",
            "Epoch 5: weights = [0.51786166 0.05262956], bias = -0.2904, errors = 2\n",
            "Epoch 6: weights = [0.41786166 0.05262956], bias = -0.3904, errors = 1\n",
            "Epoch 7: weights = [0.41786166 0.15262956], bias = -0.3904, errors = 2\n",
            "Epoch 8: weights = [0.41786166 0.25262956], bias = -0.3904, errors = 2\n",
            "Epoch 9: weights = [0.31786166 0.25262956], bias = -0.4904, errors = 1\n",
            "Epoch 10: weights = [0.31786166 0.25262956], bias = -0.4904, errors = 0\n",
            "Converged after 10 epochs\n",
            "\n",
            "Final AND gate weights: [0.31786166 0.25262956]\n",
            "Final AND gate bias: -0.4904\n",
            "AND gate accuracy: 100.00%\n",
            "\n",
            "Training Perceptron for OR gate:\n",
            "\n",
            "Initial weights: [0.61147467 0.67233421], Initial bias: 0.3975\n",
            "Epoch 1: weights = [0.61147467 0.67233421], bias = 0.2975, errors = 1\n",
            "Epoch 2: weights = [0.61147467 0.67233421], bias = 0.1975, errors = 1\n",
            "Epoch 3: weights = [0.61147467 0.67233421], bias = 0.0975, errors = 1\n",
            "Epoch 4: weights = [0.61147467 0.67233421], bias = -0.0025, errors = 1\n",
            "Epoch 5: weights = [0.61147467 0.67233421], bias = -0.0025, errors = 0\n",
            "Converged after 5 epochs\n",
            "\n",
            "Final OR gate weights: [0.61147467 0.67233421]\n",
            "Final OR gate bias: -0.0025\n",
            "OR gate accuracy: 100.00%\n"
          ]
        }
      ],
      "source": [
        "import numpy as np\n",
        "\n",
        "# Step activation function\n",
        "def step_function(x):\n",
        "    return 1 if x >= 0 else 0\n",
        "\n",
        "# Perceptron class\n",
        "class Perceptron:\n",
        "    def __init__(self, input_size, learning_rate=0.1):\n",
        "        self.weights = np.random.rand(input_size)  # Random initial weights\n",
        "        self.bias = np.random.rand()  # Random initial bias\n",
        "        self.learning_rate = learning_rate\n",
        "\n",
        "    def predict(self, X):\n",
        "        return step_function(np.dot(X, self.weights) + self.bias)\n",
        "\n",
        "    def train(self, X, y, max_epochs=100):\n",
        "        print(f\"\\nInitial weights: {self.weights}, Initial bias: {self.bias:.4f}\")\n",
        "        for epoch in range(max_epochs):\n",
        "            errors = 0\n",
        "            for i in range(len(X)):\n",
        "                prediction = self.predict(X[i])\n",
        "                error = y[i] - prediction\n",
        "                if error != 0:\n",
        "                    errors += 1\n",
        "                    # Update weights and bias\n",
        "                    self.weights += self.learning_rate * error * X[i]\n",
        "                    self.bias += self.learning_rate * error\n",
        "            # Print weights and bias after each epoch\n",
        "            print(f\"Epoch {epoch+1}: weights = {self.weights}, bias = {self.bias:.4f}, errors = {errors}\")\n",
        "            # Check for convergence (no errors)\n",
        "            if errors == 0:\n",
        "                print(f\"Converged after {epoch+1} epochs\")\n",
        "                break\n",
        "        return errors == 0\n",
        "\n",
        "    def evaluate(self, X, y):\n",
        "        correct = 0\n",
        "        for i in range(len(X)):\n",
        "            prediction = self.predict(X[i])\n",
        "            if prediction == y[i]:\n",
        "                correct += 1\n",
        "        return correct / len(X)\n",
        "\n",
        "# Input data (truth tables)\n",
        "X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Inputs for both gates\n",
        "y_and = np.array([0, 0, 0, 1])  # AND gate outputs\n",
        "y_or = np.array([0, 1, 1, 1])   # OR gate outputs\n",
        "\n",
        "# Train Perceptron for AND gate\n",
        "print(\"Training Perceptron for AND gate:\")\n",
        "and_perceptron = Perceptron(input_size=2, learning_rate=0.1)\n",
        "and_converged = and_perceptron.train(X, y_and, max_epochs=100)\n",
        "and_accuracy = and_perceptron.evaluate(X, y_and)\n",
        "print(f\"\\nFinal AND gate weights: {and_perceptron.weights}\")\n",
        "print(f\"Final AND gate bias: {and_perceptron.bias:.4f}\")\n",
        "print(f\"AND gate accuracy: {and_accuracy * 100:.2f}%\")\n",
        "\n",
        "# Train Perceptron for OR gate\n",
        "print(\"\\nTraining Perceptron for OR gate:\")\n",
        "or_perceptron = Perceptron(input_size=2, learning_rate=0.1)\n",
        "or_converged = or_perceptron.train(X, y_or, max_epochs=100)\n",
        "or_accuracy = or_perceptron.evaluate(X, y_or)\n",
        "print(f\"\\nFinal OR gate weights: {or_perceptron.weights}\")\n",
        "print(f\"Final OR gate bias: {or_perceptron.bias:.4f}\")\n",
        "print(f\"OR gate accuracy: {or_accuracy * 100:.2f}%\")"
      ]
    }
  ]
}