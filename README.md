# Deep Learning Basics: Autoencoder & MLP

This repository contains my implementations of foundational deep learning models. The project is divided into two main parts: building an Autoencoder completely from scratch using NumPy, and developing a Multi-Layer Perceptron (MLP) for a regression task.

## 1. Autoencoder from Scratch (NumPy)
In this section, a fully connected Autoencoder is implemented without using any high-level deep learning frameworks (like PyTorch or TensorFlow). 

* **Architecture:** Developed a 4-layer Autoencoder (784 -> 128 -> Latent -> 128 -> 784).
* **Implementation Details:** * Manually coded the **Forward Pass**, **Backpropagation**, and weight updates.
  * Used **He Initialization** for weights and implemented ReLU and Sigmoid activation functions along with their derivatives.
* **Evaluation:** Trained on the MNIST dataset to compress images into a low-dimensional latent space (4 and 8 neurons). Extracted the latent features and trained a separate Feedforward classifier on top to evaluate the quality of the feature extraction.

## 2. Life Expectancy Prediction (MLP)
A regression task to predict life expectancy based on the WHO dataset using `scikit-learn`.

* **Data Preprocessing:** Handled missing values using median imputation, managed outliers via IQR clipping, and applied Z-score standardization to prevent data leakage.
* **Modeling:** Trained and compared three different MLP architectures (ranging from a single hidden layer to deeper networks).
* **Evaluation:** Models were evaluated based on MSE, RMSE, and R2-score, with learning curves plotted to analyze convergence and prevent overfitting.

## Tech Stack
* Python
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
