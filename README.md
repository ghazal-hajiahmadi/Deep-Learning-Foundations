# Deep Learning Foundations: Autoencoder & MLP

This repository contains my implementations of foundational deep learning models developed as part of my coursework. The project is divided into two main parts: building an Autoencoder completely from scratch using NumPy, and developing a Multi-Layer Perceptron (MLP) for a regression task.

---

## 1. Autoencoder from Scratch (NumPy)
In this section, a fully connected Autoencoder is implemented without using any high-level deep learning frameworks (such as PyTorch or TensorFlow) to better understand the underlying mathematics of neural networks. 

* **Architecture:** 4-layer Autoencoder (`784 -> 128 -> Latent_Dim -> 128 -> 784`).
* **Implementation Details:**
  * Manually coded the **Forward Pass**, **Backpropagation**, and weight update steps.
  * Used **He Initialization** for network weights and implemented **ReLU** and **Sigmoid** activation functions along with their analytical derivatives.
* **Evaluation:** Trained on the MNIST dataset to compress images into a low-dimensional latent space (4 and 8 neurons). Extracted the latent features and trained a separate Feedforward classifier on top to evaluate the quality of the feature reduction.

## 2. Life Expectancy Prediction (MLP)
A regression task to predict life expectancy based on the World Health Organization (WHO) dataset using `scikit-learn`.

* **Data Preprocessing:**
  * Handled missing values using median imputation.
  * Managed outliers via Interquartile Range (IQR) clipping.
  * Applied Z-score standardization properly after splitting to prevent **data leakage**.
* **Modeling:** Trained and compared three different MLP architectures, ranging from a simple single hidden layer to deeper networks.
* **Evaluation:** Evaluated models using **MSE**, **RMSE**, and **R²-score**, and plotted learning curves to analyze convergence and monitor overfitting.

---

## Repository Structure
```text
├── Autoencoder_Scratch/
│   ├── autoencoder.py          # Scratch implementation of Autoencoder & Classifier
│   └── mnist_eval.ipynb        # Training and evaluation on MNIST
├── Life_Expectancy_MLP/
│   ├── Life Expectancy Data.csv
│   └── mlp_regression.ipynb    # Preprocessing and MLP training pipeline
└── README.md
