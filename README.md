# Handwritten Character Detection using Neural Network ✍️🧠

This project demonstrates the implementation of a basic two-layer neural network trained on the MNIST digit recognition dataset. It serves as an educational example, providing insights into the fundamental mathematics behind neural networks.

## 🎯 Features

- Fully connected neural network with:
  - **Input Layer**: 784 neurons (28x28 pixel images)
  - **Hidden Layer**: 10 neurons with ReLU activation
  - **Output Layer**: 10 neurons (one for each digit) with softmax activation
- Forward and backward propagation from scratch
- Gradient descent optimization
- Accuracy evaluation on the MNIST dataset

## 🛠️ Technology Stack

- **Deep Learning Framework**: Custom implementation using NumPy
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib

## 📋 Prerequisites

- Python 3.7+
- Required Python packages (see Installation)
- Kaggle API access (for dataset download)

## ⚙️ Installation & Setup

1. Clone the repository:
```
git clone https://github.com/susanthb/handwritten-neuralnet.git

cd handwritten-neuralnet
```

2. Install required packages:
```
pip install numpy pandas matplotlib
```

3. Download the dataset:
- Go to [Kaggle - Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer/data)
- Download `train.csv` and place it in the project directory

## 🚀 Usage

1. Run the training script:
```
python neuralnet.py
```

2. The system will:
- Load and preprocess the dataset
- Train the neural network from scratch
- Display training progress every 10 iterations
- Print accuracy on the training set

## 💡 How It Works

### Neural Network Architecture

Our neural network follows a simple two-layer architecture:

- **Input Layer**: \( A^{[0]} \) with 784 units (28x28 pixel images)
- **Hidden Layer**: \( A^{[1]} \) with 10 units, using ReLU activation
- **Output Layer**: \( A^{[2]} \) with 10 units, corresponding to digit classes (0-9), using softmax activation

**Forward propagation**

$$Z^{[1]} = W^{[1]} X + b^{[1]}$$
$$A^{[1]} = g_{\text{ReLU}}(Z^{[1]}))$$
$$Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}$$
$$A^{[2]} = g_{\text{softmax}}(Z^{[2]})$$

**Backward propagation**

$$dZ^{[2]} = A^{[2]} - Y$$
$$dW^{[2]} = \frac{1}{m} dZ^{[2]} A^{[1]T}$$
$$dB^{[2]} = \frac{1}{m} \Sigma {dZ^{[2]}}$$
$$dZ^{[1]} = W^{[2]T} dZ^{[2]} .* g^{[1]\prime} (z^{[1]})$$
$$dW^{[1]} = \frac{1}{m} dZ^{[1]} A^{[0]T}$$
$$dB^{[1]} = \frac{1}{m} \Sigma {dZ^{[1]}}$$

**Parameter updates**

$$W^{[2]} := W^{[2]} - \alpha dW^{[2]}$$
$$b^{[2]} := b^{[2]} - \alpha db^{[2]}$$
$$W^{[1]} := W^{[1]} - \alpha dW^{[1]}$$
$$b^{[1]} := b^{[1]} - \alpha db^{[1]}$$

**Vars and shapes**

Forward prop

- $A^{[0]} = X$: 784 x m
- $Z^{[1]} \sim A^{[1]}$: 10 x m
- $W^{[1]}$: 10 x 784 (as $W^{[1]} A^{[0]} \sim Z^{[1]}$)
- $B^{[1]}$: 10 x 1
- $Z^{[2]} \sim A^{[2]}$: 10 x m
- $W^{[1]}$: 10 x 10 (as $W^{[2]} A^{[1]} \sim Z^{[2]}$)
- $B^{[2]}$: 10 x 1

Backprop

- $dZ^{[2]}$: 10 x m ($~A^{[2]}$)
- $dW^{[2]}$: 10 x 10
- $dB^{[2]}$: 10 x 1
- $dZ^{[1]}$: 10 x m ($~A^{[1]}$)
- $dW^{[1]}$: 10 x 10
- $dB^{[1]}$: 10 x 1



## 📊 Performance Metrics

- Achieves ~90% accuracy on the training set
- Loss reduction with each iteration
- Adjustable learning rate and iterations for optimization

## 🌐 Future Improvements

- Implement batch normalization for stability
- Add support for GPU acceleration using TensorFlow/PyTorch
- Expand model to support deeper architectures

---
Developed with ❤️ for understanding neural networks from scratch!
