# MNIST Neural Network from Scratch

This project contains a Jupyter Notebook that implements a simple two-layer neural network from scratch using NumPy. The goal is to classify handwritten digits from the classic MNIST dataset without relying on high-level machine learning frameworks like TensorFlow or PyTorch. 

## Project Overview

The notebook demonstrates the basic mathematics behind deep learning, including forward propagation, backpropagation, and gradient descent. By handling the matrix operations directly with NumPy, the code highlights how weights and biases are updated to minimize classification errors.

## Dependencies

To run the code in the notebook, you need to install the following Python libraries:

```bash
pip install numpy pandas matplotlib kaggle
```

The data is sourced from Kaggle's Digit Recognizer competition dataset.

## Dataset and Preprocessing

The notebook reads a standard CSV file where each row represents a handwritten digit image:
1. **Shuffling:** The data is randomized before splitting to avoid ordered bias.
2. **Splitting:** The first 1,000 samples are separated into a development (validation) set, while the remaining rows make up the training set.
3. **Normalization:** The raw pixel values (ranging from 0 to 255) are divided by 255 to scale them between 0 and 1, which ensures more stable gradient updates during training.

## Network Architecture

The model uses a fully connected feedforward architecture:
* **Input Layer:** 784 nodes, representing the flattened 28x28 pixel layout of each image.
* **Hidden Layer:** 10 nodes using the Rectified Linear Unit (ReLU) activation function.
* **Output Layer:** 10 nodes using the Softmax activation function to yield a probability distribution for digits 0 through 9.

## Core Implementation Steps

* `init_params()`: Randomly initializes weights and biases. It scales the weights to prevent the gradients from vanishing or exploding early on.
* `forward_prop()`: Computes linear combinations and applies activation functions layer by layer.
* `backprop()`: Works backward from the error at the output layer, using the chain rule to determine the gradients for all weights and biases.
* `update_params()`: Adjusts the model weights and biases by subtracting a fraction of the gradients determined by the learning rate.

## Performance and Evaluation

The model is set to train using standard gradient descent with a learning rate of 0.1. 
* Over 100 iterations, the training accuracy steadily increases from roughly 10 percent to approximately 79 percent.
* When evaluated against the separate development set, the network achieves a final classification accuracy of about 78.8 percent.

## Visualizing Predictions

The notebook includes a helper function called `test_predictions`. This function allows you to inspect specific examples by printing the network's prediction against the true label and plotting the grayscale image directly in the interface using Matplotlib.
