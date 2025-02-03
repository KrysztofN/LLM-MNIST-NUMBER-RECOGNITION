# Neural Network from Scratch

This repository documents my 4-day journey of creating a neural network from scratch without any prior knowledge. The documentation represents my thought process and maintains a record of each step, including my thoughts and recommendations for potential improvements.

## Project Description

The project focuses on developing a neural network trained and tested on the MNIST dataset for recognizing handwritten numbers. The dataset can be accessed and downloaded from:

[MNIST Dataset on Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

## Part One - Dataset Preparation

### 1. Dataset Setup

Starting with Kaggle's MNIST scripts, I verified the dataset integrity. The dataset is split into:
- 60,000 training images
- 10,000 test images

This represents a standard ratio to prevent overfitting. Each 28x28 grayscale image is stored as a 3D array (samples × width × height), though most fully connected layers expect 2D input (samples × flattened_features).
<br><br>
<p align="center">
  <img src="https://github.com/KrysztofN/Neural_Network_From_Scratch/blob/main/img/num.png" alt="Num" width="250" height="250">
</p>
<p align="center">
<i>Train image representing number 2</i>
</p>

### 2. Normalization

Pixel values in the dataset are 8-bit integers [0, 255]. To stabilize gradient calculations during backpropagation, I normalized them to [0, 1] by dividing by 255. This normalization prevents large input values from causing erratic weight updates, which could:
- Slow down training
- Destabilize the loss landscape

### 3. Flattening

The 3D arrays (60,000×28×28) were reshaped into 2D (60,000×784). Each 28x28 image becomes a 784-pixel vector—essentially 'unfolding' the grid into a single row. While this transformation loses spatial locality, it's necessary for initial dense layer compatibility in our neural network.

### 4. Label Encoding

The labels (y_train/y_test) are integers 0-9, but neural networks output probabilities via softmax. To align labels with this probabilistic framework, I applied one-hot encoding. For example:
- Label '5' becomes [0,0,0,0,0,1,0,0,0,0]
- This creates a 10-dimensional vector where the 6th index (zero-based) is 1

After this step:
- y_test becomes (60000, 10) vector
- y_train becomes (10000, 10) vector

This normalization ensures easy comparison between the neural network's output (10 values in a vector) and the hot encoded y label vector.
<br><br><br>
## Part Two - Neural Network

### 1. Neural Network Design

The implementation uses a simple yet effective design comprising:
- Input layer
- One hidden layer (128 neurons)
- Output layer

<p align="center">
  <img src="https://github.com/KrysztofN/Neural_Network_From_Scratch/blob/main/img/network.png" alt="Num" width="400" height="350">
</p>
<p align="center">
<i>Neural network design</i>
</p>

### 2. Weights and Bias Initialization

Neural networks learn through adjustable parameters called weights and biases. These parameters are fundamental to "AI learning"—the process of calculating and fine-tuning these values to achieve desired outputs.

**Weights** represent connection strengths between neurons, determining how much influence one neuron's output has on the next neuron's input. Think of weights as determining the "importance" of each connection in the network.

A **bias** is an additional parameter added to each neuron that enables more effective learning by shifting the neuron's activation function horizontally, allowing the network to represent patterns that don't necessarily pass through the origin.

Common initialization methods include:
- *He initialization* (Best for ReLU)
- *Xavier initialization*

This implementation uses He initialization, though others were tested for performance comparison. All biases are initialized to zero, which is a common and effective practice.

The He initialization formula:

<p align="center">
  <img src="https://github.com/KrysztofN/Neural_Network_From_Scratch/blob/main/img/heinit.png" alt="Num" width="250" height="150">
</p>
<p align="center">
<i>He initialization formula</i>
</p>

We approximate weights matrix with gaussian distribution of:
- Mean value: 0
- Variance: **2/n^l** where **n^l** is number of neurons in layer l

For our network with:
- Input layer: 784 neurons (28×28 input image)
- Hidden layer: 128 neurons
- Output layer: 10 neurons (for digit classification)

The initialization requires:
```python
# Weight initialization using He initialization
W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2/input_size)
W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2/hidden_size)

# Bias initialization
b1 = np.zeros((1, hidden_size))
b2 = np.zeros((1, output_size))
```

### 3. Activation Functions

Activation functions are mathematical operations applied to a neuron's output, serving two critical purposes:
1. Introducing non-linearity into the model
2. Enabling the network to learn and represent complex patterns in data

This implementation focuses on two specific functions:

#### ReLU
- Commonly used in modern networks
- Best for hidden layers
- Works by 'zeroing' negative values

<p align="center">
  <img src="https://github.com/KrysztofN/Neural_Network_From_Scratch/blob/main/img/relu.png" alt="Num" width="400" height="150">
</p>
<p align="center">
<i>ReLU function</i>
</p>



#### Softmax
- Converts a vector of n real numbers into a probability distribution
- Commonly used as the last activation function
- Normalizes network output to a probability distribution over predicted output classes

<p align="center">
  <img src="https://github.com/KrysztofN/Neural_Network_From_Scratch/blob/main/img/softmax.png" alt="Num" width="200" height="150">
  <img src="https://github.com/KrysztofN/Neural_Network_From_Scratch/blob/main/img/softmax_stable.png" alt="Num" width="200" height="150">
</p>
<p align="center">
<i>Left: Standard Softmax, Right: Numerically stable softmax</i>
</p>


The numerically stable softmax ensures all exponentiated values will be between 0 and 1, preventing overflow errors.

### 4. Forward Propagation

Forward propagation moves input data through the neural network to generate predictions through sequential matrix operations and activation functions.

The process is described by linear combination:
```
Z = W × X + b

Where:
- Z: Layer output before activation
- W: Weight matrix
- X: Input matrix
- b: Bias vector
```

#### Matrix Dimensions and Transposition

<p align="center">
  <img src="https://github.com/KrysztofN/Neural_Network_From_Scratch/blob/main/img/mmult.png" alt="Num" width="350" height="300">
</p>
<p align="center">
<i>Matrices multiplication recap</i>
</p>


For valid matrix multiplication:
- If matrix A is (m × n)
- And matrix B is (n × p)
- Then A × B results in matrix size (m × p)
- Inner dimensions (n) must match


**First Forward Pass:**
```
Z1 = W1ᵀ × X + b1
A1 = ReLU(Z1)

Where:
- W1ᵀ: Transposed weight matrix for first layer
- X: Input data
- b1: First layer bias
- A1: First layer activation output
```

**Second Forward Pass:**
```
Z2 = W2ᵀ × A1 + b2
A2 = Softmax(Z2)

Where:
- W2ᵀ: Transposed weight matrix for second layer
- A1: Output from first layer
- b2: Second layer bias
- A2: Final network output (probabilities)
```

### 5. Loss Function

The loss function serves as a performance metric measuring how well our neural network performs. It:
- Provides a numerical way to measure prediction accuracy
- Converts "correctness" into a measurable number
- Creates a mathematical "landscape" for network improvement
- Guides weight adjustments through gradient calculation
- Lower loss indicates better predictions

This project uses the Cross-Entropy loss function:

<p align="center">
  <img src="https://github.com/KrysztofN/Neural_Network_From_Scratch/blob/main/img/cross_entropy.png" alt="Num" width="400" height="150">
</p>
<p align="center">
<i>Cross-Entropy formula</i>
</p>

Where:
- N: number of training examples
- K: number of classes
- ti,j: true value (0 or 1) for example i and class j
- pij: predicted probability for example i being in class j

### 6. Backward Propagation

Backpropagation calculates gradients of the loss function with respect to the network's weights and biases, forming the core of the learning process.

#### The Learning Process
1. **Forward Propagation**: Network makes predictions
2. **Loss Calculation**: Measure prediction error
3. **Backpropagation**: Calculate gradients
4. **Weight Update**: Adjust parameters using gradients

Backpropagation uses the chain rule from calculus to compute how each weight and bias contributed to the error. For a network with loss L, we need ∂L/∂W and ∂L/∂b for each layer:

```python
m = x_train.shape[0]  # batch size
dz2 = a2 - y_true
dW2 = np.dot(a1.T, dz2) / m
db2 = np.sum(dz2, axis=0, keepdims=True)

dz1 = np.dot(dz2, W2) * (z1 > 0)
dW1 = np.dot(x_train.T, dz1) / m
db1 = np.sum(dz1, axis=0, keepdims=True)
```

### 7. Parameters Update

The parameter update step adjusts network weights and biases based on computed gradients using the gradient descent formula:
```
parameter = parameter - learning_rate × gradient

Where:
- Parameter: Value to optimize (weights or biases)
- Learning Rate: Step size in gradient direction
- Gradient: Direction and magnitude of steepest error increase
```

Important considerations:
- Gradients point toward steepest increase
- Subtracting gradients minimizes loss function
- Learning rate controls update step size
- Too large: Can cause overshooting and oscillations
- Too small: Results in very slow learning

### 8. Training

The training process involves:
- Complete passes through dataset (epochs)
- Processing smaller data subsets (batches)
- Dataset shuffling via random permutation
- Mini-batch Stochastic Gradient Descent
- Regular evaluation of model performance

<p align="center">
  <img src="https://github.com/KrysztofN/Neural_Network_From_Scratch/blob/main/img/training.png" alt="Num" width="250" height="300">
</p>
<p align="center">
<i>Training step visualization</i>
</p>


### 9. Evaluation

The final evaluation:
- Processes test dataset through trained network
- Generates predictions for comparison
- Uses argmax function for class prediction
- Computes accuracy against true labels

<p align="center">
  <img src="https://github.com/KrysztofN/Neural_Network_From_Scratch/blob/main/img/training.png" alt="Num" width="250" height="300">
</p>
<p align="center">
<i>Training step visualization</i>
</p>


Performance Visualization:

<p align="center">
  <img src="https://github.com/KrysztofN/Neural_Network_From_Scratch/blob/main/charts/loss_accuracy.png" alt="Num" width="300" height="250">
</p>
<p align="center">
<i>Accuracy - Loss visualization over 20 epochs</i>
</p>


<p align="center">
  <img src="https://github.com/KrysztofN/Neural_Network_From_Scratch/blob/main/charts/confusion_matrix.png" alt="Num" width="300" height="300">
</p>
<p align="center">
<i>10 class consufion matrix showing prediction accuracy</i>
</p>

