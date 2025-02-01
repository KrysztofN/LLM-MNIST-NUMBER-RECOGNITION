#  Mnist dataset reader
import numpy as np 
import struct
from array import array
from os.path  import join

class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)  


    def preprocess_data(self, x_train, y_train, x_test, y_test):
        x_train = np.array(x_train, dtype='float32') # vector (60000, 28, 28)
        x_test = np.array(x_test, dtype='float32') # vector (10000, 28, 28)
        y_train = np.array(y_train, dtype='int32') # vector (60000)
        y_test = np.array(y_test, dtype='int32') # vector (10000)
        
        x_train = x_train / 255.0
        x_test = x_test / 255.0

        x_train = x_train.reshape(x_train.shape[0], -1) # vector (60000, 784)
        x_test = x_test.reshape(x_test.shape[0], -1)
        
        def to_one_hot(y, num_classes=10): # one hot encoding
            return np.eye(num_classes)[y] 
        
        y_train_onehot = to_one_hot(y_train) # vector (60000, 10)
        y_test_onehot = to_one_hot(y_test) # vector (10000, 10)
        
        return x_train, y_train_onehot, x_test, y_test_onehot
    
class miniModel():
    def __init__(self):
        pass

    def initialize_parameters(self, input_size, hidden_size, output_size):
        np.random.seed(np.random.randint(1000))
        
        # 1. Uniform distribution
        # W1 = np.random.uniform(-1/np.sqrt(input_size), 1/np.sqrt(input_size), 
        #                        size=(hidden_size, input_size))
        # W2 = np.random.uniform(-1/np.sqrt(hidden_size), 1/np.sqrt(hidden_size), 
        #                        size=(output_size, hidden_size))
        
        # 2. Xavier Normal
        # W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2.0/input_size)
        # W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2.0/hidden_size)
        
        # 3. Xavier Uniform
        # W1 = np.random.uniform(-np.sqrt(6)/np.sqrt(input_size + hidden_size),
        #                       np.sqrt(6)/np.sqrt(input_size + hidden_size), 
        #                       size=(hidden_size, input_size))
        # W2 = np.random.uniform(-np.sqrt(6)/np.sqrt(hidden_size + output_size),
        #                       np.sqrt(6)/np.sqrt(hidden_size + output_size), 
        #                       size=(output_size, hidden_size))

        # 4. He uniform
        # W1 = np.random.uniform(-np.sqrt(6/input_size), np.sqrt(6/input_size), size = (hidden_size, input_size))
        # W2 = np.random.uniform(-np.sqrt(6/input_size), np.sqrt(6/input_size), size = (output_size, hidden_size))
        
        # 5. He normal
        W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2/input_size)
        W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2/input_size)
        
        b1 = np.zeros((1, hidden_size))
        b2 = np.zeros((1, output_size))
        
        return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis = 1, keepdims=True))
        return exps/np.sum(exps, axis=1, keepdims=True)

    # Linear combination calculations
    # hiden layer activation z1 = W1 * X + b1 , ReLU a1 = ReLU(z1)
    # output layer activation z2 = W2 * a1 + b2, softmax a2 = softmax(z2)

    def forward_propagation(self, X, params):
        # Hidden Layer
        z1 = np.dot(X, params["W1"].T) + params["b1"]  
        a1 = self.relu(z1)
        
        # Output Layer
        z2 = np.dot(a1, params["W2"].T) + params["b2"] 
        a2 = self.softmax(z2)

        return {"z1":z1, "a1":a1, "z2":z2, "a2":a2}

    # Cross-Entropy loss function
    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred + 1e-15))/m

    # backward_propagation
    def backward_propagation(self, X, y_true, params, forward_cache):
        m = X.shape[0]
        
        dz2 = forward_cache["a2"] - y_true                   # (m, 10)
        dW2 = np.dot(forward_cache["a1"].T, dz2) / m         # (128, 10)
        db2 = np.sum(dz2, axis=0, keepdims=True) / m                        # (10,)
        
        dz1 = np.dot(dz2, params["W2"]) * (forward_cache["z1"] > 0)  # (m, 128)
        dW1 = np.dot(X.T, dz1) / m                           # (784, 128)
        db1 = np.sum(dz1, axis=0, keepdims=True) / m                        # (128,)
        
        return {"dW1": dW1.T, "db1": db1, "dW2": dW2.T, "db2": db2}

    def update_parameters(self, params, grads, learning_rate=0.01):
        params["W1"] -= learning_rate * grads["dW1"]
        params["b1"] -= learning_rate * grads["db1"]
        params["W2"] -= learning_rate * grads["dW2"]
        params["b2"] -= learning_rate * grads["db2"]
        return params

    def train(self, X, y, params, epochs=10, batch_size=64, learning_rate=0.01):
        for epoch in range(epochs):
            permutation = np.random.permutation(X.shape[0])
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]
            
            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                cache = self.forward_propagation(X_batch, params)
                grads = self.backward_propagation(X_batch, y_batch, params, cache)
                params = self.update_parameters(params, grads, learning_rate)
            
            cache = self.forward_propagation(X, params)
            loss = self.compute_loss(y, cache["a2"])
            predictions = np.argmax(cache["a2"], axis=1)
            accuracy = np.mean(predictions == np.argmax(y, axis=1))
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | Accuracy: {accuracy*100:.2f}%")
        
        return params

    def evaluate(self, X_test, y_test, params):
        cache = self.forward_propagation(X_test, params)
        predictions = np.argmax(cache["a2"], axis=1)
        accuracy = np.mean(predictions == np.argmax(y_test, axis=1))
        print(f"Test Accuracy: {accuracy * 100:.2f}%")


input_path = 'data'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
x_train, y_train_onehot, x_test, y_test_onehot = mnist_dataloader.preprocess_data(x_train, y_train, x_test, y_test)

model = miniModel()
params = model.initialize_parameters(784, 128, 10)
params = model.train(x_train, y_train_onehot, params, epochs=20, batch_size=64, learning_rate=0.01)
model.evaluate(x_test, y_test_onehot, params)
sample_image = x_test[20] 
predicted_digit = np.argmax(model.forward_propagation(sample_image.reshape(1, -1), params)["a2"])
print(f"Predicted: {predicted_digit}, True: {np.argmax(y_test_onehot[0])}")