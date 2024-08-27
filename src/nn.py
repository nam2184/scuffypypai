import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import random 
import time
from .paillier import *

class MLP:
    def __init__(self, input_size, hidden_size, output_size, private_key):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.private_key = private_key

        # Initialize weights and biases
        self.W1 = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(input_size)]
        self.b1 = [0 for _ in range(hidden_size)]
        self.W2 = [[random.uniform(-1, 1) for _ in range(output_size)] for _ in range(hidden_size)]
        self.b2 = [0 for _ in range(output_size)]

    def sigmoid(self,x):
      return 1 / (1 + np.exp(-x))

    def forward(self, X, prev_hidden = None):
        self.z1 = [sum((X[i] * self.W1[i][j] for i in range(self.input_size))) + self.b1[j] for j in range(self.hidden_size)]
        if self.private_key != None :
            self.a1 = decrypt_vector(self.private_key, self.z1)  # No activation function
        else : 
            self.a1 = np.array(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        #self.a2 = decrypt_vector(self.private_key, self.z2)
        self.a2 = self.sigmoid(np.array(self.z2))
        return self.a2, None


    def backward(self, y, output, learning_rate, prev_hidden = None):
      # Backpropagation to update weights and biases
      self.W1 = np.array(self.W1)
      self.b1 = np.array(self.b1, dtype = np.float64)
      self.W2 = np.array(self.W2)
      self.b2 = np.array(self.b2, dtype = np.float64)
      self.loss = mean_absolute_error(output, y)

      # Compute delta_z2
      delta_z2 = np.array(output) - np.array(y)
      # Update weights and biases for output layer (W2 and b2)
      delta_W2 = np.outer(self.a1, delta_z2)
      delta_b2 = delta_z2

      delta_z1 = np.dot(self.W2, delta_z2.reshape(-1, 1))
      delta_z1 = delta_z1.reshape(-1)

      # Update weights and biases for hidden layer (W1 and b1) excluding inputs
      delta_W1 = np.outer(delta_z1, np.ones(self.input_size))
      delta_b1 = delta_z1

      # Update weights and biases using the computed deltas and learning rate
      self.W1 -= learning_rate * delta_W1
      self.b1 -= learning_rate * delta_b1
      self.W2 -= learning_rate * delta_W2
      self.b2 -= learning_rate * delta_b2

      self.W1 = self.W1.tolist()
      self.b1 = self.b1.tolist()
      self.W2 = self.W2.tolist()
      self.b2 = self.b2.tolist()


class RNN:
    def __init__(self, input_size, hidden_size, output_size, private_key):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.private_key = private_key

        # Initialize weights and biases for input to hidden layer
        self.W1 = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(input_size)]
        self.b1 = [0 for _ in range(hidden_size)]

        # Initialize weights and biases for hidden to output layer
        self.W2 = [[random.uniform(-1, 1) for _ in range(output_size)] for _ in range(hidden_size)]
        self.b2 = [0 for _ in range(output_size)]

    def sigmoid(self,x):
      return 1 / (1 + np.exp(-x))

    def forward(self, X, prev_hidden=None):
        if prev_hidden is None:
            prev_hidden = [0 for _ in range(self.hidden_size)]

        self.z1 = [sum((X[i] * self.W1[i][j] for i in range(self.input_size))) + sum((prev_hidden[j] * self.W1[i][j] for i in range(self.input_size))) + self.b1[j] for j in range(self.hidden_size)]
        if self.private_key != None :
            self.a1 = decrypt_vector(self.private_key, self.z1)
        else : 
            self.a1 = np.array(self.z1)
        self.a1 = self.sigmoid(self.a1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)

        return self.a2, np.array(self.a1)

    def backward(self, y, output, learning_rate, prev_hidden=None):
        if prev_hidden is None:
            prev_hidden = np.zeros(self.hidden_size)

        self.W1 = np.array(self.W1)
        self.b1 = np.array(self.b1, dtype = np.float64)
        self.W2 = np.array(self.W2)
        self.b2 = np.array(self.b2, dtype = np.float64)
        self.loss = mean_squared_error(y, output)

        delta_z2 = output - y
        delta_W2 = np.outer(self.a1, delta_z2)
        delta_b2 = delta_z2

        delta_a1 = np.dot(delta_z2, self.W2.T)
        delta_z1 = delta_a1 * self.a1 * (1 - self.a1)
        delta_W1 = np.outer(prev_hidden, delta_z1)
        delta_b1 = delta_z1

        self.W2 -= learning_rate * delta_W2
        self.b2 -= learning_rate * delta_b2
        self.W1 -= learning_rate * delta_W1
        self.b1 -= learning_rate * delta_b1

        self.W1 = self.W1.tolist()          
        self.b1 = self.b1.tolist()
        self.W2 = self.W2.tolist()
        self.b2 = self.b2.tolist()
