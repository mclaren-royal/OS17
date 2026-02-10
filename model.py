
import numpy as np

class McLarenEngine:
    def __init__(self):
        # 2 inputs, 3 hidden neurons, 1 output
        self.weights1 = np.random.rand(2, 3) 
        self.weights2 = np.random.rand(3, 1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def think(self, inputs):
        # The Forward Pass
        layer1 = self.sigmoid(np.dot(inputs, self.weights1))
        output = self.sigmoid(np.dot(layer1, self.weights2))
        return output

# Initialize the system
brain = McLarenEngine()
