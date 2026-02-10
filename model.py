
import numpy as np

class McLarenEngine:
    def __init__(self):
        # Initialize the New Random Generator
        self.rng = np.random.default_rng(seed=1)
        
        # 2 inputs, 3 hidden neurons, 1 output
        # Initialize weights with small random numbers
        self.weights1 = self.rng.random((2, 3)) - 0.5
        self.weights2 = self.rng.random((3, 1)) - 0.5

    def sigmoid(self, x):
        """The activation function (squashes values between 0 and 1)"""
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """Used for backpropagation (calculus)"""
        return x * (1 - x)

    def think(self, inputs):
        """The Forward Pass: Input -> Hidden -> Output"""
        self.layer1 = self.sigmoid(np.dot(inputs, self.weights1))
        self.output = self.sigmoid(np.dot(self.layer1, self.weights2))
        return self.output

    def train(self, inputs, real_output, iterations=10000):
        """The Learning Loop: Error Calculation -> Weight Adjustment"""
        print(f"MCLAREN OS 1: Training engine for {iterations} cycles...")
        for _ in range(iterations):
            # 1. Forward Pass
            output = self.think(inputs)

            # 2. Calculate Error (The Gap)
            error_output = real_output - output
            
            # 3. Backpropagation (The adjustment)
            delta_output = error_output * self.sigmoid_derivative(output)
            error_hidden = delta_output.dot(self.weights2.T)
            delta_hidden = error_hidden * self.sigmoid_derivative(self.layer1)

            # 4. Update Weights (Turning the 'knobs')
            self.weights2 += self.layer1.T.dot(delta_output)
            self.weights1 += inputs.T.dot(delta_hidden)
        print("Training Complete.")

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    # Create the brain
    brain = McLarenEngine()

    # Training Data (XOR Logic: 0,0=0 | 0,1=1 | 1,0=1 | 1,1=0)
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([[0], [1], [1], [0]])

    # Train the OS
    brain.train(X, y, iterations=10000)

    # Test the OS
    print("\nFinal System Predictions:")
    print(brain.think(X))
