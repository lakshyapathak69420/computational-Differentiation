import numpy as np
from backward_mode_differentiation.tensor import Tensor
from backward_mode_differentiation.tensorFunction import tensorFn
class MNISTNet:
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        #Initialization for weights
        self.W1 = Tensor(np.random.randn(input_size, hidden_size) * np.sqrt(2./input_size))
        self.b1 = Tensor(np.zeros((1, hidden_size)))
        
        self.W2 = Tensor(np.random.randn(hidden_size, output_size) * np.sqrt(2./hidden_size))
        self.b2 = Tensor(np.zeros((1, output_size)))
        
        self.params = [self.W1, self.b1, self.W2, self.b2]

    def forward(self, X):
        # Input to Hidden
        # X should be (batch_size, 784)
        z1 = tensorFn.matmul(X, self.W1) + self.b1
        a1 = tensorFn.relu(z1)
        
        # Hidden to Output (Logits)
        logits = tensorFn.matmul(a1, self.W2) + self.b2
        return logits

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.value)

    def step(self, lr):
        for p in self.params:
            p.value -= lr * p.grad

    def output(self, input_vector):
        """
        returns a tuple of predicted label and probability 
        """
        
        logits = self.forward(input_vector).value[0] # Get the row vector
        
        max_value = -float('inf')
        predicted_label = 0
        
        for i in range(len(logits)):
            if logits[i] > max_value:
                max_value = logits[i]
                predicted_label = i
                        
        sum_exp = 0
        for val in logits:
            sum_exp += np.exp(val - max_value)
            
        prob = np.exp(logits[predicted_label] - max_value) / sum_exp
        
        return (predicted_label, float(prob))
                




