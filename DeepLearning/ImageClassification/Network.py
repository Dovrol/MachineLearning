import numpy as np

sigma = lambda x: 1 / (1 + np.exp(-x))
sigma_prim = lambda x: sigma(x) * (1 - sigma(x))
class Layer:
    
    def __init__(self, numberOfNeurons, numberOfInputs):
        self.numberOfInputs = numberOfInputs
        self.numberOfNeurons = numberOfNeurons
        self.weights = np.random.randn(numberOfNeurons, numberOfInputs)
        self.bias = np.random.randn(numberOfNeurons)
        
        self.tmp_input = None
        self.tmp_gradient = None
        self.tmp_output = None
        
    def forward(self, inputVector):
        self.tmp_input = np.copy(inputVector)
        self.tmp_output = sigma(np.dot(self.weights, inputVector) + self.bias)
        return self.tmp_output
        
    def backward(self, gradient):
        self.tmp_gradient = gradient * self.tmp_output * (1 - self.tmp_output)
        return self.weights.T.dot(self.tmp_gradient)
        
        
    def learn(self, learningRate):
        self.w = np.outer(self.tmp_gradient, self.tmp_input.T)
        self.b = self.tmp_gradient
        self.weights += learningRate* self.w
        self.bias += learningRate* self.b


class Network:
    def __init__(self, inputs):
        self.layers = []
        self.numberOfInputs = inputs
        self.numberOfLayers = 0
        
    def addLayer(self, neurons):
        if not self.layers:
            self.layers.append(Layer(neurons, self.numberOfInputs))
        else:
            self.layers.append(Layer(neurons, self.layers[-1].numberOfNeurons))
        
    def forward(self, inputVector):
        for i in range(len(self.layers)):
            inputVector = self.layers[i].forward(inputVector)    
        return inputVector
    
    def backward(self, gradient):
        i = len(self.layers) - 1
        while i >= 0:
            gradient = self.layers[i].backward(gradient)
            i -= 1
        return gradient
            
    def learn(self,learningRate):
        for i in self.layers:
            i.learn(learningRate)
        