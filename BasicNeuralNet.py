import numpy as np 
import time      
X=np.array(([0,1,0],[1,1,1],[0,0,0],[1,0,1]), dtype=float)
y=np.array(([1],[0],[0],[1]), dtype=float)
print ("")
print("Answer outputs should be either 1 or 0")
print ("")
time.sleep(2)
y0 = raw_input("Enter first answer output: ")
print ("")
y1 = raw_input("Enter second answer output: ")
print ("")
y2 = raw_input("Enter thrid answer output: ")
print ("")
y3 = raw_input("Enter fourth answer output: ")
print ("")
lives = raw_input("Enter the number of iterations(Usually in the thousands): ")
iterations = int (lives)
y[0] = y0
y[1] = y1
y[2] = y2
y[3] = y3
def sigmoid(t):
    return 1/(1+np.exp(-t))
def sigmoid_derivative(p):
    return p * (1 - p)
class NeuralNetwork:
    def __init__(self, x,y):
        self.input = x
        self.weights1= np.random.rand(self.input.shape[1],4)
        self.weights2 = np.random.rand(4,1)
        self.y = y
        self.output = np. zeros(y.shape)
    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        return self.layer2      
    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, 2*(self.y -self.output)*sigmoid_derivative(self.output))
        d_weights1 = np.dot(self.input.T, np.dot(2*(self.y -self.output)*sigmoid_derivative(self.output), self.weights2.T)*sigmoid_derivative(self.layer1))
        self.weights1 += d_weights1
        self.weights2 += d_weights2
    def train(self, X, y):
        self.output = self.feedforward()
        self.backprop()
NN = NeuralNetwork(X,y)
for i in range(iterations): # trains the NN 1,000 times
    if i % 100 ==0: 
        print ("for iteration # " + str(i) + "\n")
        print ("Input : \n" + str(X))
        print ("Actual Output: \n" + str(y))
        print ("Predicted Output: \n" + str(NN.feedforward()))
        #print ("Loss: \n" + str(np.mean(np.square(y - NN.feedforward())))) # mean sum squared loss
        print ("\n")
    NN.train(X, y)