"""
CS440 Spring 2016
Programming Assignment #2
Team Members: Cyril Saade, Rebecca Jellinek, Ivan Uvarov, David Wang
"""

import numpy as np
import matplotlib.pyplot as plt

#Overall, can play with #nodes in hidden-layer, learning rate parameter epsilon, number of epoch/iterations, implementing L2 regularization, coefficient of sigmoid function

class NeuralNet:    
    def __init__(self, input_dim, output_dim, epsilon, hidden_dim = 0, L2Reg = 0): # def __init__(self, input_dim, hidden_dim, output_dim, epsilon): 
        """
        Initializes the parameters of the neural network to random values
        """
        self.epsilon = epsilon
        self.L2Reg = L2Reg
        self.hd = False
        if hidden_dim > 0:
            self.hd = True
            self.Wih = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
            self.bih = np.zeros((1, hidden_dim))
            self.Who = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim)
            self.bho = np.zeros((1, output_dim))
        else:
            self.W = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
            self.b = np.zeros((1, output_dim))
        
    #--------------------------------------------------------------------------
    
    def compute_cost(self,X, y):
        """
        Computes the total loss on the dataset
        """
        num_samples = len(X)
        # Do Forward propagation to calculate our predictions -- so, essentially the predict(self, x) function
        if not self.hd:
            z = X.dot(self.W) + self.b
            exp_z = np.exp(z)
            softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
            cross_ent_err = -np.log(softmax_scores[range(num_samples), y])
            data_loss = np.sum(cross_ent_err)
            return 1./num_samples * data_loss
        else:
            zin = X.dot(self.Wih) + self.bih
            s = 1./(1 + np.exp(-zin)) #can play with -(coefficient)zin
            zout = s.dot(self.Who) + self.bho
            exp_z = np.exp(zout)
            softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
            # Calculate the cross-entropy loss
            cross_ent_err = -np.log(softmax_scores[range(num_samples), y])
            data_loss = np.sum(cross_ent_err)
            #optional adding regularization here
            #data_loss += self.L2Reg/2 * (np.sum(np.square(Wih)) + np.sum(np.square(Who))) #does it need to be L2Reg or L2Reg/2?
            return 1./num_samples * data_loss
    
    #--------------------------------------------------------------------------
 
    def predict(self,x):
        """
        Makes a prediction based on current model parameters
        """
        if not self.hd:
            # Do Forward Propagation
            z = x.dot(self.W) + self.b #the output of inner-layer right before the final output-layer does not apply the activation function -- it simply calls the softmax function
            exp_z = np.exp(z)
            softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
            return np.argmax(softmax_scores, axis=1)
        else:
        #this is for the hidden layer implementation (AKA 3 layers); above is with only 2 layers, input and output-layers
            zin = x.dot(self.Wih) + self.bih
            s = 1./(1 + np.exp(-zin)) #can play with -(coefficient)zin
            zout = s.dot(self.Who) + self.bho
            exp_z = np.exp(zout)
            softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
            return np.argmax(softmax_scores, axis=1)
        
    #--------------------------------------------------------------------------
    
    def fit(self,X,y,num_epochs):
        """
        Learns model parameters to fit the data
        """
        if self.hd:
            for i in range(num_epochs):
                #forward propagation
                zin = X.dot(self.Wih) + self.bih
                s = 1./(1 + np.exp(-zin)) #can play with -(coefficient)zin
                zout = s.dot(self.Who) + self.bho
                exp_z = np.exp(zout)
                softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True) #num_samples X 2 matrix corresponding to all inputs in Toy___X
                        
                #back-propagation
                beta_outer = softmax_scores
                beta_outer[range(len(X)), y] -= 1
                beta_inner = beta_outer.dot(self.Who.T) * (s - np.power(s,2)) #isn't softmax_scores the output of the outer layer, not s(z)?
                dWho = (s.T).dot(beta_outer)
                #dWho = (s.T).dot(beta_outer) * (softmax_scores - np.power(softmax_scores, 2)) #this is what I would have written
                dbho = np.sum(beta_outer, axis=0, keepdims=True)
                dWih = np.dot(X.T, beta_inner)
                #dWho = (X.T).dot(beta_inner) * (s - np.power(s,2)) #this is what I would have written
                dbih = np.sum(beta_inner, axis=0)
                
                #(optionally) add regularization terms (b1 and b2 don't have regularization terms) -- 
                #    partial derivative of Loss-function w/ respect to new term, L2Reg * Complexity(hypothesis), 
                #    where Complexity(hypothesis), for L2, is sum of sqr(abs(weight)) for all weights in the NN --
                #    and since sqr(-weight)= sqr(weight), it's just the partial derivative of the sum of sqr(weight), 
                #    which is just sum of 2(weight) -- the factor of 2 can just be absorbed into L2Reg when adjusting 
                #    anyways, so it's literally just self.L2Reg * self.Who (or self.Wih)
                #dWho += self.L2Reg * self.Who
                #dWih += self.L2Reg * self.Wih
                
                #gradient descent parameter update
                self.Who += -epsilon * dWho
                self.bho += -epsilon * dbho
                self.Wih += -epsilon * dWih
                self.bih += -epsilon * dbih
        else:
            for i in range(num_epochs):
                z = X.dot(self.W) + self.b
                exp_z = np.exp(z)
                softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
                beta_outer = softmax_scores
                beta_outer[range(len(X)), y] -= 1 # 500 x 2 matrix
                #delta = beta_outer * (softmax_scores - np.power(softmax_scores, 2)) #this is what I would have written...
                #X.T is a 2 x 500 matrix; X.T x beta_outer = 2 x 2 matrix
                deltaWeight = epsilon * np.dot(X.T, beta_outer) #...with np.dot(X.T, delta) -- but matrix dimensions would be all wrong
                #(optionally) add regularization terms
                #self.W += -epsilon * (self.L2Reg * self.W)
                self.W += -deltaWeight
                self.b += -epsilon * np.sum(beta_outer, axis=0)

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

def plot_decision_boundary(pred_func):
    """
    Helper function to print the decision boundary given by model
    """
    # Set min and max values
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

#Train Neural Network on
linear = False

#A. linearly separable data
if linear:
    #load data
    X = np.genfromtxt('DATA/ToyLinearX.csv', delimiter=',')
    y = np.genfromtxt('DATA/ToyLineary.csv', delimiter=',')
    y = y.astype(int)
    #plot data
    plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.show()
#B. Non-linearly separable data
else:
    #load data
    X = np.genfromtxt('DATA/ToyMoonX.csv', delimiter=',')
    y = np.genfromtxt('DATA/ToyMoony.csv', delimiter=',')
    y = y.astype(int)
    #plot data
    plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.show()

input_dim = 2 # input layer dimensionality -- dimensionality of data-set in Toy___X.csv, which is 2
output_dim = 2 # output layer dimensionality -- dimensionality of classes, which in Toy___Y.csv, is between 0 and 1 -- so 2

# Gradient descent parameters 
epsilon = 0.01 
L2Reg = 0.01
num_epochs = 5000

# Fit model
#----------------------------------------------
#Uncomment following lines after implementing NeuralNet
#----------------------------------------------
#NN = NeuralNet(input_dim, output_dim, epsilon) 
#or
NN = NeuralNet(input_dim, output_dim, epsilon, 10, L2Reg) #125 Hidden Nodes?
NN.fit(X,y,num_epochs)

print("Cost: {0}".format(NN.compute_cost(X,y)))
predictions = NN.predict(X)
correct = 0.0
for i in range(len(predictions)):
    if predictions[i] == y[i]:
        correct += 1.0
correct /= len(predictions)
print("Accuracy: {0}".format(correct))

# Plot the decision boundary
plot_decision_boundary(lambda x: NN.predict(x))
plt.title("Neural Net Decision Boundary")
            
    