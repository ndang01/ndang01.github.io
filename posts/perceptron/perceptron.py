#Implementation of the perceptron algorithm

import numpy as np
import random

class Perceptron:

    #initializes perceptron class and its instance variables - history and weight
    #history is a list of the evolution of the score over the training period
    #weight is the vector w_tilde = (w,-b)
    def __init__(self, history = None):
        self.history = []


    #finds a hyperplane that approximately divides the data into its two classes
    def fit(self, X, y, max_steps):
        #starts with random weight vector
        self.w = np.random.rand(len(X.shape) + 1)

        #performs update until maximum number of steps is reached
        for _ in range(max_steps):
            #picks a random index 
            i = np.random.randint(0,100)

            x_tilde = np.append(X, np.ones((X.shape[0], 1)), 1)
            y_tilde = 2 * y - 1

            #update stops when the accuracy reaches 1.0
            if self.score(x_tilde, y) == 1.0:
                break
            
            #compute the next weight vector
            self.w = self.w + ((y_tilde[i] * (self.w@x_tilde[i]) < 0 ) * 1) * y_tilde[i] * x_tilde[i]


    #returns y_hat vector of predicted labels
    def predict(self, X):
        return np.sign(X@self.w)
        

    #returns the accuracy of the perceptron as a number between 0 and 1
    #with 1 corresponding to perfect classification
    def score(self, X, y):
        #set the weight and X vectors to be of the same size
        if len(self.w) != X.shape[1]:
            X = np.append(X, np.ones((X.shape[0], 1)), 1)

        y_hat = self.predict(X)
        y_tilde = 2 * y - 1

        #calculates the accuracy between y_hat and y
        accuracy = (y_hat == y_tilde).mean()
        #logs the accuracy in the history variable
        self.history.append(accuracy)
        
        return accuracy 




    
