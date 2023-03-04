#Implementation of the optimization algorithms based on the gradients of functions
import numpy as np

class LogisticRegression:

    ##initializes logistic regression class and its instance variables
    def __init__(self, loss_history = None, score_history = None, w = None):
        self.score_history = []    #list of the evolution of the score
        self.loss_history = []     #list of the evolution of the loss
        self.w = np.zeros(3)       #weight vector
        
    
    #finds a hyperplane that approximately divides the data into its two classes via gradient descent
    #alpha = learning rate 
    #max_epochs = max number of iterations
    def fit(self, X, y, alpha, max_epochs):
        x_tilde = np.append(X, np.ones((X.shape[0], 1)), 1)       
        prev_loss = np.inf      #start off the loss

        # main loop
        for _ in range(max_epochs):
            y_hat = self.predict(x_tilde)

            #gradient step
            self.w = self.w - alpha * ((self.sigmoid(y_hat)-y)[:,np.newaxis] * x_tilde).mean(axis = 0)
            #log the accuracy
            self.score(x_tilde,y)
            #compute loss
            new_loss = self.loss(x_tilde,y)

            #declare convergence when the improvement in the function is small enough in magnitude
            if np.isclose(new_loss, prev_loss):          
                break
            else:
                prev_loss = new_loss  


    #finds a hyperplane that approximately divides the data into its two classes via stochastic gradient descent
    #alpha = learning rate 
    #max_epochs = max number of iterations
    #batch_size = how the data will be split by
    def fit_stochastic(self, X, y, alpha, max_epochs, batch_size):
        x_tilde = np.append(X, np.ones((X.shape[0], 1)), 1)    
        prev_loss = np.inf    #start off the loss
        n = X.shape[0]

        #some of code is taken from class notes
        for j in np.arange(max_epochs):
            order = np.arange(n)
            np.random.shuffle(order)     #shuffle the points randomly

            for batch in np.array_split(order, n // batch_size + 1):
                x_batch = x_tilde[batch,:]
                y_batch = y[batch]
                y_hat = self.predict(x_batch)

                # gradient step
                self.w = self.w - alpha * ((self.sigmoid(y_hat)-y_batch)[:,np.newaxis] * x_batch).mean(axis = 0)
            
            #log the accuracy
            self.score(x_tilde,y)
            #compute loss
            new_loss = self.loss(x_tilde,y)

            #declare convergence when the improvement in the function is small enough in magnitude
            if np.isclose(new_loss, prev_loss):          
                break
            else:
                prev_loss = new_loss  


    #returns y_hat vector of predicted labels
    def predict(self, X):
        return X@self.w
    

    #returns the accuracy of the gradient descent as a number between 0 and 1
    #with 1 corresponding to perfect classification
    def score(self, X, y):
        #set the weight and X vectors to be of the same size
        if len(self.w) != X.shape[1]:
            X = np.append(X, np.ones((X.shape[0], 1)), 1)

        y_hat = self.predict(X)

        #calculates the accuracy between y_hat and y
        y_ = 1 * (y_hat > 0)
        accuracy = (y_ == y).mean()
        #logs the accuracy in the history variable
        self.score_history.append(accuracy)
        
        return accuracy 
    

    #taken from lecture notes
    #calculates sigmoid of a number
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


    #parts of code is taken from lecture notes
    #returns the overall loss (empirical risk)
    def loss(self, X, y): 
        if len(self.w) != X.shape[1]:
            X = np.append(X, np.ones((X.shape[0], 1)), 1)

        y_hat = self.predict(X)
        loss =  (-y * np.log(self.sigmoid(y_hat)) - (1-y)*np.log(1-self.sigmoid(y_hat))).mean()
        self.loss_history.append(loss)
        return loss

