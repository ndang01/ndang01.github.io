import numpy as np

class LinearRegression:

    #initializes logistic regression class and its instance variables
    def __init__(self, loss_history = None, score_history = None):
      self.score_history = []                            #list of the evolution of the score

    def fit_analytic(self, X, y):
      X = np.append(X, np.ones((X.shape[0], 1)), 1)       

      self.w = np.linalg.inv(X.T@X)@X.T@y
    
    def fit_gradient(self, X, y, alpha, max_iter):
      X = np.append(X, np.ones((X.shape[0], 1)), 1)  
      self.w = np.zeros(len(X.shape))                #weight vector     
      P = X.T@X
      q = X.T@y

      for i in range(int(max_iter)):
        #gradient step
        self.w = self.w - alpha * (P@self.w - q)
        
        #log the accuracy
        self.score(X,y)

        if self.score_history[i] < self.score_history[i-1]:
          break
        
    

    def predict(self, X):
      return X@self.w

    def score(self, X, y):
      if len(self.w) != X.shape[1]:
            X = np.append(X, np.ones((X.shape[0], 1)), 1)

      y_bar = y.mean()
      y_hat = self.predict(X)

      num = ((y_hat - y) ** 2 ).sum()
      de = ((y_bar - y) ** 2 ).sum()
      c = 1 - (num / de)

      self.score_history.append(c)
      return c