X = np.random.rand(10, 3)
X = pad(X)
w = np.random.rand(X.shape[1])

y = X@w + np.random.randn(X.shape[0])

def predict(X, w):
    return X@w

def score(X,y,w):
    y_bar = y.mean()
    y_hat = predict(X)

    top = ((y_hat - y) ** 2 ).sum()
    bottom = ((y_bar - y) ** 2 ).sum()
    c = 1 - (top / bottom)

    return c

