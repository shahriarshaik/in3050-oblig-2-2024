import numpy as np
import matplotlib.pyplot as plt
import sklearn # This is only to generate a dataset



# Generating the dataset
from sklearn.datasets import make_blobs
X, t_multi = make_blobs(n_samples=[400, 400, 400, 400, 400], 
              centers=[[0,1],[4,2],[8,1],[2,0],[6,0]], 
              n_features=2, 
              random_state=2024, 
              cluster_std=[1.0, 2.0, 1.0, 0.5, 0.5])

# Shuffling the dataset
indices = np.arange(X.shape[0])
rng = np.random.RandomState(2024)
rng.shuffle(indices)
indices[:10]

# Splitting into train, dev and test
X_train = X[indices[:1000],:]
X_val = X[indices[1000:1500],:]
X_test = X[indices[1500:],:]
t_multi_train = t_multi[indices[:1000]]
t_multi_val = t_multi[indices[1000:1500]]
t_multi_test = t_multi[indices[1500:]]

t2_train = t_multi_train >= 3
t2_train = t2_train.astype('int')
t2_val = (t_multi_val >= 3).astype('int')
t2_test = (t_multi_test >= 3).astype('int')


def accuracy(predicted, gold):
    return np.mean(predicted == gold)



def bestHyperParameters(etas, epochs, Xtrain, Ttrain):
    '''lager en tom liste for å lagre resultatene'''
    results = []

    '''kjører gjennom alle kombinasjonene av eta og epochs'''
    for eta in etas:
        for epoch in epochs:
            '''lager en ny instans av NumpyLinRegClass'''
            cl = NumpyLogRegClass()
            '''fitter modellen med X_train, t2_train, eta og epoch som parametere'''
            cl.fit(Xtrain, Ttrain, eta=eta, epochs=epoch)
            acc = accuracy(cl.predict(Xtrain), t2_train)
            results.append((acc, eta, epoch))
    
    '''sorterer resultatene etter accuracy'''
    sorted_results = sorted(results, reverse=True, key=lambda x: x[0])

    best_acc, best_eta, best_epoch = sorted_results[0]

    return best_acc, best_eta, best_epoch






def add_bias(X, bias):
    """X is a NxM matrix: N datapoints, M features
    bias is a bias term, -1 or 1, or any other scalar. Use 0 for no bias
    Return a Nx(M+1) matrix with added bias in position zero
    """
    N = X.shape[0]
    biases = np.ones((N, 1)) * bias # Make a N*1 matrix of biases
    # Concatenate the column of biases in front of the columns of X.
    return np.concatenate((biases, X), axis  = 1) 






class NumpyClassifier():
    """Common methods to all Numpy classifiers --- if any"""








class NumpyLogRegClass(NumpyClassifier):

    def __init__(self, bias=-1): 
        self.bias=bias
        self.losses = []
        self.accuracies = []

    # a) logistic regression classifier
    def logistic(self, x):
        return 1 / (1 + np.exp(-x))
    


    '''IMPORTANT the variable 'lr' has been changed to 'eta' '''
    def fit(self, X_train, t_train, eta = 0.1, epochs=10):

        """X_train is a NxM matrix, N data points, M features
        t_train is avector of length N,
        the target class values for the training data
        lr is our learning rate
        """
        
        if self.bias:
            X_train = add_bias(X_train, self.bias)
            

        (N, M) = X_train.shape
        
        self.weights = weights = np.zeros(M)
        
        for e in range(epochs):
            y = self.logistic(X_train @ weights)

            gradient = X_train.T @ (y - t_train) / N
            
            # Update weights
            weights -= eta * gradient
            #weights -= eta / N *  X_train.T @ (X_train @ weights - t_train)

            # Calculate loss and accuracy at the end of each epoch for training set
            train_loss = -np.mean(t_train * np.log(y) + (1-t_train) * np.log(1-y))
            train_acc = accuracy(self.predict(X_train), t_train)
            self.losses.append(train_loss)
            self.accuracies.append(train_acc)      
    


    def predict(self, X):
        # Calculate predicted values using sigmoid function
        y_pred = self.logistic(X @ self.weights)
        # Classify as 1 or 0 based on whether predicted value is above or below 0.5 threshold
        return (y_pred >= 0.5).astype(int)


    #predict_probability() which predict the probability of the data belonging to the positive class
    def predict_probability(self, X):
        # Calculate predicted probabilities using sigmoid function
        y_prob = self.logistic(X @ self.weights)
        return y_prob
    

'''generer 500 tall for eta'''
etas3 = np.linspace(0.001, 0.01, num=100) 
etas2 = np.linspace(0.01, 0.1, num=200) 
etas1 = np.linspace(0.1, 1, num=200) 

'''setter de sammen til en array'''
etas_combined = np.concatenate(( etas3, etas2, etas1))

'''generer 10 tall for epochs'''
epochs = np.linspace(195, 205, num=10, dtype=int)




best_acc, best_eta, best_epoch = bestHyperParameters(etas_combined, epochs, X_train, t2_train)

def plot_decision_regions(X, t, clf=[], size=(8,6)):
    """Plot the data set (X,t) together with the decision boundary of the classifier clf"""
    # The region of the plane to consider determined by X
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # Make a prediction of the whole region
    h = 0.02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Classify each meshpoint.
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=size) # You may adjust this

    # Put the result into a color plot
    plt.contourf(xx, yy, Z, alpha=0.2, cmap = 'Paired')

    plt.scatter(X[:,0], X[:,1], c=t, s=10.0, cmap='Paired')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Decision regions")
    plt.xlabel("x0")
    plt.ylabel("x1")


print("\nBest hyperparameters:")
print('eta =', best_eta, ', epochs =', best_epoch, ', accuracy =', best_acc)


cl = NumpyLogRegClass()
cl.fit(X_train, t2_train, eta=best_eta, epochs=best_epoch)
plot_decision_regions(X_train, t2_train, cl)
 