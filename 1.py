import numpy as np
import matplotlib.pyplot as plt
import sklearn # This is only to generate a dataset
from sklearn.datasets import make_blobs


scale_x = True




class StandardScaler():
    """
    Normalizing the data set should be done by the same transform for every set.
    Reason: if you have a A set where x (coordinate) values are between
    0 and 10 and a B set where x values are between 10 and 20,
    both of them would be normalized to values between 0 and 1. And it would seem
    the model is fitted to both sets, when in reality it should not be fitted to the B set.
    """
    def __init__(self):
        self.mean = 0
        self.std = 0

    def fit(self, X):
        """
        Compute the mean and std of the dataset X
        Usally done on the whole set (my opinion)
        """
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def transform(self, X):
        """
        Transform the dataset X with the mean and std found in the fit function
        """
        return (X - self.mean) / self.std


class NumpyClassifier():
    """Common methods to all Numpy classifiers --- if any"""


    def add_bias(self, X, bias):
        """X is a NxM matrix: N datapoints, M features
        bias is a bias term, -1 or 1, or any other scalar. Use 0 for no bias
        Return a Nx(M+1) matrix with added bias in position zero
        """
        N = X.shape[0]
        biases = np.ones((N, 1)) * bias # Make a N*1 matrix of biases
        # Concatenate the column of biases in front of the columns of X.
        return np.concatenate((biases, X), axis  = 1)

    
    def logistic(self, x):
        return 1/(1+np.exp(-x))
    
    def cross_entropy(self, y, t):
        return - np.mean(t * np.log(y) + (1-t) * np.log(1-y))
    

    def show_stats(self):
        """Plot the accuracy and cross entropy for the training and validation set
        Only works if variables are present. They are present in the logistic regression class
        and the multi-class logistic regression class."""
        plt.plot(self.accuracies)
        plt.plot(self.accuracies_val)
        plt.title("Accuracy for the training and validation set")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Training set", "Validation set"])
        plt.show()

        plt.plot(self.cross_entropies)
        plt.plot(self.cross_entropies_val)
        plt.title("Cross entropy loss for the training and validation set")
        plt.ylabel("Cross entropy loss")
        plt.xlabel("Epoch")
        plt.legend(["Training set", "Validation set"])
        plt.show()


class NumpyLinRegClass(NumpyClassifier):
    def __init__(self, bias=-1):
        self.bias=bias
    
    def fit(self, X_train, t_train, lr = 0.1, epochs=10):
        """X_train is a NxM matrix, N data points, M features
        t_train is avector of length N,
        the target class values for the training data
        lr is our learning rate
        """
        
        if self.bias:
            X_train = self.add_bias(X_train, self.bias)
            
        (N, M) = X_train.shape
        
        self.weights = weights = np.zeros(M)
        
        for _ in range(epochs):
            weights -= lr / N *  X_train.T @ (X_train @ weights - t_train) 

    
    def predict(self, X, threshold=0.5):
        """X is a KxM matrix for some K>=1
        predict the value for each point in X"""
        if self.bias:
            X = self.add_bias(X, self.bias)
        ys = X @ self.weights
        return ys > threshold
    

class NumpyLogRegClass(NumpyClassifier):
    def __init__(self, bias=-1):
        self.bias = bias
        self.epochs = 0
        self.accuracies = []
        self.cross_entropies = []
        self.accuracies_val = []
        self.cross_entropies_val = []
    
    def forward(self, x):
        return self.logistic(x @ self.weights)



    def fit(self, 
            X_train, 
            t_train,
            X_val = None,
            t_val = None,
            lr = 1,
            n_epochs_no_update = 5, 
            tol = 0.001):
        """X_train is a Nxm matrix, N data points, m features
        t_train is avector of length N,
        the targets values for the training data"""

        temp_epochs_no_update = 0
        run = True

        (N, m) = X_train.shape
        X_train = self.add_bias(X_train, self.bias)
        
        self.weights = weights = np.zeros(m+1)
        
        while run:
            weights -= lr / N *  X_train.T @ (self.forward(X_train) - t_train)

            # Calculate the cross entropy and accuracy for the training and validation set
            self.accuracies.append(accuracy(self.predict(X_train[:,1:3]), t_train)) # removing the bias from X_train
            self.cross_entropies.append(self.cross_entropy(self.forward(X_train), t_train))
            if X_val is not None:
                self.accuracies_val.append(accuracy(self.predict(X_val), t_val))
                self.cross_entropies_val.append(self.cross_entropy(self.forward(self.add_bias(X_val, self.bias)), t_val))
            
            # Check if we should stop
            if len(self.accuracies) > n_epochs_no_update:
                #if self.cross_entropies[-1]  + tol >= self.cross_entropies[-2]:
                if self.accuracies[-1] <= self.accuracies[-1-n_epochs_no_update] + tol:
                    temp_epochs_no_update += 1
                    if temp_epochs_no_update >= n_epochs_no_update:
                        run = False
                else:
                    temp_epochs_no_update = 0
        self.epochs = len(self.accuracies)
    

    def predict(self, x, threshold=0.5):
        """X is a Kxm matrix for some K>=1
        predict the value for each point in X"""
        z = self.add_bias(x, self.bias)
        return (self.forward(z) > threshold).astype('int')


    def predict_probability(self, x):
        """X is a Kxm matrix for some K>=1
        predict the probability for each point in X"""
        z = self.add_bias(x, self.bias)
        return self.forward(z)
    

class NumpyMultiLogRegClass(NumpyClassifier):
    def __init__(self, bias=-1):
        self.bias=bias
        self.epochs = 0
        self.classes = []


    def fit(self, 
            X_train, 
            t_train,
            lr = 1,
            n_epochs_no_update = 5, 
            tol = 0.001):
        """X_train is a Nxm matrix, N data points, m features
        t_train is avector of length N,
        the targets values for the training data"""

        (N, m) = X_train.shape

        for i_class in range(t_multi_train.max() + 1):
            t_class = t_train == i_class

            cl = NumpyLogRegClass()
            cl.fit(X_train, np.asarray(t_class, dtype=int), lr=lr, n_epochs_no_update=n_epochs_no_update, tol=tol)
            self.classes.append(cl)

            #plot_decision_regions(X_train, np.asarray(t_class, dtype=int), cl)
            #cl.show_stats()


    def predict(self, x):
        """
        predict which class each point in X belongs to
        Does so by predicting the probability for each class and then
        choosing the class with the highest probability (one-vs-rest)
        """

        predictions = []
        for i_class in self.classes:
            predictions.append(i_class.predict_probability(x))
        
        predictions = np.array(predictions)
        max_value = []
        for i in range(len(predictions[0])):
            tmp_list = [x for x in predictions[:,i]]
            max_value.append(tmp_list.index(max(tmp_list)))

        return np.array(max_value)


    def predict_probability(self, x):
        """Can be implemented by the predict function"""
        predictions = []
        for i_class in self.classes:
            predictions.append(i_class.predict_probability(x))
        
        return np.array(predictions)


class MLPBinaryLinRegClass(NumpyClassifier):
    """A multi-layer neural network with one hidden layer"""
    
    def __init__(self, bias=-1, dim_hidden = 6):
        """Intialize the hyperparameters"""
        self.bias = bias
        # Dimensionality of the hidden layer
        self.dim_hidden = dim_hidden

        self.activ = self.logistic
        self.activ_diff = self.logistic_diff

        self.epochs = 0
        self.cross_entropies = []
        self.accuracies = []
        self.cross_entropies_val = []
        self.accuracies_val = []

    def logistic_diff(self, y):
        return y * (1 - y)
        
    def forward(self, X):
        """TODO: 
        Perform one forward step. 
        Return a pair consisting of the outputs of the hidden_layer
        and the outputs on the final layer"""
        
        hidden_outs = self.add_bias(self.logistic(X @ self.weights1), self.bias)
        outputs = self.logistic(hidden_outs @ self.weights2)
        return hidden_outs, outputs

    
    def fit(self,
            X_train,
            t_train,
            X_val = None,
            t_val = None,
            lr=0.001,
            n_epochs_no_update = 5, 
            tol = 0.001):
        """Intialize the weights. Train *epochs* many epochs.
        
        X_train is a NxM matrix, N data points, M features
        t_train is a vector of length N of targets values for the training data, 
        where the values are 0 or 1.
        lr is the learning rate
        """

        temp_epochs_no_update = 0
        run = True

        
        # Turn t_train into a column vector, a N*1 matrix:
        T_train = t_train.reshape(-1,1)
            
        dim_in = X_train.shape[1] 
        dim_out = T_train.shape[1]
        
        # Initialize the weights
        self.weights1 = (np.random.rand(
            dim_in + 1, 
            self.dim_hidden) * 2 - 1)/np.sqrt(dim_in)
        self.weights2 = (np.random.rand(
            self.dim_hidden+1, 
            dim_out) * 2 - 1)/np.sqrt(self.dim_hidden)
        X_train_bias = self.add_bias(X_train, self.bias)
        
        while run:
            # One epoch
            # The forward step:
            hidden_outs, outputs = self.forward(X_train_bias)
            # The delta term on the output node:
            out_deltas = (outputs - T_train)
            # The delta terms at the output of the hidden layer:
            hiddenout_diffs = out_deltas @ self.weights2.T
            # The deltas at the input to the hidden layer:
            hiddenact_deltas = (hiddenout_diffs[:, 1:] * 
                                self.activ_diff(hidden_outs[:, 1:]))

            # Update the weights:
            self.weights2 -= lr * hidden_outs.T @ out_deltas
            self.weights1 -= lr * X_train_bias.T @ hiddenact_deltas



            # Calculate the accuracy and cross entropy for the training and validation set
            self.accuracies.append(accuracy(self.predict(X_train), t_train)) # removing the bias from X_train
            self.cross_entropies.append(self.cross_entropy(self.predict_probability(X_train), t_train))
            if X_val is not None:
                self.accuracies_val.append(accuracy(self.predict(X_val), t_val))
                self.cross_entropies_val.append(self.cross_entropy(self.predict_probability(X_val), t_val))
            
            
            # Check if we should stop
            if len(self.accuracies) > n_epochs_no_update:
                #if self.cross_entropies[-1]  + tol >= self.cross_entropies[-2]:
                if self.accuracies[-1] <= self.accuracies[-1-n_epochs_no_update] + tol:
                    temp_epochs_no_update += 1
                    if temp_epochs_no_update >= n_epochs_no_update:
                        run = False
                else:
                    temp_epochs_no_update = 0

        self.epochs = len(self.accuracies)
    
    def predict(self, X):
        """Predict the class for the members of X"""
        Z = self.add_bias(X, self.bias)
        forw = self.forward(Z)[1]
        score= forw[:, 0]
        return (score > 0.5)
    
    def predict_probability(self, X):
        """Predict the probability for the members of X"""
        Z = self.add_bias(X, self.bias)
        forw = self.forward(Z)[1]
        return forw[:, 0]
    

def accuracy(predicted, gold):
    return np.mean(predicted == gold)


def plot_decision_regions(X, t, clf=[], size=(10,8)):
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
    plt.show()


def standarize(X):
    """Standardize the dataset X with values between 0 and 1"""
    feature1 = X[:, 0]
    feature2 = X[:, 1]
    feature1_norm = (feature1-np.min(feature1))/(np.max(feature1)-np.min(feature1))
    feature2_norm = (feature2-np.min(feature2))/(np.max(feature2)-np.min(feature2))
    return np.column_stack((feature1_norm, feature2_norm))

# Generating the dataset
X, t_multi = make_blobs(n_samples=[400, 400, 400, 400, 400], centers=[[0,1],[4,2],[8,1],[2,0],[6,0]], 
                  n_features=2, random_state=2024, cluster_std=[1.0, 2.0, 1.0, 0.5, 0.5])

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

# New dataset (t2) with binary labels
t2_train = t_multi_train >= 3
t2_train = t2_train.astype('int')
t2_val = (t_multi_val >= 3).astype('int')
t2_test = (t_multi_test >= 3).astype('int')


if scale_x:
    scaler = StandardScaler()
    scaler.fit(X)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)


def linear_regression_example():
    cl = NumpyLinRegClass()
    if scale_x:
        cl.fit(X_train, t2_train, lr=1.2, epochs=10)
    else:
        cl.fit(X_train, t2_train, lr=0.07, epochs=200)
    
    print("Accuracy on the train set:", accuracy(cl.predict(X_train), t2_train))
    print("Accuracy on the validation set:", accuracy(cl.predict(X_val), t2_val))
    plot_decision_regions(X_train, t2_train, cl)
    plot_decision_regions(X_val, t2_val, cl)


def logistical_regression_example():
    cl = NumpyLogRegClass()
    if scale_x:
        cl.fit(X_train, t2_train, X_val, t2_val, 
           lr=5, n_epochs_no_update=5, tol=0.001)
    else:
        cl.fit(X_train, t2_train, X_val, t2_val, 
           lr=0.2, n_epochs_no_update=10, tol=0.001)
    print("Accuracy on the train set:", accuracy(cl.predict(X_train), t2_train))
    print("Accuracy on the validation set:", accuracy(cl.predict(X_val), t2_val))
    cl.show_stats()
    plot_decision_regions(X_train, t2_train, cl)


def multi_class_regression_example():
    cl = NumpyMultiLogRegClass()
    if scale_x:
        cl.fit(X_train, t_multi_train, lr=10, n_epochs_no_update=15, tol=0.001)
    else:
        cl.fit(X_train, t_multi_train, lr=0.35, n_epochs_no_update=200, tol=0.001)
    print("Accuracy on the train set:", accuracy(cl.predict(X_train), t_multi_train))
    print("Accuracy on the validation set:", accuracy(cl.predict(X_val), t_multi_val))
    plot_decision_regions(X_train, t_multi_train, cl)
    plot_decision_regions(X_val, t_multi_val, cl)


def multi_layer_neural_network_example():
    if scale_x:
        cl = MLPBinaryLinRegClass(dim_hidden=6)
        cl.fit(X_train, t2_train, X_val, t2_val, lr=0.002, n_epochs_no_update=100, tol=0.001)
    else:
        cl = MLPBinaryLinRegClass(dim_hidden=5)
        cl.fit(X_train, t2_train, X_val, t2_val, lr=0.0004, n_epochs_no_update=200, tol=0.001)
    
    print("Number of epochs:", cl.epochs)
    print("Accuracy on the train set:", accuracy(cl.predict(X_train), t2_train))
    print("Accuracy on the validation set:", accuracy(cl.predict(X_val), t2_val))
    # cl.show_stats()
    # plot_decision_regions(X_train, t2_train, cl)
    # plot_decision_regions(X_val, t2_val, cl)


def multi_layer_neural_network_10_example():
    accuracies_train = []
    accuracies_val = []
    if scale_x:
        for _ in range(10):
            cl = MLPBinaryLinRegClass(dim_hidden=6)
            cl.fit(X_train, t2_train, X_val, t2_val, lr=0.002, n_epochs_no_update=100, tol=0.001)
            accuracies_train.append(accuracy(cl.predict(X_train), t2_train))
            accuracies_val.append(accuracy(cl.predict(X_val), t2_val))
    else:
        for _ in range(10):
            cl = MLPBinaryLinRegClass(dim_hidden=5)
            cl.fit(X_train, t2_train, X_val, t2_val, lr=0.0004, n_epochs_no_update=200, tol=0.001)
            accuracies_train.append(accuracy(cl.predict(X_train), t2_train))
            accuracies_val.append(accuracy(cl.predict(X_val), t2_val))

    
    print(f"mean accuracy on the train set: {np.mean(accuracies_train):.2f}")
    print(f"std accuracy on the train set: {np.std(accuracies_train):.4f}")
    print(f"mean accuracy on the validation set: {np.mean(accuracies_val):.2f}")
    print(f"std accuracy on the validation set: {np.std(accuracies_val):.4f}")


def final_testing():
    """
    This is the final testing of the models.
    """
    # Should only be done to scaled data. As that gave best results.
    if not scale_x:
        print("You need to scale the data")
        return
    
    cl1 = NumpyLinRegClass()
    cl2 = NumpyLogRegClass()
    cl3 = MLPBinaryLinRegClass()

    cl1.fit(X_train, t2_train, lr=0.07, epochs=200)
    cl2.fit(X_train, t2_train, X_val, t2_val, lr=5, n_epochs_no_update=5, tol=0.001)
    cl3.fit(X_train, t2_train, X_val, t2_val, lr=0.002, n_epochs_no_update=10000, tol=0.001)

    accuracies = np.zeros([3, 3])

    accuracies[0,0] = (accuracy(cl1.predict(X_train), t2_train))
    accuracies[0,1] = (accuracy(cl2.predict(X_train), t2_train))
    accuracies[0,2] = (accuracy(cl3.predict(X_train), t2_train))

    accuracies[1,0] = (accuracy(cl1.predict(X_val), t2_val))
    accuracies[1,1] = (accuracy(cl2.predict(X_val), t2_val))
    accuracies[1,2] = (accuracy(cl3.predict(X_val), t2_val))

    accuracies[2,0] = (accuracy(cl1.predict(X_test), t2_test))
    accuracies[2,1] = (accuracy(cl2.predict(X_test), t2_test))
    accuracies[2,2] = (accuracy(cl3.predict(X_test), t2_test))

    print(accuracies)

    

#linear_regression_example()
#logistical_regression_example()
#multi_class_regression_example()
#multi_layer_neural_network_example()
#multi_layer_neural_network_10_example()
final_testing()


