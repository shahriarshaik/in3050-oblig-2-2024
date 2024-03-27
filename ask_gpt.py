from sklearn.metrics import accuracy_score

class NumpyLogRegClass(NumpyClassifier):
    def __init__(self, bias=-1): 
        self.bias = bias
        self.losses = []  # Store losses for each epoch
        self.accuracies = []  # Store accuracies for each epoch
        self.val_losses = []  # Store validation losses for each epoch
        self.val_accuracies = []  # Store validation accuracies for each epoch

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X_train, t_train, eta=0.1, epochs=10, X_val=None, t_val=None):
        if self.bias:
            X_train = add_bias(X_train, self.bias)
            if X_val is not None:
                X_val = add_bias(X_val, self.bias)
            
        (N, M) = X_train.shape
        
        self.weights = weights = np.zeros(M)
        
        for epoch in range(epochs):
            # Calculate predicted probabilities for training set
            y_pred = self.sigmoid(X_train @ weights)
            
            # Calculate binary cross-entropy loss for training set
            loss = -np.mean(t_train * np.log(y_pred) + (1 - t_train) * np.log(1 - y_pred))
            self.losses.append(loss)
            
            # Calculate accuracy for training set
            y_pred_binary = y_pred > 0.5
            accuracy = accuracy_score(t_train, y_pred_binary)
            self.accuracies.append(accuracy)
            
            # Calculate predicted probabilities for validation set
            if X_val is not None and t_val is not None:
                y_val_pred = self.sigmoid(X_val @ weights)
                
                # Calculate binary cross-entropy loss for validation set
                val_loss = -np.mean(t_val * np.log(y_val_pred) + (1 - t_val) * np.log(1 - y_val_pred))
                self.val_losses.append(val_loss)
                
                # Calculate accuracy for validation set
                y_val_pred_binary = y_val_pred > 0.5
                val_accuracy = accuracy_score(t_val, y_val_pred_binary)
                self.val_accuracies.append(val_accuracy)
            
            # Update weights using gradient descent
            weights -= eta / N * X_train.T @ (y_pred - t_train)
    
    def predict(self, X, threshold=0.5):
        if self.bias:
            X = add_bias(X, self.bias)
        y_pred = self.sigmoid(X @ self.weights)
        return y_pred > threshold
    
    def predict_probability(self, X):
        if self.bias:
            X = add_bias(X, self.bias)
        return self.sigmoid(X @ self.weights)

