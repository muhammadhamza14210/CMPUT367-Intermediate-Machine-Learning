import numpy as np
from tqdm import tqdm

class SoftmaxReg:
    def __init__(self, N_class, learning_rate=0.1, batch_size=100):
        self.N_class = N_class
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.b = None


    def _batch_update(self, X_batch, t_batch):
    # Initialize gradients and loss
        grad_W = np.zeros_like(self.W)
        grad_b = np.zeros(self.N_class)
        batch_loss = 0

        # Iterate over each data point in the batch
        for i in range(X_batch.shape[0]):
            # Compute the logits for a single data point
            z_i = np.dot(self.W.T, X_batch[i]) + self.b
            # Normalize the logits to avoid numerical overflow
            z_i_max = np.max(z_i)
            z_i -= z_i_max
            
            # Compute the softmax probabilities
            exp_z_i = np.exp(z_i)
            y_i = exp_z_i / np.sum(exp_z_i)
            
            # Convert the label to a one-hot vector
            t_one_hot_i = np.eye(self.N_class)[t_batch[i]]

            # Compute the loss for the single data point
            loss_i = -np.sum(t_one_hot_i * np.log(y_i + 1e-7))  # Small constant for numerical stability
            batch_loss += loss_i

            # Compute the gradient for the single data point
            grad_W_i = np.outer(X_batch[i], (y_i - t_one_hot_i))
            grad_b_i = y_i - t_one_hot_i
            
            # Sum the gradients
            grad_W += grad_W_i
            grad_b += grad_b_i

        # Average the gradients and the loss
        grad_W /= X_batch.shape[0]
        grad_b /= X_batch.shape[0]
        batch_loss /= X_batch.shape[0]

        # Update the weights and bias
        self.W -= self.learning_rate * grad_W
        self.b -= self.learning_rate * grad_b

        return batch_loss




    def train(self, X_train, t_train, X_val, t_val, max_epoch=50):
        N_train, N_feature = X_train.shape


        self.W = np.zeros((N_feature, self.N_class))
        self.b = np.zeros(self.N_class)
        best_epoch, best_acc, best_W = -1, -1, None
        train_losses, valid_accs = [], []

        for epoch in tqdm(range(max_epoch)):
            indices = np.random.permutation(N_train)
            X = X_train[indices]
            t = t_train[indices]

            loss_sum = 0
            for batch_start in range(0, N_train, self.batch_size):
                X_batch = X[batch_start:batch_start + self.batch_size]
                t_batch = t[batch_start:batch_start + self.batch_size].reshape(-1)
                loss_sum += self._batch_update(X_batch, t_batch)
            loss_avg = loss_sum/N_train
            train_losses.append(loss_avg)

            _, valid_acc, _ = self.predict(X_val, t_val)
            valid_accs.append(valid_acc)

            if valid_acc > best_acc:
                best_epoch, best_acc, best_W = epoch, valid_acc, self.W

        self.W = best_W

        return best_epoch, best_acc, train_losses, valid_accs


    def predict(self, X, t=None):
    # Compute the logits without adding a column for the bias
        z = np.dot(X, self.W) + self.b  # Now the shapes should align correctly
        
        # Normalize the logits to avoid numerical overflow
        z_max = np.max(z, axis=1, keepdims=True)
        z_stable = z - z_max
        
        # Apply the softmax function
        exp_z = np.exp(z_stable)
        y = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        
        # The predicted class is the one with the highest probability
        t_hat = np.argmax(y, axis=1)
        
        # If true labels are provided, calculate the accuracy
        acc = None
        if t is not None:
            # Check if t needs to be flattened
            if t.ndim > 1:
                t = t.ravel()
            acc = np.mean(t_hat == t)

        return t_hat, acc, y

