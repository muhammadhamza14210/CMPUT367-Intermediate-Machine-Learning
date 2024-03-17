import numpy as np
from tqdm import tqdm

class SoftmaxRegularization:
    def __init__(self, N_class, learning_rate=0.1, batch_size=100, reg_lambda=0.1):
        self.N_class = N_class
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.reg_lambda = reg_lambda  # Regularization strength
        self.W = None
        self.b = None

    def _batch_update(self, X_batch, t_batch):
        grad_W = np.zeros_like(self.W)
        grad_b = np.zeros(self.N_class)
        batch_loss = 0

        for i in range(X_batch.shape[0]):
            z_i = np.dot(self.W.T, X_batch[i]) + self.b
            z_i_max = np.max(z_i)
            z_i -= z_i_max

            exp_z_i = np.exp(z_i)
            y_i = exp_z_i / np.sum(exp_z_i)

            t_one_hot_i = np.eye(self.N_class)[t_batch[i]]

            loss_i = -np.sum(t_one_hot_i * np.log(y_i + 1e-7))
            batch_loss += loss_i

            grad_W_i = np.outer(X_batch[i], (y_i - t_one_hot_i))
            grad_b_i = y_i - t_one_hot_i

            grad_W += grad_W_i
            grad_b += grad_b_i

        grad_W /= X_batch.shape[0]
        grad_b /= X_batch.shape[0]
        batch_loss /= X_batch.shape[0]

        # Include L2 regularization in the loss and gradient
        batch_loss += 0.5 * self.reg_lambda * np.sum(self.W ** 2)
        grad_W += self.reg_lambda * self.W

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