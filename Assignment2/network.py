import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import add_bias_dim
from tqdm import tqdm
import numpy as np
from cyclic_lr import CyclicLRScheduler

class TwoLayerClassifier:
    def __init__(self, input_dim=28*28, hidden_dim=64, output_dim=10, batch_size=100):
        # Attributes need to be aligned with the skeleton code provided in the train method
        self.layer1 = {}
        self.layer2 = {}
        
        # Initialize weights and biases for layer1 and layer2
        self.layer1['W'] = torch.randn(input_dim, hidden_dim) * np.sqrt(2. / input_dim)
        self.layer1['b'] = torch.zeros(hidden_dim)
        self.layer2['W'] = torch.randn(hidden_dim, output_dim) * np.sqrt(1. / hidden_dim)
        self.layer2['b'] = torch.zeros(output_dim)
        
        self.batch_size = batch_size
    def forward(self, X):
        self.z1 = torch.matmul(X, self.layer1['W']) + self.layer1['b']
        self.a1 = torch.sigmoid(self.z1)  
        z2 = torch.matmul(self.a1, self.layer2['W']) + self.layer2['b']
        return F.softmax(z2, dim=1)

    def backward(self, X, output, target, learning_rate):
        # Convert targets to one-hot encoding if they are not already
        if target.dim() == 1 or target.size(1) == 1:
            target_one_hot = F.one_hot(target.view(-1), num_classes=output.size(1)).float()
        else:
            target_one_hot = target
        
        # Compute the error/loss gradients
        output_error = output - target_one_hot
        dW2 = torch.matmul(self.a1.t(), output_error) / X.size(0)  # Average over batch size
        db2 = output_error.sum(0) / X.size(0)
        
        # Backpropagate through the second layer to the activations of the first layer
        delta2 = torch.matmul(output_error, self.layer2['W'].t()) * self.a1 * (1 - self.a1)
        dW1 = torch.matmul(X.t(), delta2) / X.size(0)
        db1 = delta2.sum(0) / X.size(0)
        
        # Update weights and biases
        self.layer1['W'] -= learning_rate * dW1
        self.layer1['b'] -= learning_rate * db1
        self.layer2['W'] -= learning_rate * dW2
        self.layer2['b'] -= learning_rate * db2
        
        # Calculate the cross-entropy loss
        loss = F.cross_entropy(output, target.view(-1), reduction='mean')
        return loss.item()
    
    def train(self, optimizer, X_train, t_train, X_val, t_val, max_epoch=50, scheduler=None):
        N_train, N_feature = X_train.shape

        best_epoch, best_acc = -1, -1
        best_layer1_W, best_layer1_b, best_layer2_W, best_layer2_b = None, None, None, None
        train_losses, valid_accs = [], []

        for epoch in tqdm(range(max_epoch)):
            if scheduler:
                scheduler.update_optimizer_lr(optimizer)
            indices = np.random.permutation(N_train)
            X = X_train[indices]
            t = t_train[indices]

            loss_sum = 0
            for batch_start in range(0, N_train, self.batch_size):
                X_batch = X[batch_start:batch_start + self.batch_size]
                t_batch = t[batch_start:batch_start + self.batch_size].reshape(-1)
                loss_sum += optimizer.update(self, X_batch, t_batch)
            loss_avg = loss_sum / N_train
            train_losses.append(loss_avg)

            _, valid_acc, _ = self.predict(X_val, t_val)
            valid_accs.append(valid_acc)

            if valid_acc > best_acc:
                best_epoch, best_acc = epoch, valid_acc
                # Save the current best weights and biases
                best_layer1_W, best_layer1_b = self.layer1['W'].clone(), self.layer1['b'].clone()
                best_layer2_W, best_layer2_b = self.layer2['W'].clone(), self.layer2['b'].clone()

        # Restore the best weights and biases after training
        self.layer1['W'], self.layer1['b'] = best_layer1_W, best_layer1_b
        self.layer2['W'], self.layer2['b'] = best_layer2_W, best_layer2_b

        return best_epoch, best_acc, train_losses, valid_accs


    def predict(self, X, t=None):
        acc = None

        with torch.no_grad():
            logits = self.forward(X)
            y_pred = logits.argmax(dim=1)
        
        if t is not None:
            t = t.squeeze()  # Ensure t is properly shaped
            acc = (y_pred == t).float().mean().item()

        return y_pred, acc, logits
