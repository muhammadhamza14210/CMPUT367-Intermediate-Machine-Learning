import torch.nn.functional as F
from utils import add_bias_dim
import torch


class TwoLayerSgdOptimizer():
    def __init__(self, learning_rate=1e-3):
        self.learning_rate = learning_rate


    def update(self, model, X, t):
        '''Given the model, input X, and target t, update the model with
        stochastic gradient descent.
        Returns:
            The loss for this batch.
        '''
        outputs = model.forward(X)

        # Convert the targets to one-hot encoding
        t_one_hot = F.one_hot(t.squeeze().to(torch.int64), num_classes=10)
        
        # Compute the gradient of the output layer
        grad_output = outputs - t_one_hot
        grad_wrt_W2 = model.a1.t().mm(grad_output)
        grad_wrt_b2 = grad_output.sum(0)
        
        # Backpropagate through the network
        grad_hidden = grad_output.mm(model.layer2['W'].t()) * (model.a1 * (1 - model.a1))
        grad_wrt_W1 = X.t().mm(grad_hidden)
        grad_wrt_b1 = grad_hidden.sum(0)
        
        # Update the parameters for both layers
        with torch.no_grad():
            model.layer2['W'] -= self.learning_rate * grad_wrt_W2
            model.layer2['b'] -= self.learning_rate * grad_wrt_b2
            model.layer1['W'] -= self.learning_rate * grad_wrt_W1
            model.layer1['b'] -= self.learning_rate * grad_wrt_b1

        # Compute the loss using the outputs from the forward pass and the original targets
        loss = F.cross_entropy(outputs, t.squeeze())
        return loss.item()