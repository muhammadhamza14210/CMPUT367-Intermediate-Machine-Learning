# -*- coding: utf-8 -*-
from utils import readMNISTdata
from question3 import SoftmaxRegularization
import matplotlib.pyplot as plt
import seaborn as sns

import os

MNIST_PATH = "../MNIST/"


X_train, t_train, X_val, t_val, X_test, t_test = readMNISTdata(path=MNIST_PATH)

print("Data shape:")
print(X_train.shape, t_train.shape, X_val.shape, t_val.shape, X_test.shape, t_test.shape)


regressor = SoftmaxRegularization(N_class=10, learning_rate=0.01)
best_epoch, best_acc, train_losses, valid_accs = regressor.train(X_train, t_train, X_val, t_val)
_, accuracy, _ = regressor.predict(X_test, t_test)

print('Best epoch:', best_epoch)
print('Validation acc:', best_acc)


sns.lineplot(x=range(len(train_losses)), y=train_losses)
plt.ylabel("Cross Entropy Loss")
plt.xlabel("Epoch")
plt.title("Training Loss")
plt.savefig('train_lossq3-3.png')
plt.clf()
sns.lineplot(x=range(len(valid_accs)), y=valid_accs)
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.title("Validation Accuracy")
plt.savefig('valid_accq3-3.png')
plt.clf()

print('test acc:', accuracy)