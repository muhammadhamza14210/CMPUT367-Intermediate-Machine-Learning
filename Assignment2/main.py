# -*- coding: utf-8 -*-
from utils import readMNISTdata
from network import TwoLayerClassifier
from optimizer import TwoLayerSgdOptimizer
from cyclic_lr import CyclicLRScheduler
import matplotlib.pyplot as plt
import seaborn as sns


MNIST_PATH = "../MNIST/"

X_train, t_train, X_val, t_val, X_test, t_test = readMNISTdata(path=MNIST_PATH)

print("Data shape:")
print(X_train.shape, t_train.shape, X_val.shape, t_val.shape, X_test.shape, t_test.shape)

base_lr = 0.01
max_lr = 0.01
step_size_up = 250

classifier = TwoLayerClassifier()
optimizer = TwoLayerSgdOptimizer()
scheduler = CyclicLRScheduler(base_lr, max_lr, step_size_up)
best_epoch, best_acc, train_losses, valid_accs = classifier.train(optimizer, X_train, t_train, X_val, t_val, max_epoch=50, scheduler=scheduler)
_, accuracy, _ = classifier.predict(X_test, t_test)

print('Best epoch:', best_epoch)
print('Validation acc:', best_acc)

sns.lineplot(x=range(len(train_losses)), y=train_losses)
plt.ylabel("Cross Entropy Loss")
plt.xlabel("Epoch")
plt.title("Training Loss")
plt.savefig('train_loss5.png')
plt.clf()
sns.lineplot(x=range(len(valid_accs)), y=valid_accs)
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.title("Validation Accuracy")
plt.savefig('valid_acc5.png')
plt.clf()

print('test acc:', accuracy)