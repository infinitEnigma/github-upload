from pathlib import Path
import requests

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

# This dataset is in numpy array format, and has been stored using pickle, 
# a python-specific format for serializing data.

import pickle
import gzip

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")


# Each image is 28 x 28, we need to reshape it to 2d first.
from matplotlib import pyplot
import numpy as np

pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
print(x_train.shape)


# PyTorch uses torch.tensor, rather than numpy arrays, 
# so we need to convert our data.
import torch

x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape
x_train, x_train.shape, y_train.min(), y_train.max()
print(x_train, y_train)
print(x_train.shape)
print(y_train.min(), y_train.max())


# Neural net from scratch (no torch.nn)
# We are initializing the weights here with 
# Xavier initialisation (by multiplying with 1/sqrt(n)).
import math

weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)

# write a plain matrix multiplication and 
# broadcasted addition to create a simple linear model. 
# we also need an activation function
def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

def model(xb):
    return log_softmax(xb @ weights + bias)



# We will call our function on one batch of data (in this case, 64 images). 
# This is one forward pass.
# Note that our predictions won’t be any better than random at this stage, 
# since we start with random weights.
bs = 64  # batch size

xb = x_train[0:bs]  # a mini-batch from x
preds = model(xb)  # predictions
preds[0], preds.shape
print(preds[0], preds.shape)


# Let’s implement negative log-likelihood to use as the loss function
def nll(input, target):
    return -input[range(target.shape[0]), target].mean()

loss_func = nll

# Let’s check our loss with our random model, 
# so we can see if we improve after a backprop pass later.
yb = y_train[0:bs]
print(loss_func(preds, yb))


# implement a function to calculate the accuracy of our model
# if the index with the largest value matches the target value, 
# then the prediction was correct
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

print(accuracy(preds, yb))


# We can now run a training loop. For each iteration, we will:
# *   select a mini-batch of data (of size bs)
# *   use the model to make predictions
# *   calculate the loss
# *   loss.backward() updates the gradients of the model, in this case, weights and bias.
from IPython.core.debugger import set_trace

lr = 0.5  # learning rate
epochs = 2  # how many epochs to train for

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        # Uncomment set_trace() below to try it out
        #         set_trace()
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()

print(loss_func(model(xb), yb), accuracy(model(xb), yb))


# Using torch.nn.functional

# We will now refactor our code, so that it does the same thing as before, 
# only we’ll start taking advantage of 
# PyTorch’s nn classes to make it more concise and flexible.
import torch.nn.functional as F

loss_func = F.cross_entropy

def model(xb):
    return xb @ weights + bias
    # Note that we no longer call log_softmax
    # return log_softmax(xb @ weights + bias)
# confirm that it works the same
print(loss_func(model(xb), yb), accuracy(model(xb), yb))


# Refactor using nn.Module

# In this case, we want to create a class that holds our 
# weights, bias, and method for the forward step
from torch import nn

class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, xb):
        return xb @ self.weights + self.bias

# Since we’re now using an object instead of just using a function, 
# we first have to instantiate our model:
model = Mnist_Logistic()

print(loss_func(model(xb), yb))

# Now we can take advantage of model.parameters() and model.zero_grad()
# with torch.no_grad():
#     for p in model.parameters(): p -= p.grad * lr
#     model.zero_grad()

# We’ll wrap our little training loop in a fit function 
# so we can run it again later.
def fit():
    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= p.grad * lr
                model.zero_grad()

fit()

# Let’s double-check that our loss has gone down:
print(loss_func(model(xb), yb))


# Refactor using nn.Linear

# Instead of manually defining and initializing self.weights and self.bias
# use the Pytorch class nn.Linear for a linear layer
class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        return self.lin(xb)

# instantiate our model and calculate the loss in the same way as before:
model = Mnist_Logistic()
print(loss_func(model(xb), yb))

# We are still able to use our same fit method as before.
fit()

print(loss_func(model(xb), yb))


# Refactor using optim

# This will let us replace our previous manually coded optimization step:
# ```
# with torch.no_grad():
#    for p in model.parameters(): p -= p.grad * lr
#    model.zero_grad()
# ```
# with: `opt.step()` and `opt.zero_grad()`
from torch import optim

# We’ll define a little function to create our model and optimizer 
# so we can reuse it in the future.

def get_model():
    model = Mnist_Logistic()
    return model, optim.SGD(model.parameters(), lr=lr)

model, opt = get_model()
print(loss_func(model(xb), yb))

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))


# Refactor using Dataset

# A Dataset can be anything that has a __len__ function 
#  and a __getitem__ function as a way of indexing into it.

# example of creating a custom 
# FacialLandmarkDataset class as a subclass of Dataset
from torch.utils.data import TensorDataset

# Both x_train and y_train can be combined in a single TensorDataset, 
# which will be easier to iterate over and slice.
train_ds = TensorDataset(x_train, y_train)

model, opt = get_model()

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        xb, yb = train_ds[i * bs: i * bs + bs]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))



# Refactor using DataLoader

# Pytorch’s DataLoader is responsible for managing batches. 
# You can create a DataLoader from any Dataset.

from torch.utils.data import DataLoader

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs)

model, opt = get_model()

for epoch in range(epochs):
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))
# our training loop is now dramatically smaller and easier to understand



# Add validation

# We’ll use a batch size for the validation set 
# that is twice as large as that for the training set.
train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)

# We will calculate and print the validation loss at the end of each epoch.
model, opt = get_model()

for epoch in range(epochs):
    model.train()
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

    model.eval()
    with torch.no_grad():
        valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)

    print(epoch, valid_loss / len(valid_dl))



# Create fit() and get_data()

# We pass an optimizer in for the training set, and use it to perform backprop. 
# For the validation set, we don’t pass an optimizer, 
# so the method doesn’t perform backprop.
def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

# fit runs the necessary operations to train our model 
# and compute the training and validation losses for each epoch.
import numpy as np

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)

# get_data returns dataloaders for the training and validation sets.
def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )

# Now, our whole process of obtaining the data loaders and fitting the model 
# can be run in 3 lines of code:
train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
model, opt = get_model()
fit(epochs, model, loss_func, opt, train_dl, valid_dl)
# You can use these basic 3 lines of code to train a wide variety of models. 
# Let’s see if we can use them to train a convolutional neural network (CNN)!



# Switch to CNN

# We are now going to build our neural network with three convolutional layers. 
# Because none of the functions in the previous section 
# assume anything about the model form, 
# we’ll be able to use them to train a CNN without any modification.

# We will use Pytorch’s predefined Conv2d class as our convolutional layer
# Each convolution is followed by a ReLU. At the end, we perform an average pooling
class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))

lr = 0.1

# Momentum is a variation on stochastic gradient descent 
# that takes previous updates into account as well 
# and generally leads to faster training.
model = Mnist_CNN()
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

fit(epochs, model, loss_func, opt, train_dl, valid_dl)



# nn.Sequential

# PyTorch doesn’t have a view layer, and we need to create one for our network. 
# Lambda will create a layer that we can then use 
# when defining a network with Sequential.
class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def preprocess(x):
    return x.view(-1, 1, 28, 28)

# The model created with Sequential is simply:
model = nn.Sequential(
    Lambda(preprocess),
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AvgPool2d(4),
    Lambda(lambda x: x.view(x.size(0), -1)),
)

opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

fit(epochs, model, loss_func, opt, train_dl, valid_dl)



# Wrapping DataLoader

# Our CNN is fairly concise, but it only works with MNIST
# It assumes the input is a 28*28 long vector and 
# that the final CNN grid size is 4*4
# Let’s get rid of these two assumptions

# remove the initial Lambda layer but moving the data preprocessing into a generator:
def preprocess(x, y):
    return x.view(-1, 1, 28, 28), y


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))

train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
train_dl = WrappedDataLoader(train_dl, preprocess)
valid_dl = WrappedDataLoader(valid_dl, preprocess)

# Next, we can replace nn.AvgPool2d with nn.AdaptiveAvgPool2d
model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    Lambda(lambda x: x.view(x.size(0), -1)),
)

opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# Let's try it out:
fit(epochs, model, loss_func, opt, train_dl, valid_dl)





# Using GPU

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#Let’s update preprocess to move batches to the GPU:
def preprocess(x, y):
    return x.view(-1, 1, 28, 28).to(dev), y.to(dev)


train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
train_dl = WrappedDataLoader(train_dl, preprocess)
valid_dl = WrappedDataLoader(valid_dl, preprocess)

# Finally, we can move our model to the GPU.
model.to(dev)
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

fit(epochs, model, loss_func, opt, train_dl, valid_dl)

