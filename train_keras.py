import pickle
import numpy
numpy.random.seed(123)
from model import *
import sys
sys.setrecursionlimit(10000)

train_ratio = 0.9

f = open('feature_train_data.pickle', 'rb')
(X, y) = pickle.load(f)

num_records = len(X)

train_size = int(train_ratio * num_records)

X_train = X[:train_size]
X_val = X[train_size:]
y_train = y[:train_size]
y_val = y[train_size:]


model = NN_with_EntityEmbedding(X_train, y_train, X_val, y_val)


def evaluate_models(model, X, y):
    assert(min(y) > 0)
    guessed_sales = model.guess(X)
    mean_sales = guessed_sales.mean(axis=0)
    relative_err = numpy.absolute((y - mean_sales) / y)
    result = numpy.sum(relative_err) / len(y)
    return result

print("Evaluate combined model...")
print("Training error...")
r_train = evaluate_models(model, X_train, y_train)
print(r_train)

print("Validation error...")
r_val = evaluate_models(model, X_val, y_val)
print(r_val)

