import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import copy

np.set_printoptions(precision=2)

file_path_train = "train.csv"

train_df = pd.read_csv(file_path_train)
print(train_df.head(5))
print(f"\nShape: {train_df.shape}")

# Drop the ID column since it is not needed
train_df = train_df.drop(axis=1, labels="Id")
print(train_df.head(5))

print(train_df['SalePrice'].describe())
# Distribution of sale prices

for col in train_df.columns:
    if pd.api.types.is_numeric_dtype(train_df[col]):
        train_df[col] = train_df[col].fillna(train_df[col].mean())
#Fill all of the numeric NA's with the mean of the whole column

le = LabelEncoder()

for col in train_df.columns:
    if not pd.api.types.is_numeric_dtype(train_df[col]):
        train_df[col] = le.fit_transform(train_df[col])

print(train_df.head(5))

X = train_df.drop(axis=1, labels="SalePrice")
y = train_df["SalePrice"]

def find_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost += (f_wb_i - y[i])**2
    cost = cost / (2 * m)
    return cost

def find_gradient(X, y, w, b):
    m,n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0

    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] += (err * X[i, j])
        dj_db += err
    dj_dw /= m
    dj_db /= m

    return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_db, dj_dw = find_gradient(X, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {find_cost(X, y, w, b)}")

    return w, b

# The code above contains a function to find the cost given a vector of inputs X, targets y, vector of attributes w, 
# and attribute b. Also contains a function to find the gradient/rate of change to be used for vector w and scalar b
# The last funciton uses the two funcitons above to implement gradient descent for multi variable linear regression to
# find values for w and b. This much code is not required, simply used to show gradient descent for multi variable linear
# regression in more detail.