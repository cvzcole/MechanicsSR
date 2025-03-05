import numpy as np
from numpy import zeros, ones, hstack, asarray
import itertools

def basis_vector(n, i):
    x = zeros(n, dtype=int)
    x[i] = 1
    return x

def as_tall(x):
    return x.reshape(x.shape + (1,))

def predict(xs, params, powers):
    return sum(p * (xs ** power).prod(axis=1) for p, power in zip(params, powers))

def compute_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def multipolyfit(xs, y, deg, learning_rate=0.001, n_iterations=10000):
    y = asarray(y).squeeze()
    rows = y.shape[0]
    xs = asarray(xs)
    try:
        num_covariates = xs.shape[1]
    except:
        num_covariates = 1
        xs = np.reshape(xs, (len(xs), 1))

    xs = hstack((ones((xs.shape[0], 1), dtype=xs.dtype), xs))
    
    generators = [basis_vector(num_covariates + 1, i) for i in range(num_covariates + 1)]
    
    # All combinations of degrees
    powers = list(map(sum, itertools.combinations_with_replacement(generators, deg)))

    # Initialize parameters randomly
    params = np.random.randn(len(powers))

    # Gradient descent
    for _ in range(n_iterations):
        y_pred = predict(xs, params, powers)
        errors = y_pred - y
        gradients = [np.mean(errors * (xs ** power).prod(axis=1)) for power in powers]
        params -= learning_rate * np.array(gradients)

    # Compute the final root mean square (rms) error
    rms = np.sqrt(compute_mse(y, predict(xs, params, powers)))

    return params, rms

def getBest(xs, y, max_deg, learning_rate=0.001, n_iterations=10000):
    results = []
    for i in range(0, max_deg + 1):
        results.append(multipolyfit(xs, y, i, learning_rate, n_iterations))
    
    # Get the parameters and error of the fit with the lowest rms error
    min_error_index = np.argmin([result[1] for result in results])
    params, error = results[min_error_index]
    deg = min_error_index
    print('params=', params)
    return params, error, deg
