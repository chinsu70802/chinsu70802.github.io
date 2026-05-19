"""
VISUALIZING BIAS-VARIANCE RELATION USING THE EXERCISE ON ESTIMATING CONDITIONAL MEAN ESTIMATE OF BERNOULLI HIDDEN VARIABLE UNDER ADDITIVE GAUSSIAN NOISE
"""

import matplotlib.pyplot as plt
import numpy as np

test_dataset_size = 1000
n_datasets = 50                                                             # Number of different trials/datasets to obtain different estimators
n_entries_per_dataset = 100
std_dev_noise = 1                                                           # Std Dev of noise under consideration
degree_polynomial_estimators = [1, 3, 5, 7]                                 # Using odd degree polynomial estimators as tanh is an odd function
x_test = np.random.choice([-1,+1], size=test_dataset_size)
n_test = np.random.normal(0, std_dev_noise, size=test_dataset_size)
y_test = x_test + n_test
x_hat_theoretical = np.tanh(y_test/(std_dev_noise**2))                      # Theoretical optimal estimate (conditional mean estimate) on the test data

train_datasets = []
for i in range(n_datasets):                                                 # Generating different training datasets from the same distribution for obtaining different estimators 
    x = np.random.choice([-1,+1], size = n_entries_per_dataset)
    n = np.random.normal(0, std_dev_noise, size = n_entries_per_dataset)
    y = x + n
    train_datasets.append(np.column_stack((np.ones(y.shape[0]),y,x)))

theoretical_optimal_mmse = np.mean((x_test - x_hat_theoretical)**2)         # The first term in the bias-variance relation

bias_across_estimators = []
variance_across_estimators = []
mse_across_estimators = []
weight_variance_across_estimators = []
residues_across_estimators = []
for i in degree_polynomial_estimators:
    weights_across_datasets = []
    mse_across_datasets = []
    if i == 1:
        for j in range(n_datasets):
            cols = train_datasets[j]
            X = cols[:, :-1]
            y = cols[:, -1]
            lmse_weights_poly_estimator = np.linalg.pinv(X) @ y             # Using the LMSE solution to get the weights of the polynomial estimator
            weights_across_datasets.append(lmse_weights_poly_estimator)
    else:        
        for j in range(n_datasets):
            y_new = np.array([train_datasets[j][:,1]**m for m in range(2,i+1)]).T
            cols = np.column_stack((train_datasets[j][:,:-1], y_new, train_datasets[j][:, -1]))
            X = cols[:, :-1]
            y = cols[:, -1]
            lmse_weights_poly_estimator = np.linalg.pinv(X) @ y
            weights_across_datasets.append(lmse_weights_poly_estimator)
    average_estimator_across_datasets = np.mean(np.array(weights_across_datasets), axis=0)      
    y_new = np.array([y_test**m for m in range(i+1)])
    mean_x_hat_estimator = y_new.squeeze().T @ average_estimator_across_datasets  # This approximates the expectation of the estimators (E[c_D(y)])
    bias = np.mean((x_hat_theoretical - mean_x_hat_estimator)**2)
    bias_across_estimators.append(bias)
    weights_across_datasets = np.array(weights_across_datasets).squeeze()
    weight_variance_across_estimators.append(np.mean(np.var(weights_across_datasets, axis=0)))
    estimates_across_datasets = y_new.T @ weights_across_datasets.T
    variance = np.mean(np.var(estimates_across_datasets, axis = 1))
    variance_across_estimators.append(variance)
    mse = np.mean((estimates_across_datasets - x_test[:, None])**2)                # MSE reported here is the sum total of the theoretical optimal MSE, bias and variance
    mse_across_estimators.append(mse)
    residue = x_test[:, None] - estimates_across_datasets
    residues_across_estimators.append(residue) 
    
bias_across_estimators = np.array(bias_across_estimators)
variance_across_estimators = np.array(variance_across_estimators)
mse_across_estimators = np.array(mse_across_estimators)
weight_variance_across_estimators = np.array(weight_variance_across_estimators)
plt.xticks(degree_polynomial_estimators)
x = degree_polynomial_estimators
plt.figure()
plt.plot(x, bias_across_estimators.squeeze(), label='Bias as computed using a test set', marker='o')
plt.plot(x, variance_across_estimators.squeeze(), label='Variance as computed using a test set', marker='s')
plt.plot(x, mse_across_estimators.squeeze(), label='MSE as computed using a test set', marker='s')
plt.plot(x, np.array([theoretical_optimal_mmse,]*len(x)).squeeze(), label=f'Theoretical Optimal MMSE : {theoretical_optimal_mmse}', marker='*')
plt.xlabel('Order of estimator')
plt.ylabel('Bias-Variance relation component values')
plt.legend()

plt.figure()
plt.xticks(degree_polynomial_estimators)
plt.plot(x, weight_variance_across_estimators.squeeze(), label='Variance of learned weights for different estimators', marker='s')
plt.xlabel('Order of estimator')
plt.ylabel('Variance of learned weights')
plt.legend()

plt.figure()
for idx, order in enumerate(degree_polynomial_estimators):
    residue_vals = residues_across_estimators[idx].flatten()
    y_coords = np.full(residue_vals.shape, order)
    plt.scatter(residue_vals, y_coords, s=5)
plt.yticks(degree_polynomial_estimators)
plt.xlabel('real axis')
plt.ylabel('order')
plt.title('scatter diagrams of estimation errors')

plt.show()