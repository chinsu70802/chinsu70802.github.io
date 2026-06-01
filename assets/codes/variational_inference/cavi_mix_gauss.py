"""
Coordinate-Ascent Variational Inference (CAVI) for Bayesian Mixture of Gaussians
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def compute_elbo(x,variational_means,variational_var,variational_cat,var_mu): # Function to compute the ELBO for a given variational density (Courtesy: ChatGPT)
    N, K = variational_cat.shape
    elbo = 0.0
    elbo += np.sum(
        -0.5 * np.log(2 * np.pi * var_mu)
        -0.5 * (
            (variational_means**2 + variational_var)
            / var_mu
        )
    )
    elbo += np.sum(
        variational_cat * np.log(1 / K)
    )
    for k in range(K):

        expected_sq_error = (
            (x - variational_means[k])**2
            + variational_var[k]
        )

        elbo += np.sum(
            variational_cat[:, k]
            * (
                -0.5 * np.log(2 * np.pi)
                -0.5 * expected_sq_error
            )
        )
    elbo -= np.sum(
        variational_cat
        * np.log(variational_cat + 1e-12)
    )
    elbo += np.sum(
        0.5 * np.log(
            2 * np.pi * np.e * variational_var
        )
    )
    return elbo

np.random.seed(42)

K = 2           # Number of components
var_mu = 4      # The variance hyperparameter in the distribution from which means are drawn from
mu_params = np.random.normal(0, np.sqrt(var_mu), size=K)  # List of K means sampled from normal distribution with mean 0 and variance var_mu

N = 1000        # Number of observations
categories = [i for i in range(K)] # Index of the cluster means
n_categories = np.random.choice(categories, size=N) # Sampling N categories for N observations
x = np.array([np.random.normal(mu_params[i], 1) for i in n_categories]) # Sampling observations from the distribution as in the generative model

variational_means = np.random.normal(0, 1, size=K) # Randomly initializing the variational means for the mu parameter in the generative model
variational_var = np.ones(K)   # Randomly initializing the variational variances for the mu parameter in the generative model
variational_cat = np.random.dirichlet(alpha=np.ones(K), size=N) # Randomly initializing the variational categorical probabilities for the c parameter in the generative model

elbo_prev = -np.inf
elbo = compute_elbo(x, variational_means, variational_var, variational_cat, var_mu)

max_iters = 1000
iters = 0

elbo_history = []

while ((abs(elbo - elbo_prev) >= 1e-3) and (iters < max_iters)): # Coordinate Ascent Variational Inference
    phi_i_list = []

    for i in range(N):
        phi_i_logit = variational_means*x[i] - ((variational_var + (variational_means)**2)/2)
        phi_i_logit -= np.max(phi_i_logit)
        phi_i = np.exp(phi_i_logit)
        phi_i /= phi_i.sum()
        phi_i_list.append(phi_i)
    
    variational_cat = np.array(phi_i_list)
    
    for k in range(K):
        variational_means[k] = np.sum(variational_cat[:,k] * x)
        variational_means[k] /= ((1/var_mu) + variational_cat[:, k].sum()) 

        variational_var[k] = 1
        variational_var[k] /= ((1/var_mu) + variational_cat[:, k].sum()) 
    
    elbo_prev = elbo
    elbo = compute_elbo(x, variational_means, variational_var, variational_cat, var_mu)
    elbo_history.append(elbo)
    iters += 1

# Code for plotting stuff

plt.figure()
plt.plot(elbo_history)
plt.title('ELBO across iterations during CAVI')
plt.xlabel('Iteration')
plt.ylabel('ELBO')
plt.savefig('../../images/variational_inference/elbo_plot.png')

plt.figure()
x_range = np.linspace(x.min() - 1, x.max() + 1, 300)
plt.hist(x, bins=40, density=True, alpha=0.4, label='True Density')
mixture_curve = np.zeros_like(x_range)
for k in range(K):
    weight = variational_cat[:,k].mean()
    curve = weight * norm.pdf(x_range, loc=variational_means[k], scale=1)
    mixture_curve += curve
plt.plot(x_range, mixture_curve, 'k--', label='Full Mixture obtained from the variational factors')
plt.title('Data histogram vs fitted cluster components')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.savefig('../../images/variational_inference/obs_vs_variational.png')

plt.figure()
for k in range(K):              # Courtesy: Claude
    assigned = x[n_categories == k]
    n_k = len(assigned)
    
    true_post_var = 1 / (1/var_mu + n_k)
    true_post_mean = true_post_var * np.sum(assigned)
    mu_range = np.linspace(true_post_mean - 4*np.sqrt(true_post_var),true_post_mean + 4*np.sqrt(true_post_var), 300)

    plt.plot(mu_range, norm.pdf(mu_range, true_post_mean, np.sqrt(true_post_var)), label=f'True mean posterior for component k={k}')
    plt.plot(mu_range, norm.pdf(mu_range, variational_means[k], np.sqrt(variational_var[k])), linestyle='--', label=f'q(μ_{k})')
plt.title('Comparing the true mean posterior with variational factors for the means')
plt.xlabel('Means')
plt.ylabel('Density')
plt.legend()
plt.savefig('../../images/variational_inference/posterior_mean_comp.png')

