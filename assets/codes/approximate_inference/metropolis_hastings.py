"""
METROPOLIS-HASTINGS ALGORITHM IN ACTION
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma
from scipy.stats import truncnorm, norm
from scipy import integrate
import seaborn as sns
from statsmodels.stats.diagnostic import acorr_ljungbox
import arviz as az
from scipy.stats import pearsonr
from tqdm import tqdm

def g(x):           # Numerator of the target distribution (proportional to the target distribution whose support exists from 0 to 2)
    if x < 0 or x > 2:
        return 0
    return ((1 + np.sin(3*np.pi*x))**2)*np.exp(-x)

def log_g(x):       # Logarithm of the numerator of target distribution for numerical stability and ease of computation
    if x < 0 or x > 2:
        return -np.inf
    return np.log((1 + np.sin(3*np.pi*x))**2) + (-x)

def log_normal_proposal_pdf(x, mu, var):        #The proposal PDF is a truncated normal distribution from 0 to 2
    sigma = np.sqrt(var)
    a, b  = 0, 2
    if x < a or x > b:
        return -np.inf
    log_Z     = np.log(norm.cdf((b - mu)/sigma) - norm.cdf((a - mu)/sigma))
    log_kernel = -((x - mu)**2) / (2 * var) - 0.5 * np.log(2 * np.pi * var)
    return log_kernel - log_Z

def metropolis_hastings(var, J):            # Implementation of Metropolis-Hastings as described in the blog
    a = 0
    sigma = np.sqrt(var)
    b = 2
    mu = 0
    alpha = (a - mu) / (sigma)
    beta = (b - mu) / (sigma)
    x_prev = truncnorm.rvs(alpha, beta, loc=mu, scale=sigma)
    samples = [x_prev]
    num_accept = 0
    for i in range(J):
        mu = x_prev
        alpha = (a - mu) / (sigma)
        beta = (b - mu) / (sigma)
        x_candidate = truncnorm.rvs(alpha, beta, loc=mu, scale=sigma)
        log_A = (log_g(x_candidate) - log_g(x_prev)) + (log_normal_proposal_pdf(x_prev, x_candidate, var) - log_normal_proposal_pdf(x_candidate, x_prev, var))
        if np.log(np.random.uniform()) < log_A:
            x_prev = x_candidate
            num_accept += 1
        samples.append(x_prev)
    return samples, num_accept

def x_gx(x):            # Integrand of the expectation of the target distribution
    return x * g(x)

Z,_ = integrate.quad(g, 0, 2)       # Obtaining normalizing constant of the target distribution for purpose of comparison with MCMC samples
numerator, _ = integrate.quad(x_gx, 0, 2)      
true_mean = numerator / Z           # Actual expectation (that will be compared against running mean of samples in the markov chain)
J = 10000
var = [0.001]           # Different variances of the truncated normal PDF that will be considered that will be considered
accept_rate = []
ess = []
burn_in = 1000                      # We cannot expect the initial samples to be representative of the target distribution as the chain may not have attained a stationary distribution yet!
x = np.linspace(-2,2, num=5000)
g_y = [g(k)/Z for k in x]           # Plotting the PDF of the true distribution
for i in tqdm(var):
    samples, num_accept = metropolis_hastings(i, J)
    running_mean = np.cumsum(samples) / np.arange(1, len(samples) + 1)          # Computing the cumulative mean of samples to compare against the true expectation
    plt.figure()        
    plt.plot(x,g_y, label='The target distribution')
    plt.hist(samples[burn_in:],bins=100,density=True, label='Histogram of the samples drawn via Metropolis-Hastings')   # Considering samples only after the burn_in period
    plt.title(f'Histogram of MCMC samples for variance = {i}')
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(f'../../images/approximate_inference/hist_var_{i}_just.png')

    plt.figure()
    plt.plot(samples[burn_in:2500])       # Trace plot of the chain
    plt.xlabel('Iteration')
    plt.ylabel('Values')
    plt.title(f'MCMC Trace Plot for all iterations for variance = {i}')
    plt.savefig(f'../../images/approximate_inference/trace_var_{i}_just.png')

    plt.figure()
    plt.plot(running_mean, label='Running mean')            # Comparing the running mean of MCMC samples with the actual expectation
    plt.axvline(burn_in, color='red', linestyle='--', label='Burn-in cutoff')
    plt.axhline(true_mean, color='black', linestyle='--', label='True mean')
    plt.xlabel('Iteration')
    plt.ylabel('Running mean')
    plt.title(f'Running mean for variance = {i}')
    plt.legend()
    plt.savefig(f'../../images/approximate_inference/running_mean_var_{i}_just.png')

    accept_rate_num = (num_accept/J)*100           # Computing the acceptance rate for the given variance of proposal distribution
    accept_rate.append(accept_rate_num)

    samples = samples[burn_in:]

    lags = [j for j in range(1,101)]
    acf = [pearsonr(samples[:-k], samples[k:])[0] for k in lags]            # Computing the autocorrelation factor of the samples for different lags
    tau = 1 + 2*sum(acf)
    ess.append(len(samples)/tau)                                            # Computing Effective Sample Size (ESS)
    plt.figure()
    plt.plot(lags, acf)
    plt.xlabel('Lags')
    plt.ylabel('AutoCorrelation Factor (ACF)')
    plt.title(f'AutoCorrelation Factor for variance = {i}')
    plt.savefig(f'../../images/approximate_inference/acf_var_{i}_just.png')

plt.figure()
plt.plot(var, accept_rate, marker='s')
plt.xlabel('Variance of proposal distribution')
plt.ylabel('Acceptance Rate')
plt.title('Acceptance Rate for different variances of the proposal distribution')
plt.savefig('../../images/approximate_inference/ar_just.png')

plt.figure()
plt.plot(var, ess, marker='s')
plt.xlabel('Variance of proposal distribution')
plt.ylabel('Effective Sample Size')
plt.title('Effective Sample Size (ESS) for different variances of the proposal distribution')
plt.savefig('../../images/approximate_inference/ess_just.png')


print("\nSummary:")
for v, ar, es in zip(var, accept_rate, ess):
    print(f"var={v} | acceptance rate={ar:.1f}% | ESS={es:.0f}")

plt.tight_layout()
plt.show()