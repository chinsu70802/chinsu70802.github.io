"""
GIBBS SAMPLING IN ACTION!
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma
from scipy.stats import invgamma
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import ccf


def normal_pdf(x, mu, sigma):               # PDF of normal distribution
    return (np.exp((-(x - mu)**2)/(2*(sigma**2))))/(sigma*np.sqrt(2*np.pi))

def inv_gamma_pdf(x, alpha, beta):          # PDF of inverse gamma distribution for the prior of variance
    return ((beta**alpha)/(gamma(alpha)))*(x**(-alpha-1))*np.exp(-(beta/x))

def log_joint_pdf(y, mu, sigma, alpha, beta, m, sigma_mu, n):           # Logarithm of joint PDF (Logarithm was taken for numerical stability)
    log_joint = -((n/2) + alpha + 1)*np.log(sigma**2) - ((np.sum((y - mu)**2))/(2*(sigma**2))) - (((mu - m)**2)/(2*(sigma_mu**2))) - (beta/(sigma**2))
    return log_joint

def get_obs_sequence(mu, sigma, n):                                     # Getting observation sequence Y from a normal distribution with mean and variance taken from a normal and inverse gamma distribution respectively
    return np.random.normal(loc=mu, scale=sigma, size=n)

def mu_prior(mu, m, sigma_mu):                                          # A normal prior for the mean 
    return normal_pdf(mu, m, sigma_mu)

def var_prior(var, alpha, beta):                                        # An inverse gamma prior for the variance 
    return inv_gamma_pdf(var, alpha, beta)

def params_mu_cond_var_and_obs(y, var, m, sigma_mu):                    # Full conditional distribution of the mean  given the variance and observartion for Gibbs Sampler
    N = len(y)
    mean = (((sigma_mu**2)*N*np.mean(y)) + m*var)/(N*(sigma_mu**2) + var)
    variance = (var*(sigma_mu**2))/(var + N*(sigma_mu**2))
    return mean, variance

def params_var_cond_mu_and_obs(y, mu, a, b):                            # Full conditional distribution of the var given the mean and observation for Gibbs Sampleer
    N = len(y)
    alpha =  (N/2) + a
    beta = N*(np.mean((y - mu)**2)/2) + b
    return alpha, beta

def gibbs_sampling(y, m, sigma_mu, a, b, T):                            # Gibbs sampling algorithm as explained in the blog
    mu_init = np.random.normal(loc=0, scale=1)
    var_init = invgamma.rvs(a=3,scale=2)
    samples = [(mu_init, var_init)]
    for i in range(T):
        mean, variance = params_mu_cond_var_and_obs(y, var_init, m, sigma_mu)
        mu_init = np.random.normal(loc=mean, scale=np.sqrt(variance))
        alpha, beta = params_var_cond_mu_and_obs(y, mu_init, a, b)
        var_init = invgamma.rvs(a=alpha, scale=beta)
        samples.append((mu_init, var_init))
    return samples

m = 0                      # Mean hyperparameter of the gaussian prior of the mean 
sigma_mu = 1                # Variance hyperaparameter of the gaussian prior of the mean
a = 2                       # alpha parameter of the inverse gamma prior on the variance
b = 1                    # beta parameter of the inverse gamma prior on the variance
T = 10000                   # Number of iterations in the Gibbs Sampler
N = 100                      # Total number of observations Y

# The following lines of code are for the purposes of getting the samples from the Gibbs Sampler and plotting stuff for analysis

mu = 4
var = 0.5
y = get_obs_sequence(mu, np.sqrt(var), N)
sample_mean = np.mean(y)
sample_std = np.std(y)
mu_grid = np.linspace(sample_mean - 3*sample_std, sample_mean + 3*sample_std, 300)
var_grid = np.linspace(0.01, (sample_std**2) * 3, 300)
mu_joint, var_joint = np.meshgrid(mu_grid, var_grid)
log_p = np.vectorize(lambda mu, var: log_joint_pdf(y, mu, np.sqrt(var), a, b, m, sigma_mu, N))(mu_joint, var_joint)
p = np.exp(log_p - log_p.max())
samples = gibbs_sampling(y, m ,sigma_mu, a, b, T)
mu_samples, var_samples = zip(*samples)

mu_samples = np.array(mu_samples)
var_samples = np.array(var_samples)

burn_in = 500

print("True Mean: ", mu)
print("True Variance: ", var)

plt.figure()
plt.title('Contour of joint pdf v/s samples from Gibbs Sampler')
plt.contourf(mu_joint, var_joint, p, levels=30, cmap='Blues')
plt.colorbar()
plt.xlabel(r'$\mu$')
plt.ylabel(r'$\sigma^2$')
plt.scatter(mu_samples[burn_in:], var_samples[burn_in:], s=1, alpha=0.4, color='red', label='Samples obtained via Gibbs Sampling')
plt.savefig('../../images/approximate_inference/contour_vs_samples_gibbs.png')

plt.figure()
plt.title('Trace plot of the mean')
plt.plot(mu_samples)
plt.xlabel('Iterations')
plt.ylabel(r'$\mu$')
plt.savefig('../../images/approximate_inference/mean_trace_gibbs.png')

plt.figure()
plt.title('Trace plot of the variance')
plt.plot(var_samples)
plt.xlabel('Iterations')
plt.ylabel(r'$\sigma^2$')
plt.savefig('../../images/approximate_inference/var_trace_gibbs.png')

plt.figure()
plt.title('Estimated Marginal Distribution of the mean')
plt.hist(mu_samples[burn_in:], density=True, bins=100)
plt.xlabel(r'$\mu$')
plt.ylabel('Density')
plt.axvline(x=np.mean(mu_samples[burn_in:]), color='red', linestyle='--', label=f'Mean of Posterior mean = {np.mean(mu_samples[burn_in:]):.2f}')
plt.axvline(x=mu, color='black', linestyle='--', label=f'True $\mu$ = {mu:.2f}')
plt.legend()
plt.savefig('../../images/approximate_inference/mean_marginal_gibbs.png')

plt.figure()
plt.title('Estimated Marginal Distribution of the variance')
plt.hist(var_samples[burn_in:], density=True, bins=100)
plt.xlabel(r'$\sigma^2$')
plt.ylabel('Density')
plt.axvline(x=np.mean(var_samples[burn_in:]), color='red', linestyle='--', label=f'Mean of Posterior variance = {np.mean(var_samples[burn_in:]):.2f}')
plt.axvline(x=var, color='black', linestyle='--', label=f'True $\sigma^2$ = {var:.2f}')
plt.legend()
plt.savefig('../../images/approximate_inference/var_marginal_gibbs.png')

plt.figure()
plot_acf(mu_samples, lags=100)
plt.title(r'Autocorrelation of $\mu$ samples')
plt.savefig('../../images/approximate_inference/mu_acf.png')

plt.figure()
plot_acf(var_samples, lags=100)
plt.title(r'Autocorrelation of $\sigma^2$ samples')
plt.savefig('../../images/approximate_inference/var_acf.png')

plt.figure()
cc = ccf(mu_samples, var_samples)
plt.stem(cc[:50])
plt.title('Cross-correlation between $\mu$ and $\sigma^2$ samples')
plt.xlabel('Lag')
plt.savefig('../../images/approximate_inference/cross_corr_mu_var.png')

plt.show()