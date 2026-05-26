import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.integrate import quad
from scipy.stats import norm
from tqdm import tqdm
import numpy as np

def mixture_of_beta(x, p_1 = 0.4, p_2 = 0.6, a_1 = 2, b_1 = 10, a_2 = 10, b_2 = 2): # PDF of mixture of beta
    if x < 0 or x > 1:
        return 0
    return p_1*(gamma(a_1 + b_1)/(gamma(a_1)*gamma(b_1)))*((x)**(a_1 - 1))*((1 - x)**(b_1 - 1)) + p_2*(gamma(a_2 + b_2)/(gamma(a_2)*gamma(b_2)))*((x)**(a_2 - 1))*((1 - x)**(b_2 - 1))

def sample_from_mixture_of_beta(n):    # Sampling from the above distribution
    samples = []
    while len(samples) < n:
        component = np.random.choice([0,1], p=[0.4, 0.6])
        if component == 0:
            samples.append(np.random.beta(2,10))
        else:
            samples.append(np.random.beta(10,2))
    return samples

def f(y):               # We want to compute expectation of this function over the mixture of beta
    return np.exp((-(y - 0.5)**2)/0.000001)

def normal_pdf(x, mu, var):  # PDF of truncated normal distribution which will be considered as one of the proposal distribution
    sigma = np.sqrt(var)
    a = 0
    b = 1
    Z = (norm.cdf((b - mu)/sigma)-norm.cdf((a - mu)/sigma))
    if x < a or x > b:
        return 0
    return (np.exp(-((x - mu)**2)/(2*var))/(np.sqrt(2*np.pi*var) * Z))

def integrand(y):           # Integrand of the actual expectation of the function over the mixture of beta
    return f(y)*mixture_of_beta(y)


res_norm = []
res_unif = []
res_true = []
N = 5000   # Drawing N samples (from either the actual mixture of beta or from proposal distributions under consideration) to compute the empirical expectation
for i in tqdm(range(100), desc='Computing variance of estimates'):   # Doing 100 runs of estimating the expectation of the function
    samples_norm = []  
    while len(samples_norm) < N:
        s = np.random.normal(loc=0.5,scale=1)
        if (s >= 0) and (s <= 1):
            samples_norm.append(s)                  # Samples drawn from the proposal truncated normal
    samples_beta = sample_from_mixture_of_beta(N)   # Samples drawn from the actual mixture of beta 
    summand_norm = np.array([(mixture_of_beta(i)*f(i))/normal_pdf(i, 0.5, 1) for i in samples_norm])  # Computing (P(x)f(x))/(g(x)) for the sampled values from the proposal distribution (g is the proposal distribution which is the truncated normal here)
    summand_true = np.array([f(i) for i in samples_beta])   # Computing the value of the function at values sampled from the actual mixture of beta
    estimated_exp_norm = np.mean(summand_norm)     # Computing the empirical expectation of (P(x)f(x))/(g(x)) where x is drawn from the proposal distribution
    estimated_exp_true = np.mean(summand_true)     # Computing empirical expectation of f(x) where x is drawn from the actual mixture of beta
    res_norm.append(estimated_exp_norm)
    res_true.append(estimated_exp_true)

res_norm = np.array(res_norm)
res_true = np.array(res_true)
mean_exp_norm = np.mean(res_norm)     # Computing mean of 100 estimates of expectation obtained by sampling from proposal distribution
mean_exp_true = np.mean(res_true)     # Computing mean of 100 estimates of expectation obtained by sampling from the actual mixture of beta
var_exp_norm = np.var(res_norm)       # Computing variance of 100 estimates of expectation obtained by sampling from the proposal distribution
var_exp_true = np.var(res_true)       # Computing variance of 100 estimates of expectation obtained by sampling from the actual mixture of beta
actual_exp, _ = quad(integrand, 0, 1)  # Computing the true expectation of the the function over mixture of beta (This is done by integrating the product P(x)*f(x) where x varies from 0 to 1)
print(f"Actual Expectation: {actual_exp}")
print(f"Mean of estimated expectation using gaussian proposal: {mean_exp_norm}")
print(f"Mean of estimated expectation using true proposal: {mean_exp_true}")

print(f"Variance of estimated expectation using gaussian proposal: {var_exp_norm}")
print(f"Variance of estimated expectation using true proposal: {var_exp_true}")

x = np.linspace(0, 1, num=5000)
mix_beta_y = [mixture_of_beta(i) for i in x]
gauss_y = [normal_pdf(i, 0.5, 1) for i in x]
f_y = [f(i) for i in x]
norm_y = [(mixture_of_beta(i)*f(i))/normal_pdf(i, 0.5, 1) for i in x]
plt.figure()
plt.plot(x, mix_beta_y, label='True Beta Mixture')
plt.plot(x, gauss_y, label='Gaussian PDF')
plt.plot(x, f_y, label='Function of interest')
plt.plot(x, norm_y, label='Ratio with normal density')
plt.legend()
plt.show()