"""
VISUALIZING GAUSSIAN APPROXIMATIONS OBTAINED VIA LAPLACE METHOD FOR:
1) CHI-SQUARED
2) MIXTURE OF GAUSSIANS
"""

import matplotlib.pyplot as plt
from scipy.special import gamma
import numpy as np

def chi_squared(x, k):
    return (x**((k/2) - 1)*np.exp(-x/2))/(gamma(k/2) * (2**(k/2))) #PDF of chi-squared

def mixture_of_gaussians(x):
    return (0.5*np.exp((-((x - 1)**2))/2) + 0.5*np.exp((-((x - 5)**2))/2))/(np.sqrt(2*np.pi)) #PDF of mixture of gaussians

def gaussian(x, mu, var):
    return (np.exp((-((x - mu)**2))/(2*var)))/np.sqrt(2*np.pi*var) #PDF of gaussian distribution

def gradient_of_log(posterior, x, k):     #Helper function to compute the gradient of the log joint density with respect to latent variable for gradient ascent
    if posterior == 'chi_squared':
        return ((k/2)- 1)*(1/(x + 1e-6)) - 0.5
    if posterior == 'mixture_of_gaussians':
        return ((np.exp((-((x - 1)**2))/2)*(-(x - 1))) + (np.exp((-((x - 5)**2))/2)*(-(x - 5))))/(np.exp((-((x - 1)**2))/2) + np.exp((-((x - 5)**2))/2))

def gradient_ascent(posterior, k):
    if posterior == 'chi_squared':
        x = 0.5
        x_prev = 0
        i = 1
        while (abs(x - x_prev) >= 1e-4):
            x_prev = x
            x = x_prev + (1/i) * gradient_of_log('chi_squared', x_prev, k) #Learning rate is inversely proportional to iteration index (Refer to the explanation for more details regarding this choice)
            i += 1
        return x
    if posterior == 'mixture_of_gaussians':
        x = 0.5
        x_prev = 0
        i = 1
        while ((x - x_prev) >= 1e-4):
            x_prev = x
            x = x_prev + (1/i) * gradient_of_log('mixture_of_gaussians', x_prev, k)
            i += 1
        return x

def second_derivative_of_log_gaussian_mixture(x):  #Helper function to compute the variance of the laplace approximation of the mixture of gaussians
    f = np.exp((-((x - 1)**2))/2) + np.exp((-((x - 5)**2))/2)
    fp = ((-(x - 1))*np.exp((-((x - 1)**2))/2)) + ((-(x - 5))*np.exp((-((x - 5)**2))/2))
    fpp = ((((x - 1)**2) - 1)*np.exp((-((x - 1)**2))/2)) + ((((x - 5)**2) - 1)*np.exp((-((x - 5)**2))/2))
    return ((fpp*f) - (fp**2))/((f + 1e-6)**2)

def laplace(posterior, k=None):
    mean = gradient_ascent(posterior, k)
    if posterior == 'chi_squared':
        variance = 2*(k - 2)
    else:
        variance =  ((-second_derivative_of_log_gaussian_mixture(mean)) + 1e-6)**(-1)
    return mean, variance

k = 4    #Degrees of freedom of chi-squared distribution
x_mix = np.linspace(-50,50, num = 5000)
x_chi = np.linspace(0.001,3*k, num=5000)
chi_y = [chi_squared(i, k) for i in x_chi]
mix_y = [mixture_of_gaussians(i) for i in x_mix]
gaussian_approx_chi_mean, gaussian_approx_chi_var = laplace('chi_squared', k)
gaussian_approx_mix_mean, gaussian_approx_mix_var = laplace('mixture_of_gaussians')
gaussian_approx_chi = [gaussian(i, gaussian_approx_chi_mean, gaussian_approx_chi_var) for i in x_chi]
gaussian_approx_mix = [gaussian(i, gaussian_approx_mix_mean, gaussian_approx_mix_var) for i in x_mix]

plt.figure()
plt.plot(x_chi, chi_y, label='Actual Chi-Squared')
plt.xlabel('x')
plt.ylabel('Density')
plt.title(f'Laplace approximation of Chi-Squared distribution with k = {k}')
plt.plot(x_chi, gaussian_approx_chi, label='Gaussian Approximated Chi-Squared')
plt.legend()

plt.figure()
plt.plot(x_mix, mix_y, label='Actual Mixture of Gaussians')
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Laplace approximation of Mixture of Gaussians with 2 components')
plt.plot(x_mix, gaussian_approx_mix, label='Gaussian Approximated Mixture of Gaussians')
plt.legend()

plt.show()

