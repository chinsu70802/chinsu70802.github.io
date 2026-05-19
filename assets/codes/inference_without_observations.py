import matplotlib.pyplot as plt
import numpy as np

def gaussian(x, mu, sigma):
    return (np.exp((-(x - mu)**2) / (2 * (sigma**2)))) / (np.sqrt((sigma**2)*2*np.pi))\

x = np.linspace(-20,20,1000)
y = gaussian(x, 3, 2)

"""
We know that a particular hidden variable is distributed in some way, but we do not know what realization of the distribution nature decides to produce at any given moment. We are estimating that realization using inference techniques.
"""

plt.plot(x, y, label='Distribution of the hidden variable x')

"""
We are given the mean of the gaussian in this case (we are also told that it is a gaussian). But no other information is given to you. Below, we are estimating the residue using the best possible estimate, the mean itself.
"""
x_tilde = x - 3 
y_tilde = gaussian(x_tilde, 0, 2)

plt.plot(x_tilde, y_tilde, label=r'Distribution of the residue ($\hat{X} = X - \bar{X}$)')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.grid(True)

plt.show()