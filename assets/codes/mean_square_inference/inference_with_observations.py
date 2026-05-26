import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

x = np.random.choice([-1, +1], size=100000)
std_dev_n = 1
n = np.random.normal(0, std_dev_n, size=100000)
y = x + n

x_hat = np.tanh(y/(std_dev_n**2)) # Optimal estimate in the LMSE sense is tanh(y/(variance of noise))!
x_tilde = x - x_hat # Residue with respect to the conditional mean estimate

"""
Plotting the histogram of x and kernel density estimate of x_tilde to observe any potential variance reduction in the residue
"""
var_x = np.var(x)
var_x_tilde = np.var(x_tilde)
plt.hist(x, density=True, label=f"Histogram of x (Variance = {var_x:.2f})")
sns.kdeplot(x_tilde, label=f"KDE of the residue (Variance = {var_x_tilde:.2f})")
plt.legend(loc='upper right')
plt.show()