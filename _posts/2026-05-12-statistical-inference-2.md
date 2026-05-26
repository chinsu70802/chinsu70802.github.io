---

layout: post
title: "Approximate Inference"
date: 2026-05-25
categories: [Statistics]
tags: [Statistics]
math: True
---

## PROLOGUE

In the previous post on mean square inference, we observed that the conditional expectation of the hidden variable given the observable, 
$\mathbb{E}[X\|Y]$, is the optimal estimate. It is worthwhile to look at another example where the evaluation of such an expectation under MSE risk is revelatory:

Consider the following probabilistic model of a Bayesian Mixture of Gaussians - 

$$
\begin{align}
&\mu_k \sim \mathcal{N}(0, \sigma^2) \qquad k = 1, 2, \cdots, M \label{eqn1}\\
&c_i \sim Cat(\frac{1}{M},\frac{1}{M},\cdots,\frac{1}{M}), \qquad i = 1, 2, \cdots, N \label{eqn2}\\
&x_i \sim \mathcal{N}(c_{i}^T\mathbf{\mu}, 1), \qquad i = 1, 2, \cdots, N  \label{eqn3}
\end{align}
$$

There are $M$ components/clusters in this setup. Each cluster has its centroid as $\mu_k$, which is drawn from a normal distribution as specified in $\eqref{eqn1}$. $\sigma^2$ is a fixed hyperparameter, and this same variance is used in all the $M$ clusters. We can put all these scalar centroids into a $M$-component vector, $\mathbf{\mu}$. This is a hidden variable (Recall that 'hidden' means you know the underlying distribution of the variable, but do not know what gets realized from the distribution). After fixing $\mathbf{\mu}$, we collect observables $x_i$ from this mixture. Each observation may have been collected from any one of the clusters at random. We define another variable $c_i$, which is a categorical variable that corresponds to the cluster assigned to $x_i$. Let us assume each of the $M$ clusters are equally likely to have been used to sample (or assigned to) $x_i$. Then, $x_i$ follows a normal distribution as defined in $\eqref{eqn3}$, and it is clearly dependent on $c_i$ and $\mathbf{\mu}$. $c_i$ is a hidden categorical variable that can take one of the values at any given moment from $1$ to $M$. This can also be represented as an $M$-dimensional one-hot encoding, where $c_{im} = 1$ when $c_i = m$ ($1 \leq m \leq M$). As $\mathbf{\mu}$ is also a vector, the inner product between $c_i$ and $\mathbf{\mu}$ will give you the centroid of the $m^{th}$ component if $c_i = m$. Note that the centroids are independent of each other. The $N$ observations have cluster assignments that do not dependent on the cluster assigned to any other observation. The $N$ observations themselves are drawn independent of any other observation.

Given the sequence of observations 
$\mathbf{x}$ $=$ {$x_1, x_2, \cdots, x_N$}, with $\mathbf{c}$ being a sequence of $N$ cluster assignments corresponding to the $N$ observations ($\mathbf{c}$ is independent of $\mathbf{\mu}$, and the former can be visualized as a $M \times N$ matrix), it would be interesting to know the location of all the centroids in the aforementioned model. In other words, we are interested in computing $\mathbb{E}[\mathbf{\mu}|\mathbf{x}]$. For example, imagine each cluster depicts a topic and each observable is a word sampled from some topic. Estimating the location of all centroids enables us to understand the separability of topics. In an exercise in the previous post, you would have computed the conditional expectation by first calculating the posterior distribution. We will start in a similar fashion here as well. (Imagine having access to a dataset where for a given sequence of observations, we are told the location of centroid. We can empirically compute the MSE risk now!)

$$
\begin{align}
&\mathbb{P}(\mathbf{\mu}|\mathbf{x}) = \frac{\mathbb{P}(\mathbf{\mu},\mathbf{x})}{\mathbb{P}(\mathbf{x})} \label{eqn4} \\
&\mathbb{P}(\mathbf{x}) = \int_{\mathbf{\mu}} \sum_{\mathbf{c}} \mathbb{P}(\mathbf{\mu},\mathbf{c},\mathbf{x}) \,d\mathbf{\mu} \qquad \text{(Evaluating the denominator in \eqref{eqn4})} \label{eqn5} \\
\end{align}
$$

$$
\begin{align}
\mathbb{P}(\mathbf{\mu},\mathbf{c},\mathbf{x}) &= \mathbb{P}(\mathbf{\mu})\mathbb{P}(\mathbf{x},\mathbf{c}|\mathbf{\mu}) \qquad \text{(Evaluating the joint probability in \eqref{eqn5})} \\
                    &= \mathbb{P}(\mathbf{\mu})\mathbb{P}(\mathbf{c})\mathbb{P}(\mathbf{x}|\mathbf{c},\mathbf{\mu}) \qquad \text{(Using the fact that $\mathbf{c}$ is independent of $\mathbf{\mu}$)}\\
                    &= \mathbb{P}(\mathbf{\mu})\prod_{i = 1}^{N} \mathbb{P}(c_i) \mathbb{P}(x_i|c_i, \mathbf{\mu}) \qquad \text{(Using the fact that $c_i$'s are independent of each other, and the same for $x_i$'s as well)} \label{eqn6}\\
\mathbb{P}(\mathbf{x}) &= \int_{\mathbf{\mu}} \sum_{\mathbf{c}} \mathbb{P}(\mathbf{\mu})\prod_{i = 1}^{N} \mathbb{P}(c_i) \mathbb{P}(x_i|c_i, \mathbf{\mu}) \,d\mathbf{\mu} \qquad \text{(Substituting \eqref{eqn6} in \eqref{eqn5})} \\
                       &= \int_{\mathbf{\mu}} \sum_{\mathbf{c}} \mathbb{P}(\mathbf{\mu}) \mathbb{P}(\mathbf{c}) \prod_{i = 1}^{N} \mathbb{P}(x_i|c_i, \mathbf{\mu}) \,d\mathbf{\mu} \qquad \text{(As $c_i$'s are drawn independent of each other, $\mathbb{P}(c) = \prod_{i = 1}^{N} \mathbb{P}(c_i)$)} \\
                       &= \int_{\mathbf{\mu}} \mathbb{P}(\mathbf{\mu}) \sum_{\mathbf{c}} \mathbb{P}(\mathbf{c}) \prod_{i = 1}^{N} \mathbb{P}(x_i|c_i, \mathbf{\mu}) \,d\mathbf{\mu} \qquad \text{($\mathbf{\mu}$ is not dependent on $\mathbf{c}$)} \\
                       &=  \sum_{\mathbf{c}} \mathbb{P}(\mathbf{c}) \int_{\mathbf{\mu}} \mathbb{P}(\mathbf{\mu}) \prod_{i = 1}^{N} \mathbb{P}(x_i|c_i, \mathbf{\mu}) \,d\mathbf{\mu} \qquad \text{(The integral can be taken inside the summation)} \label{eqn7}\\
\end{align}
$$

The term within the summation in 
$\eqref{eqn7}$ can be simplified succinctly - we know that 
$\mathbb{P}(x_i|c_i, \mathbf{\mu})$ follows a normal distribution as defined in 
$\eqref{eqn3}$. The product of these normal distributions with $\mathbb{P}(\mathbf{\mu})$ can be easily computed via completion of squares (to get a form proportional to the gaussian density of $\mathbf{\mu}$). Multiplication of the previous result with $\mathbb{P}(\mathbf{c})$ is trivial.

> Try getting a closed form expression for 
>$\int_{\mathbf{\mu}} \mathbb{P}(\mathbf{\mu}) \prod_{i = 1}^{N} \mathbb{P}(x_i|c_i, \mathbf{\mu})$. You will notice that it simplifies into a form proportional to the gaussian density!
{: .prompt-exercise}

But the catch is the summation over all possible cluster assignments. Since there are $M$ possible cluster assignments for each of the $N$ observations, we have a total of $M^N$ summands (each with different normalization constants dependent on $\mathbf{x}$ and $\mathbf{c}$, which adds to the complexity). If $M$ and $N$ were small numbers (like $2$, $3$ etc.), then evaluating 
$\mathbb{P}(\mathbf{x})$ is a trivial task. Imagine choosing $100$ clusters, and $1000$ observations. We have ${100}^{1000}$ summands in $\eqref{eqn7}$. Getting a computationally feasible expression is not possible for this situation (which is realistic in many machine learning applications). We call such distributions as intractable. The above example requires computation of the order of $\theta(M^N)$, which is not computationally tractable. It is hard to compute $\mathbb{P}(\mathbf{\mu}|\mathbf{x})$ as $\mathbb{P}(x)$ is intractable, which in turn makes it hard to compute $\mathbb{E}[\mathbf{\mu}|\mathbf{x}]$. Are there any smart techniques that can be used to overcome this issue? These techniques fall under the umbrella of Approximate Inference!

{% raw %}
$$
\DeclareMathOperator\erf{erf}
$$
{% endraw %}

> A mathematical formulation is said to have a closed-form expression if it can be expressed using basic operations/functions like addition, subtraction, multiplication, division, trignometric functions, logarithms etc (also called basic). For example, the quadratic formula is a closed-form expression. On the other hand, $\erf(x) = \frac{2}{\pi}\int_{0}^{x}e^{-t^2} \,dt$ is not known to have any closed-form expression. This defintion varies across textbooks. Some authors consider $\erf(x)$ as basic, and any formula that can be expressed in terms of it is considered a closed-form expression. Irrespective of these definitions, all that matters is compactness and computational feasibility of the concerned distribution. In the bayesian mixture of gaussians example, we have a finite closed-form expression (going by the definition in the first line of this info box) for $\mathbb{P}(x)$. As there are exponential number of terms in it, computational infeasibility comes into picture, and that makes it an intractable distribution! 
{: .prompt-info}

## LAPLACE METHOD

This is one of the earliest attempts to overcome the issue of intractable distributions.

$$
\begin{align}
\mathbb{P}(\mathbf{\mu}|\mathbf{x}) &= \frac{\mathbb{P}(\mathbf{x},\mathbf{\mu})}{\mathbb{P}(\mathbf{x})} \\
                &= \frac{\exp{\ln\mathbb{P}(\mathbf{x},\mathbf{\mu})}}{\exp{\ln\mathbb{P}(\mathbf{x})}} \label{eqn19}\\
\end{align}
$$

In most of the real-world cases, it is easy to get a reasonable closed-form expression for $\mathbb{P}(\mathbf{x},\mathbf{\mu})$ than $\mathbb{P}(\mathbf{x})$. Given that we have access to $\mathbb{P}(\mathbf{x},\mathbf{\mu})$, we can find $\ln\mathbb{P}(\mathbf{x},\mathbf{\mu})$ and identify the $\mu$ that maximizes it. The $\mu$ that maximizes $\ln\mathbb{P}(\mathbf{x},\mathbf{\mu})$ also maximizes the posterior. Essentially, that $\mathbf{\mu}$ is $\mu_{MAP}$ (Maximum A-Posteriori). 

To find $\mu_{MAP}$, we can perform gradient ascent on $\ln\mathbb{P}(\mathbf{x},\mathbf{\mu})$ with respect to $z$ until convergence is achieved: 
$$\mathbf{\mu}_m = \mathbf{\mu}_{m - 1} + \lambda_{m}\nabla_{\mathbf{\mu}}\ln\mathbb{P}(\mathbf{x},\mathbf{\mu}) \Big|_{\mathbf{\mu} = \mathbf{\mu}_{m - 1}}$$. Here, $m$ denotes the iteration index, and convergence is achieved when $\mathbf{\mu}_m \approx \mathbf{\mu}_{m - 1}$. Also, $\lambda_{m}$ is the learning rate at iteration $m$. As $\mathbf{\mu}$ gets closer to the maxima, the learning rate is expected to decrease. 

Once 
$$\mathbf{\mu}_{MAP}$$ is achieved via gradient ascent, we can do a second-order taylor series expansion of $\ln\mathbb{P}(\mathbf{x},\mathbf{\mu})$ at $\mathbf{\mu} = \mathbf{\mu}_{MAP}$:

$$
\begin{align}
\ln\mathbb{P}(\mathbf{x},\mathbf{\mu}) &\approx \ln\mathbb{P}(\mathbf{x},\mathbf{\mu}_{MAP}) + \nabla_{\mathbf{\mu}} \ln\mathbb{P}(x,\mathbf{\mu})^T \Big|_{\mathbf{\mu} = \mathbf{\mu}_{MAP}} (\mathbf{\mu} - \mathbf{\mu}_{MAP}) + \frac{1}{2}(\mathbf{\mu} - \mathbf{\mu}_{MAP})^T \nabla_{\mathbf{\mu}}^2 \ln\mathbb{P}(x,\mathbf{\mu}) \Big|_{\mathbf{\mu} = \mathbf{\mu}_{MAP}} (\mathbf{\mu} - \mathbf{\mu}_{MAP}) \\
                   &\approx \ln\mathbb{P}(x,\mathbf{\mu}_{MAP}) + \frac{1}{2}(\mathbf{\mu} - \mathbf{\mu}_{MAP})^T \nabla_{\mathbf{\mu}}^2 \ln\mathbb{P}(x,\mathbf{\mu}) \Big|_{\mathbf{\mu} = \mathbf{\mu}_{MAP}} (\mathbf{\mu} - \mathbf{\mu}_{MAP}) \qquad \text{(At $\mathbf{\mu}_{MAP}$, $\ln\mathbb{P}(x,\mathbf{\mu})$ attains the maximum and the gradient is zero there.)} \\
                   &\approx \ln\mathbb{P}(x,\mathbf{\mu}_{MAP}) - \frac{1}{2}(\mathbf{\mu} - \mathbf{\mu}_{MAP})^T (-\nabla_{\mathbf{\mu}}^2 \ln\mathbb{P}(x,\mathbf{\mu})) \Big|_{\mathbf{\mu} = \mathbf{\mu}_{MAP}} (\mathbf{\mu} - \mathbf{\mu}_{MAP}) \\
                   &\approx \ln\mathbb{P}(x,\mathbf{\mu}_{MAP}) - \frac{1}{2}(\mathbf{\mu} - \mathbf{\mu}_{MAP})^T (-(\nabla_{\mathbf{\mu}}^2 \ln\mathbb{P}(x,\mathbf{\mu}))^{-1})^{-1} \Big|_{\mathbf{\mu} = \mathbf{\mu}_{MAP}} (\mathbf{\mu} - \mathbf{\mu}_{MAP}) \label{eqn16} \\
\end{align}
$$ 

We can consider 
$$-(\nabla_{\mathbf{\mu}}^2 \ln\mathbb{P}(x,\mathbf{\mu}))^{-1} \Big|_{\mathbf{\mu} = \mathbf{\mu}_{MAP}}$$ as $R$. Then $\eqref{eqn16}$ becomes 
$$\ln\mathbb{P}(x,\mathbf{\mu}_{MAP}) - \frac{1}{2}(\mathbf{\mu} - \mathbf{\mu}_{MAP})^T R^{-1} (\mathbf{\mu} - \mathbf{\mu}_{MAP})^{-1}$$. Rewriting 
$\eqref{eqn19}$ using the above result, we have:

$$
\begin{align}
\mathbb{P}(\mathbf{\mu}|x) &\approx \frac{\exp{(\ln\mathbb{P}(x,\mathbf{\mu}_{MAP}) - \frac{1}{2}(\mathbf{\mu} - \mathbf{\mu}_{MAP})^T R^{-1} (\mathbf{\mu} - \mathbf{\mu}_{MAP}))}}{\exp{\ln\mathbb{P}(x)}} \\
                &\approx \frac{\exp{(\ln\mathbb{P}(x,\mathbf{\mu}_{MAP}) - \frac{1}{2}(\mathbf{\mu} - \mathbf{\mu}_{MAP})^T R^{-1} (\mathbf{\mu} - \mathbf{\mu}_{MAP}))}}{\exp{(\ln\int_{\mathbf{\mu}}\mathbb{P}(x,\mathbf{\mu}) \,d\mathbf{\mu})}} \\
                &\approx \frac{\exp{(\ln\mathbb{P}(x,\mathbf{\mu}_{MAP}) - \frac{1}{2}(\mathbf{\mu} - \mathbf{\mu}_{MAP})^T R^{-1} (\mathbf{\mu} - \mathbf{\mu}_{MAP}))}}{\int_{\mathbf{\mu}}\exp{(\ln\mathbb{P}(x,\mathbf{\mu}) \,d\mathbf{\mu})}} \qquad \text{(As both $\exp$ and $\ln$ are continous functions, the integration can be pushed outside.)} \label{eqn25}\\
                &\approx \frac{\exp{(\ln\mathbb{P}(x,\mathbf{\mu}_{MAP}) - \frac{1}{2}(\mathbf{\mu} - \mathbf{\mu}_{MAP})^T R^{-1} (\mathbf{\mu} - \mathbf{\mu}_{MAP}))}}{\int_{\mathbf{\mu}}\exp{(\ln\mathbb{P}(x,\mathbf{\mu}_{MAP}) - \frac{1}{2}(\mathbf{\mu} - \mathbf{\mu}_{MAP})^T R^{-1} (\mathbf{\mu} - \mathbf{\mu}_{MAP}) \,d\mathbf{\mu})}} \qquad \text{(Substituting $\eqref{eqn16}$ in the denominator of $\eqref{eqn25}$)} \\
                &\approx \frac{\exp{(-\frac{1}{2}(\mathbf{\mu} - \mathbf{\mu}_{MAP})^T R^{-1} (\mathbf{\mu} - \mathbf{\mu}_{MAP}))}}{\int_{\mathbf{\mu}}\exp{(-\frac{1}{2}(\mathbf{\mu} - \mathbf{\mu}_{MAP})^T R^{-1} (\mathbf{\mu} - \mathbf{\mu}_{MAP})) \,d\mathbf{\mu}}} \qquad \text{(Cancelling $\ln\mathbb{P}(x,\mathbf{\mu}_{MAP})$ from numerator and denominator)} \\
                &\approx \frac{\mathcal{N}(\mathbf{\mu}|\mathbf{\mu}_{MAP},R)}{\int_{\mathbf{\mu}}\mathcal{N}(\mathbf{\mu}|\mathbf{\mu}_{MAP},R) \,d\mathbf{\mu}} \qquad \text{(Both the numerator and denominator (within integral) when provided the apt normalization constant becomes a gaussian; both will have same normalization constant)}\\
                &\approx \mathcal{N}(\mathbf{\mu}|\mathbf{\mu}_{MAP},R) \qquad \text{(The denominator integrates to 1)}
\end{align}
$$

In effect, Laplace method tries to get a gaussian approximation of the posterior distribution, making it a tractable distribution! As can be observed immediately from the derivation, the hessian of the logarithm of joint density ($R$) is expected to be invertible. If this isn't guaranteed, then this method cannot be applied to approximate that particular posterior. It is said that in higher dimensions, we cannot expect the hessian to be invertible. So, it works mostly for continuous latent variables $\mathbf{\mu}$ in lower dimensions. With this laplace approximation of the intractable posterior, we can proceed to computing things of our interest - expectation, variance, maybe drawing samples etc.

```python
"""
VISUALIZING GAUSSIAN APPROXIMATIONS OBTAINED VIA LAPLACE METHOD FOR:
1) CHI-SQUARED (UNIVARIATE, UNIMODAL)
2) MIXTURE OF GAUSSIANS (UNIVARIATE, BIMODAL)
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
        x = 0.5         #Gradient ascent initialization
        x_prev = 0
        i = 1
        while (abs(x - x_prev) >= 1e-4):
            x_prev = x
            x = x_prev + (1/i) * gradient_of_log('chi_squared', x_prev, k) #Learning rate is inversely proportional to iteration index (Refer to the explanation for more details regarding this choice)
            i += 1
        return x
    if posterior == 'mixture_of_gaussians':
        x = 0.5         #Gradient ascent initialization
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
```

Chi-Squared is a right-skewed distribution. Most of the probability mass lies to the left, and a few extremely high values pull the tail outwards to the right. Let us visually see the laplace approximation for this distribution:

![Laplace Approximation of Chi-Squared](/assets/images/approximate_inference/laplace_chi_squared.png)

In this case, the gaussian approximation could reasonably capture the trend around the mode of the Chi-Squared distribution. But it failed to model the right-skewness seen in the actual Chi-Squared. The derivation of the gaussian approximation was done with second-order taylor series expansion. That could only capture the curvature of the target distribution (Chi-Squared here), but not the skewness or kurtosis. This is one of the failure cases if Laplace method is used for approximating posteriors.

> Try making the gaussian approximation work for the Chi-Squared case by changing the degree of freedom ($k$ in the code) and observing the fit!
{: .prompt-exercise}

Mixture of gaussians is a bimodal distribution. Let us visually see the laplace approximation for this distribution:

![Laplace Approximation of Mixture of Gaussians](/assets/images/approximate_inference/laplace_mix_gauss.png)

The gaussian approximation could only model one of the components appreciably. This is because the initial latent variable in the gradient ascent algorithm was set to $0.5$ in the code. It could ascend to the nearest local maxima and stop there. This is another failure case of Laplace method.

> Try varying the initial latent variable in the gradient descent algorithm ($x = 0.5$ line in the code) to observe its effect on the Laplace approximation.
{: .prompt-exercise}

> Argue why $\mathbb{P}(z) \propto e^{\frac{-z^4}{4}}$ will not get a good gaussian approximation using Laplace method. You can use desmos graphing calculator to visually see it. 
{: .prompt-exercise}

## MONTE CARLO

### IMPORTANCE SAMPLING

Laplace approximation helps us get a gaussian approximation of an intractable posterior distribution. But instead of trying to approximate the form of the posterior, why not answer the question about the posterior directly? Maybe we are interested in computing the expectation of the posterior, 
$\mathbb{E}[\mathbf{\mu}|\mathbf{x}]$. Over a large number of samples $N$ from the posterior, we can approximate the expectation as 
$$\approx \frac{1}{N}\sum_{n = 1}^{N} \mathbf{\mu}_n$$, where $\mathbf{\mu}_n \sim \mathbb{P}[\mathbf{\mu}|\mathbf{x}]$. We can extend the above, and may as well wish to compute $$\mathbb{E}_{\mathbf{\mu}|\mathbf{x}}[f(\mathbf{\mu})] \approx \frac{1}{N}\sum_{n = 1}^{N} f(\mathbf{\mu}_n)$$, which is expecation of some function $f$ over the posterior distribution. The approximation described here is possible only if sampling from the posterior is easy. This is not always the case!

> Sampling from a distribution is difficult when we do not have any existing efficient procedure to sample from the distribution. For example, Inverse CDF is a commonly used technique to sample from distributions. But finding an inverse may not be easy for all distributions (especially higher dimensions). Likewise, there may be other sampling techniques for which a particular distribution may not be a suitable candidate (for that technique to be applied on it). When a distribution fails to satisfy criteria of all existing sampling techniques, then we say that it is hard to sample from that distribution. 
{: .prompt-info}

If we are interested in only computing functions of posterior distributions like expectation, variance etc., then we can focus on distributions that are easy to sample from (called as proposal distributions), which may be fundamentally different from the posterior distribution in question. We can do the following:

$$
\begin{align}
\mathbb{E}_{\mathbf{\mu}|\mathbf{x}}[f(\mathbf{\mu})] &= \int_{\mathbf{\mu}}f(\mathbf{\mu})\mathbb{P}[\mathbf{\mu}|\mathbf{x}] \,d\mathbf{\mu} \\
                                                      &= \int_{\mathbf{\mu}}f(\mathbf{\mu})\frac{\mathbb{P}[\mathbf{\mu}|\mathbf{x}]}{\mathbb{\pi}[\mathbf{\mu}]}\mathbb{\pi}[\mathbf{\mu}] \,d\mathbf{\mu} \qquad \text{Multiply and divide by another distribution $\mathbb{\pi}$ such that $\mathbb{P}[\mathbf{\mu}|\mathbf{x}] > 0 \implies \mathbb{\pi}[\mathbf{\mu}] > 0 $}\\
                                                      &= \mathbb{E}_{\mathbb{\pi}}[f(\mathbf{\mu})\frac{\mathbb{P}[\mathbf{\mu}|\mathbf{x}]}{\mathbb{\pi}[\mathbf{\mu}]}]
\end{align}
$$

We can now approximate 
$$\mathbb{E}_{\mathbf{\mu}|\mathbf{x}}[f(\mathbf{\mu})]$$ as
$$\frac{1}{N}\sum_{n = 1}^{N}f(\mathbf{\mu_n})\underbrace{\frac{\mathbb{P}[\mathbf{\mu_n}|\mathbf{x}]}{\mathbb{\pi}[\mathbf{\mu_n}]}}_{w_n}$$ for large $N$, and $\mathbf{\mu}_n \sim \mathbb{\pi}[\mathbf{\mu}]$ instead of sampling from the posterior (considered hard usually; sampling from $\mathbb{\pi}$ should be easier and there should be efficient techniques to sample from it). $w_n$ is referred to as importance weights, and this whole technique is called importance sampling. 

> Note that there is no such thing as "sampling from the posterior" happening here. $\mathbf{\mu_n}$ is sampled from $\mathbb{\pi}$, and will not be called as a 'sample' in the traditional sense (as we are not sampling from the posterior we are interested in). Instead, $\mathbf{\mu_n}$ is called a 'particle'. This is where the theory of 'Particle Filtering' begins. But we will not be taking a look at it here.
{: .prompt-info}

```python
"""
CODE TO UNDERSTAND THE SIGNIFICANCE OF IMPORTANCE SAMPLING
"""
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
print(f"Mean of estimated expectation using true distribution: {mean_exp_true}")

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
```

```bash
$ python importance_sampling.py

Actual Expectation: 0.00019041059772826734
Mean of estimated expectation using gaussian proposal: 0.00019316710895392217
Variance of estimated expectation using gaussian proposal: 2.3927776240948028e-09
Variance of estimated expectation using true proposal: 2.20303668910995e-08
```

![Visualizing Importance Sampling](/assets/images/approximate_inference/importance_sampling.png)

The shell output reveals a few important things. Firstly, the actual expectation and the mean of the estimated expectations over 100 runs (100 different samples of size $N$ leads to 100 different estimates of the expectation) are very close to each other. This means that the estimated expectation obtained by sampling from the truncated gaussian proposal distribution (which is $\mathbb{\pi}$) is an unbiased estimate ($$\mathbb{E}_{\mathbb{P}}[f(x)]=\mathbb{E}_{\mathbb{\pi}}[\frac{1}{N} \sum_{n = 1}^{N} \frac{f(x)\mathbb{P}(x)}{\mathbb{\pi}(x)}] $$). Secondly, the variance of the estimated expectations across 100 runs by sampling particles from gaussian proposal is $10$x less than the variance of estimated expectations across these runs by sampling values from the mixture of beta distribution. 

From the figure, it can be seen that the function $f$ takes high values in regions of low density in the mixture of beta, and low values in regions of high density in the same distribution. Across different runs, we are most likely to sample from high density regions of mixture of beta, leading to low empirical average of $f(x)$ where $x$ is drawn from the mixture. In some runs, we may sample from the low density region, leading to high values of $f(x)$. The average within these runs will be significantly higher compared to the other runs. Intuitively, the variance of the empirical averages estimated across runs will be high if samples are taken from the mixture of beta. In case of the chosen proposal distribution in the code above, there is a significant decrease in the variance of the estimates. The $\mathbb{\pi}$ chosen in the code above is such that its high density regions surrounds the spot where $f(x)$ peaks. This choice leads to a significant reduction in variance of estimated empirical averages!

> There seems to be some sort of connection between the chosen $\mathbb{\pi}$ and $f$ (in other words, a clever $\mathbb{\pi}$ can be chosen by considering the nature of $f$). I have come across youtube videos and blogs that say that the chosen $\mathbb{\pi}$ must be high wherever 
>$|p(x)*f(x)|$ is high for the variance of the empirical averages to undergo a reduction! This is the case in the aforementioned code and plot as well. But the art of choosing the right $\mathbb{\pi}$ is not explored in this blog. Reference section will have appropriate resources to help the interested readers satiate their curiosity about this connection!
{: .prompt-info}

### METROPOLIS-HASTINGS

What if we could come up with a way to sample from the posterior distribution directly, despite it being tricky to do so? This achieves two things simultaneously: we would have samples from the intractable posterior, and also be in a position to get good estimates of expectation by finding 
$$\frac{1}{N} \sum_{n = 1}^{N} f(\mu_n)$$, 
where $\mu_n \sim \mathbb{P}(\mathbf{\mu}|\mathbf{x})$ 
and $N$ is very large. Metropolis-Hastings algorithm is a way to achieve the above. 

<div style="font-family:monospace;border:1.5px solid #000;border-radius:4px;overflow:hidden;max-width:680px;margin:1.5rem 0;background:#fff;color:#000">
  <div style="background:#f5f5f5;border-bottom:1.5px solid #000;padding:8px 16px;font-family:sans-serif;font-size:13px;font-weight:600;text-align:center;color:#000">
    Algorithm 1 — Metropolis–Hastings
  </div>
  <div style="padding:12px 16px 14px;font-size:13.5px;line-height:1.9;color:#000">
    <span style="display:block"><b>Objective:</b> generate realisations from a pdf <i>f<sub>x</sub>(x)</i></span>
    <span style="display:block"><b>Given:</b> <i>g(x) ∝ f<sub>x</sub>(x)</i></span>
    <span style="display:block"><b>Choose</b> proposal distribution, <i>π<sub>x</sub>(x | x<sub>j−1</sub>)</i></span>
    <span style="display:block"><b>Sample initial condition:</b> <i>x<sub>0</sub> ~ π<sub>x</sub>(x | 0)</i></span>
    <span style="display:block;margin-top:4px"><b>repeat for</b> <i>j = 1, 2, …, J</i><b>:</b></span>
    <hr style="border:none;border-top:0.5px solid #999;margin:6px 0">
    <span style="display:block;padding-left:2em">sample <i>x′ ~ π<sub>x</sub>(x | x<sub>j−1</sub>)</i></span>
    <span style="display:block;padding-left:2em;margin-top:4px">
      calculate Hastings ratio &nbsp;
      <i>A(x′, x<sub>j−1</sub>) =
        <span style="display:inline-flex;flex-direction:column;vertical-align:middle;text-align:center;line-height:1.4;color:#000">
          <span style="border-bottom:1px solid #000;padding:0 4px">g(x′) &nbsp; π<sub>x</sub>(x<sub>j−1</sub> | x′)</span>
          <span style="padding:0 4px">g(x<sub>j−1</sub>) &nbsp; π<sub>x</sub>(x′ | x<sub>j−1</sub>)</span>
        </span>
      </i>
    </span>
    <span style="display:block;padding-left:2em;margin-top:6px">sample from the uniform distribution, <i>u ~ 𝒰[0, 1]</i></span>
    <span style="display:block;padding-left:2em;margin-top:6px">
      set &nbsp;<i>x<sub>j</sub></i> =
      <span style="display:inline-flex;flex-direction:column;vertical-align:middle;border-left:1.5px solid #000;padding-left:6px;line-height:2.1;color:#000">
        <span><i>x′</i>, &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; if <i>A(x′, x<sub>j−1</sub>) ≥ u</i></span>
        <span><i>x<sub>j−1</sub></i>, &nbsp;&nbsp; otherwise</span>
      </span>
    </span>
    <hr style="border:none;border-top:0.5px solid #999;margin:6px 0">
    <span style="display:block"><b>end</b></span>
  </div>
</div>

Just like in Importance Sampling, we choose a proposal distribution and sample values from it. The twist here is we introduce dependence between the current sample and the previous sample. The sample drawn from the proposal doesn't automatically become a sample from the posterior. We instead call the current sample as a candidate sample $x'$. We condition the proposal distribution on the previous sample $x_{j-1}$ and sample candidate $x'$ from this conditional distribution. We compute the Hastings ratio ($A$) as mentioned in the algorithm. A value from the Uniform$(0,1)$ distribution is sampled, and is compared against the Hastings ratio. A uniform random variable takes a value less than the Hastings ratio with probability 
$\mathbb{P}(\{u \leq A(x', x_{j - 1})\}) = A(x', x_{j - 1})$. The Hastings ratio (which is a probability) helps in deciding if the candidate sample $x'$ must be accepted or not. With probability of $A(x', x_{j - 1})$, we retain candidate $x'$ as the current sample, and with probability $1 - A(x', x_{j - 1})$, we reject $x'$ and consider the previous sample $x_{j - 1}$ as the current sample. 

Upon a closer look, we can understand that a first-order dependence is being established above. The current sample is solely dependent on the previous sample. In other words, we are forming a first-order markov chain here. The random variable in each iteration (the sample) can take values from the domain defined for a given problem, and that forms a sequence of random variables when we consider a large number of iterations $J$. Depending on the problem, it can be a discrete or continuous state markov chain. For example, if the samples are drawn from a discrete proposal distribution (like Bernoulli), then we get a discrete-state markov chain. If they are drawn from a continuous proposal distribution (like Gaussian), then we get a continuous-state markov chain. The transition probabilites are defined by the proposal distribution. If the previous sample is 
$x_{j - 1}$, then the probability that the current sample will be 
$x'$ is defined as 
$$\mathbb{\pi}_{X_j}(X_{j} = x'|X_{j - 1} = x_{j - 1})$$. In a similar fashion, we can interpret $$\mathbb{\pi}_{X_j}(X_{j} = x_{j - 1}|X_{j - 1} = x')$$.

Interestingly, if the transition probabilities are symmetric 
($$\mathbb{\pi}_{X_j}(X_{j} = x'|X_{j - 1} = x_{j - 1}) = \mathbb{\pi}_{X_j}(X_{j} = x_{j - 1}|X_{j - 1} = x')$$), then we end up with a variant known as the Metropolis Algorithm. We can cancel the transition probabilities from the numerator and denominator. For example, when transition probability (or the proposal distribution) is considered a gaussian conditioned on the previous sample (as mean of the gaussian), then the transition probabilities are symmetric (In the exponent term, $(x' - x_{j - 1})^2 = (x_{j - 1} - x')^2$). Since both forward and backward directions have the same transition probability, all that matters is how likely the candidate or the previous sample is under the target distribution to make the decision as in the algorithm. 

Why does this algorithm guarantee getting samples from the target distribution? In each iteration, we have a probability distribution over values that the sample at that iteration can take. This distribution will vary across iterations for a brief period. After a point in time, the distribution no longer varies. It stays the same across iterations. This is called as the stationary distribution of the markov chain. All of the effort in the Metropolis-Hastings algorithm is directed towards ensuring that 
$f_X(x)$ becomes the stationary distribution of the markov chain. This is guaranteed by the balance equation 
$f(x')\mathbb{\pi}(x|x') = f(x)\mathbb{\pi}(x'|x)$, and ergodicity of the markov chain. The markov chain will attain a unique stationary distribution $f$ (which is the target distribution) due to the two conditions. Intuitively, what this means is that starting from $x'$ as a previous sample in an iteration, the probability with which we consider $x$ as current sample should be as good as the probability of starting from $x$ as a previous sample and choosing $x'$ as the current sample. By satisfying this balance, we are ensuring that the markov chain has the flexibility to come back to $x$ even if $x'$ is considered as the current sample (and vice-versa). In effect, the markov chain can freely explore the space. Observe the Hastings ratio and compare that with the balance equation ($f(x)$ is $kg(x)$ where $k$ is the normalizing constant of $f$). The ratio of transition probabilities plays the role of a correction factor to ensure balance is maintained. If $g(x')$ is more likely, a naive thought would be to choose $x'$ as the current sample (The density ratio 
$\frac{g(x')}{g(x)}$ will be high). But the possibility of not coming back to previous sample $x$ in future iterations of the algorithm is concerning. The ratio of transition probabilities rightly penalizes the naive thought by lowering the density ratio. We will not be making this discussion proof-heavy for now. I will be writing a blog post on markov chains sometime later. We can revisit this algorithm that time from a mathematical perspective.

It is now time to understand this algorithm via a simulation.

```python
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
var = [0.001, 0.05, 2, 8]           # Different variances of the truncated normal PDF that will be considered that will be considered
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
    plt.savefig(f'../../images/approximate_inference/hist_var_{i}.png')

    plt.figure()
    plt.plot(samples)       # Trace plot of the chain
    plt.xlabel('Iteration')
    plt.ylabel('Values')
    plt.title(f'MCMC Trace Plot for all iterations for variance = {i}')
    plt.savefig(f'../../images/approximate_inference/trace_var_{i}.png')

    plt.figure()
    plt.plot(running_mean, label='Running mean')            # Comparing the running mean of MCMC samples with the actual expectation
    plt.axvline(burn_in, color='red', linestyle='--', label='Burn-in cutoff')
    plt.axhline(true_mean, color='black', linestyle='--', label='True mean')
    plt.xlabel('Iteration')
    plt.ylabel('Running mean')
    plt.title(f'Running mean for variance = {i}')
    plt.legend()
    plt.savefig(f'../../images/approximate_inference/running_mean_var_{i}.png')

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
    plt.savefig(f'../../images/approximate_inference/acf_var_{i}.png')

plt.figure()
plt.plot(var, accept_rate, marker='s')
plt.xlabel('Variance of proposal distribution')
plt.ylabel('Acceptance Rate')
plt.title('Acceptance Rate for different variances of the proposal distribution')
plt.savefig('../../images/approximate_inference/ar.png')

plt.figure()
plt.plot(var, ess, marker='s')
plt.xlabel('Variance of proposal distribution')
plt.ylabel('Effective Sample Size')
plt.title('Effective Sample Size (ESS) for different variances of the proposal distribution')
plt.savefig('../../images/approximate_inference/ess.png')


print("\nSummary:")
for v, ar, es in zip(var, accept_rate, ess):
    print(f"var={v} | acceptance rate={ar:.1f}% | ESS={es:.0f}")

plt.tight_layout()
```

<style>
.mcmc-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 14px;
  margin: 1.5rem 0;
}

@media (min-width: 900px) {
  .mcmc-grid {
    grid-template-columns: repeat(4, 1fr);
  }
}

.mcmc-grid figure {
  margin: 0;
  border: 1px solid #ddd;
  border-radius: 8px;
  overflow: hidden;
  background: #fff;
}

.mcmc-grid img {
  width: 100%;
  display: block;
}

.mcmc-grid figcaption {
  padding: 8px;
  font-size: 0.82rem;
  text-align: center;
  color: #666;
}
</style>

<div class="mcmc-grid">

<figure>
  <img 
    src="/assets/images/approximate_inference/hist_var_0.001.png"
    alt="Histogram of MCMC samples for proposal variance 0.001">
  <figcaption>Histogram — var = 0.001</figcaption>
</figure>

<figure>
  <img 
    src="/assets/images/approximate_inference/trace_var_0.001.png"
    alt="Trace plot of Markov chain samples for proposal variance 0.001">
  <figcaption>Trace Plot — var = 0.001</figcaption>
</figure>

<figure>
  <img 
    src="/assets/images/approximate_inference/running_mean_var_0.001.png"
    alt="Running mean convergence plot for proposal variance 0.001">
  <figcaption>Running Mean — var = 0.001</figcaption>
</figure>

<figure>
  <img 
    src="/assets/images/approximate_inference/acf_var_0.001.png"
    alt="Autocorrelation factor plot for proposal variance 0.001">
  <figcaption>ACF — var = 0.001</figcaption>
</figure>

<figure>
  <img 
    src="/assets/images/approximate_inference/hist_var_0.05.png"
    alt="Histogram of MCMC samples for proposal variance 0.05">
  <figcaption>Histogram — var = 0.05</figcaption>
</figure>

<figure>
  <img 
    src="/assets/images/approximate_inference/trace_var_0.05.png"
    alt="Trace plot of Markov chain samples for proposal variance 0.05">
  <figcaption>Trace Plot — var = 0.05</figcaption>
</figure>

<figure>
  <img 
    src="/assets/images/approximate_inference/running_mean_var_0.05.png"
    alt="Running mean convergence plot for proposal variance 0.05">
  <figcaption>Running Mean — var = 0.05</figcaption>
</figure>

<figure>
  <img 
    src="/assets/images/approximate_inference/acf_var_0.05.png"
    alt="Autocorrelation factor plot for proposal variance 0.05">
  <figcaption>ACF — var = 0.05</figcaption>
</figure>

<figure>
  <img 
    src="/assets/images/approximate_inference/hist_var_2.png"
    alt="Histogram of MCMC samples for proposal variance 2">
  <figcaption>Histogram — var = 2</figcaption>
</figure>

<figure>
  <img 
    src="/assets/images/approximate_inference/trace_var_2.png"
    alt="Trace plot of Markov chain samples for proposal variance 2">
  <figcaption>Trace Plot — var = 2</figcaption>
</figure>

<figure>
  <img 
    src="/assets/images/approximate_inference/running_mean_var_2.png"
    alt="Running mean convergence plot for proposal variance 2">
  <figcaption>Running Mean — var = 2</figcaption>
</figure>

<figure>
  <img 
    src="/assets/images/approximate_inference/acf_var_2.png"
    alt="Autocorrelation factor plot for proposal variance 2">
  <figcaption>ACF — var = 2</figcaption>
</figure>

<figure>
  <img 
    src="/assets/images/approximate_inference/hist_var_8.png"
    alt="Histogram of MCMC samples for proposal variance 8">
  <figcaption>Histogram — var = 8</figcaption>
</figure>

<figure>
  <img 
    src="/assets/images/approximate_inference/trace_var_8.png"
    alt="Trace plot of Markov chain samples for proposal variance 8">
  <figcaption>Trace Plot — var = 8</figcaption>
</figure>

<figure>
  <img 
    src="/assets/images/approximate_inference/running_mean_var_8.png"
    alt="Running mean convergence plot for proposal variance 8">
  <figcaption>Running Mean — var = 8</figcaption>
</figure>

<figure>
  <img 
    src="/assets/images/approximate_inference/acf_var_8.png"
    alt="Autocorrelation factor plot for proposal variance 8">
  <figcaption>ACF — var = 8</figcaption>
</figure>

</div>
![Acceptance Rate for different variances of the proposal distribution](/assets/images/approximate_inference/ar.png)

![Effective Sample Size for different variances of the proposal distribution](/assets/images/approximate_inference/ess.png)

The code explores the behavior of the markov chain for different variances of the proposal distribution (which is a truncated gaussian in this case; it is symmetric). We are running the algorithm for $10000$ iterations. Before delving into an analysis, it is good to know what a burn-in period is. We cannot expect the markov chain to have attained a stationary distribution within the first few iterations of the algorithm. This means that the first few samples may not actually be from the target distribution. We tend to drop these initial samples. The amount of time taken by the chain to attain a stationary distribution is called the burn-in period. 

#### LOW VARIANCE CASE
If the variance is very low, then the markov chain progresses very slowly. If we start with sample $x$, there is a very high chance that the markov chain will continue crawling around this $x$ for a while (a long while perhaps!) before taking a leap into a different region. When the variance is $0.001$, the trace plot (plotting the MCMC sampled values) shows that the chain never explored the entire $[0,2]$ range. By taking the plotted histograms of the MCMC samples into account, we can see that the samples were more or less from around the first mode of the target distribution. As a result of this, the running mean of the samples are skewed towards the values in the lower part of the $[0,2]$ range. But the actual mean is far away from what could be computed with samples from these $10000$ iterations. 

Autocorrelation factor (ACF) computes the correlation between samples at different lags. By lag, we mean to say the gap between successive samples. This gap must be such that the correlation between samples is less. Ideally, we want to be able to take independent samples from the target distribution. But Independence is a very strong criterion to meet. So we settle with being able to take uncorrelated samples from the distribution. Intuitively, if most of the samples are correlated, then there is nothing much informative (about the space on which the distribution is defined) in those samples. This may be equivalent to one (or more) samples that contains all information needed about the $10000$ samples. For extremely low variance like $0.001$, the ACF is close to zero when we consider every $i^{th}$ sample where $i \geq 40$. This means that every $40^{th}$ sample reveals something informative about the space all the $10000$ samples are in, and it is more efficient to work with them than work with all $10000$ samples to arrive at the same conclusion on any analysis that is carried out. To quantify this intuition, we look at Effective Sample Size (ESS). It is expressed as 
$\frac{N}{1 + 2\sum_{k = 1}^{\infty} \rho_k}$. If the correlation remains consistently high between samples for different lags $k$, then ESS approaches $0$. If the correlation remains consistently low between samples for different lags $k$, then ESS approaches $N$. This can be mapped to the earlier explanation on the informativeness of the samples. For low variance case, the ESS will be low. This is because the samples remain crawling around the same region for a long time. There is a high chance that successive samples are significantly correlated. 

Since the variance is very low, the algorithm will accept many of the candidate samples. Given that the proposal distribution is symmetric, Hastings ratio will be close to $1$ most of the time (provided that the target distribution is smooth around the neighborhood of any arbitrary point). The acceptance rate will be very high for low variance case.

#### HIGH VARIANCE CASE 

As the variance becomes large, we can expect the candidates $x'$ (sampled from the proposal conditioned on $x_{j - 1}$) to take wild leaps to less expressive regions of the target distribution. If this happens, then the Hastings ratio will be significantly smaller than $1$, rejecting $x'$ most of the time. It implies that the markov chain will be stuck in the same sample value (previous sample) for sometime. This may not be clear from the image in the slide deck. Below are the zoomed in versions of the trace plot for both low and high variance case. In the low variance case, the previous sample was not repeated most of the times. Due to explanation given for the high variance case, we can see that the previous sample value is retained for sometime before taking a leap to a different sample. This implies that the acceptance rate for high variance will be less when compared to the low variance case.

![Zoomed in version of trace plot for low variance](/assets/images/approximate_inference/zoomed_var_0.001.png)

![Zoomed in version of trace plot for high variance](/assets/images/approximate_inference/zoomed_var_8.png)

The ACF approaches zero within a few lags than the low variance case. So, the ESS will be more here. With high variance, the chain has good enough flexibility to explore more (at the cost of getting a candidate rejected). This can be seen in the trace plot when variance is $8$. The chain also covers the entire range from $0$ to $2$. The histogram plotted from MCMC samples for this variance has covered all the three modes in the target distribution. We can expect the samples here to be more informative. That explains the high ESS. But the running mean takes a lot of iterations to come closer to the actual mean. With variance set to $2$, the running mean converges to the actual mean within $2000$ iterations. 

> With the above explanations as a reference, try to understand the behavior of the chain for variances $0.05$ and $2$ from the plots in the slide deck. Feel free to modify the code and observe the effects of the modifications!
{: .prompt-exercise}

Choosing the right parameters for the proposal distribution (and choosing the distribution itself) requires a multi-faceted analysis. We have covered a subset of the analysis tools commonly used to observe the chain and interpret it.

> The theory of markov chains has only been treated axiomatically here. For a more rigorous treatment, you can refer to books by Shiryaev on Probability.
{: .prompt-info}

### GIBBS SAMPLING

What if we do not want to get into the messy details of the proposal distribution for sampling? As seen in the previous section, the choice of the proposal distribution (and its parameters) influences the behavior of the chain. Gibbs Sampling provides a way to work around this problem. We have a joint posterior that is difficult to sample from. But, we know the full conditional distributions the joint distribution splits into. As an example, consider three random variables 
$\mu_1$, $\mu_2$, and $\mu_3$. It is hard to sample from 
$\mathbb{P}(\mu_1,\mu_2,\mu_3)$. We are given 
$\mathbb{P}(\mu_1|\mu_2,\mu_3)$, $\mathbb{P}(\mu_2|\mu_1,\mu_3)$, and $\mathbb{P}(\mu_3|\mu_1,\mu_2)$. These are called full conditional distributions - a conditional probability distribution of one variable with respect to all other variables, 
$\mathbb{P}(\mu_i|\mu_{\setminus i})$ where $\backslash i$ means everything other than $i$. The algorithm for Gibbs sampling is mentioned below: 

<div style="font-family:monospace;border:1.5px solid #000;border-radius:4px;overflow:hidden;max-width:680px;margin:1.5rem 0;background:#fff;color:#000">
  <div style="background:#f5f5f5;border-bottom:1.5px solid #000;padding:8px 16px;font-family:sans-serif;font-size:13px;font-weight:600;text-align:center;color:#000">
    Algorithm 2 — Gibbs Sampling
  </div>
  <div style="padding:12px 16px 14px;font-size:13.5px;line-height:1.9;color:#000">
    <span style="display:block"><b>1.</b> Initialize {<i>z<sub>i</sub> : i = 1, …, M</i>}</span>
    <span style="display:block;margin-top:4px"><b>2. repeat for</b> <i>τ = 1, …, T</i><b>:</b></span>
    <hr style="border:none;border-top:0.5px solid #999;margin:6px 0">
    <span style="display:block;padding-left:2em">
      sample &nbsp;<i>z<sub>1</sub><sup>(τ+1)</sup> ~ p(z<sub>1</sub> | z<sub>2</sub><sup>(τ)</sup>, z<sub>3</sub><sup>(τ)</sup>, …, z<sub>M</sub><sup>(τ)</sup>)</i>
    </span>
    <span style="display:block;padding-left:2em">
      sample &nbsp;<i>z<sub>2</sub><sup>(τ+1)</sup> ~ p(z<sub>2</sub> | z<sub>1</sub><sup>(τ+1)</sup>, z<sub>3</sub><sup>(τ)</sup>, …, z<sub>M</sub><sup>(τ)</sup>)</i>
    </span>
    <span style="display:block;padding-left:4em;color:#555">⋮</span>
    <span style="display:block;padding-left:2em">
      sample &nbsp;<i>z<sub>j</sub><sup>(τ+1)</sup> ~ p(z<sub>j</sub> | z<sub>1</sub><sup>(τ+1)</sup>, …, z<sub>j−1</sub><sup>(τ+1)</sup>, z<sub>j+1</sub><sup>(τ)</sup>, …, z<sub>M</sub><sup>(τ)</sup>)</i>
    </span>
    <span style="display:block;padding-left:4em;color:#555">⋮</span>
    <span style="display:block;padding-left:2em">
      sample &nbsp;<i>z<sub>M</sub><sup>(τ+1)</sup> ~ p(z<sub>M</sub> | z<sub>1</sub><sup>(τ+1)</sup>, z<sub>2</sub><sup>(τ+1)</sup>, …, z<sub>M−1</sub><sup>(τ+1)</sup>)</i>
    </span>
    <hr style="border:none;border-top:0.5px solid #999;margin:6px 0">
    <span style="display:block"><b>end</b></span>
  </div>
</div>

Within an iteration, consider each Gibbs sampling step. Before and after the $i^{th}$ step, the marginal 
$\mathbb{P}(z_{\setminus i})$ remains the same (or we can say invariant because the other values are fixed before and after the step). Since we have access to the full conditional distribution, 
$\mathbb{P}(z_i|z_{\setminus i})$ is also invariant. Within an iteration, we collect $M$ samples and consider it to be a tuple from the joint distribution. We can build a markov chain over these tuples across iterations. Gibbs sampling is considered as a special case of Metropolis-Hastings when we take the invariants established so far into consideration and compute acceptance ratio 
$$A(z^*, z)$$, where 
$$z^*$$ is the tuple of $M$ observations after a gibbs sampling step, and $z$ is the tuple before that. The full conditional distribution is the proposal distribution in Metropolis-Hastings algorithm. So,
$$\mathbb{\pi}(z^*|z) = \mathbb{P}(z_i^*|z_{\setminus i})$$

$$
\begin{align}
A(z*,z) &= \frac{\mathbb{P}(z^*)\mathbb{\pi}(z|z^*)}{\mathbb{P}(z)\mathbb{\pi}(z^{*}|z)} \\
        &= \frac{\mathbb{P}(z_{\setminus i}^*)\mathbb{P}(z_i^{*}|z_{\setminus i}^*)\mathbb{P}(z_i|z_{\setminus i}^*)}{\mathbb{P}(z_{\setminus i})\mathbb{P}(z_i|z_{\setminus i})\mathbb{P}(z_i^*|z_{\setminus i})} \label{eqnGibbs}\\
        &= 1 \qquad \text{(In $\eqref{eqnGibbs}$, we know from the invariance established earlier that $\mathbb{P}(z_{\setminus i}^*) = \mathbb{P}(z_{\setminus i})$. Also, $z_{\setminus i} = z_{\setminus i}^*$ as values other than $z_i$ remains the same before and after a step)}
\end{align}
$$

Each Gibbs sampling step within an iteration, when considered as a Metropolis-Hastings (MH) step, has an acceptance ratio of $1$. This implies that the samples drawn from the full conditional distributions are accepted almost surely as a MH step.

What guarantees that the tuples after several iterations of Gibbs sampling are from the joint distribution? Notice that the balance equation holds for each Gibbs sampling step. This is because the acceptance ratio (or Hastings ratio) is $1$ from the derivation above. Recall the connect between Hastings ratio and balance equations in the previous section. Let the tuple from end of iteration $i - 1$ be 
$z$. After iteration $i$, we get tuple 
$$z^{*}$$. If the balance equation holds for these tuples across iterations as well, then we can be sure that the stationary distribution of the markov chain (with tuples as states of the markov chain) will be the joint distribution (we need ergodicity as well for us to claim that the joint distribution is the unique stationary distribution, but we can skip that part for now). We will see a simple proof for the case of two random variables. It can be easily extended to the general case of $n$ random variables.
 
Let $z = (x,y)$ and 
$$z^{*} = (x^{*}, y^{*})$$. Within an iteration of Gibbs sampling, the forward traversal looks like $$(x,y) \rightarrow (x^{*},y) \rightarrow (x^{*},y^{*})$$, where $$(x^{*},y)$$ and $$(x^{*},y^{*})$$ are the result of sampling $$x^{*}$$ and $$y^{*}$$ from the two full conditional distributions $\mathbb{P}(x|y)$ and $$\mathbb{P}(y|x^{*})$$. The reversal traversal looks like $$(x^{*},y^{*}) \rightarrow (x^{*},y) \rightarrow (x,y)$$. 

$$
\begin{align}
\mathbb{P}(x,y)\mathbb{P}((x^{*},y)|(x,y)) = \mathbb{P}(x^{*},y)\mathbb{P}((x,y)|(x^{*},y)) \qquad \text{From balance equation for $1$ step of Gibbs sampling in the forward traversal $(x,y) \rightarrow (x^{*},y)$} \\
\mathbb{P}(x^{*},y)\mathbb{P}((x^{*},y^{*})|(x^{*},y)) = \mathbb{P}(x^{*},y^{*})\mathbb{P}((x^{*},y)|(x^{*},y^{*})) \qquad \text{From balance equation for $1$ step of Gibbs sampling in the forward traversal $(x^{*},y) \rightarrow (x^{*},y^{*})$} \\
\end{align}
$$

Multiply both the balance equations above:

$$
\begin{align}
\mathbb{P}(x,y)\mathbb{P}((x^{*},y)|(x,y))\cancel{\mathbb{P}(x^{*},y)}\mathbb{P}((x^{*},y^{*})|(x^{*},y)) = \cancel{\mathbb{P}(x^{*},y)}\mathbb{P}((x,y)|(x^{*},y))\mathbb{P}(x^{*},y^{*})\mathbb{P}((x^{*},y)|(x^{*},y^{*})) \\
\mathbb{P}(x,y)\underbrace{\mathbb{P}(x^{*}|y)\mathbb{P}(y^{*}|x^{*})}_{\text{Transition probability from $(x,y) \rightarrow (x^{*},y^{*})$ via Gibbs Sampling steps}} = \mathbb{P}(x^{*},y^{*})\underbrace{\mathbb{P}(y|x^{*})\mathbb{P}(x|y)}_{\text{Transition from $(x^{*},y^{*}) \rightarrow (x,y)$ via Gibbs Sampling steps}}
\end{align}
$$

Compare with the balance equation seen in the discussion on Metropolis-Hastings, and convince yourself that the joint distribution will be the stationary distribution of the markov chain.

Let us take a look at a simulation of this sampling approach to understand more about it.

```python
"""
GIBBS SAMPLING IN ACTION
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
N = 30                      # Total number of observations Y

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

plt.show()
```

```bash
$ python gibbs_sampling.py

True Mean: 4
True Variance: 0.5
```

![Contour vs samples from gibbs sampler](/assets/images/approximate_inference/contour_vs_samples_gibbs.png)

As is evident from the plot above, the samples are mostly from the high density region of the target joint distribution.

![Trace plot of mean from gibbs sampler](/assets/images/approximate_inference/mean_trace_gibbs.png)

![Trace plot of variance from gibbs sampler](/assets/images/approximate_inference/var_trace_gibbs.png)

The mean samples seem to have stabilized around a region within a few iterations. That region is close to the true mean as in the shell output. As with the mean, the variance seem to have stabilized around a region close to the true variance as in the shell output. Though the burn-in period is set to $500$ in the code, the chain seems to have already reached the region of interest (the region of high density in the joint distribution)

![Estimated marginal distribution of mean from gibbs sampler](/assets/images/approximate_inference/mean_marginal_gibbs.png)

![Estimated marginal distribution of variance from gibbs sampler](/assets/images/approximate_inference/var_marginal_gibbs.png)

Plotting just the first and second coordinates ($\mu$ and $\sigma^2$) separately gives an approximation of their posterior marginal distributions, 
$\mathbb{P}(\mu|Y)$ and $\mathbb{P}(\sigma^2|Y)$. Since Gibbs Sampler got samples from the posterior joint density, we get these posterior marginal distributions  for free (at the cost of having to know the full conditional distribution). We can carry out estimation problems like getting the expected posterior mean, $\mathbb{E}[\mu|Y]$. In this case, the expected posterior mean is close to the true mean, but the variance is slightly off. That can be corrected by increasing the number of observations $N$ in the code. We can also notice that the posterior and the prior of both the mean and variance have the same form. The posterior marginal distribution of the mean looks like a gaussian, and that of the variance is right-skewed just like the inverse gamma distribution.  

![Autocorrelation across iterations for the mean](/assets/images/approximate_inference/mu_acf.png)

![Autocorrelation across iterations for the mean](/assets/images/approximate_inference/var_acf.png)

The means sampled from the joint density does not have a lot of correlation amongst each other across several lags. So is the case with the sampled variances. 

In Metropolis-Hastings, we were worried about choosing the right proposal distribution. Though we overcame that trouble, we still need to be able to get the full conditional distributions to work with Gibbs Sampler!

## EPILOGUE

The concerning problem dealt with in this article is intractability of posterior distributions. Intuitively what we wanted was an approximation of the posterior that is as close as possible to the true posterior. Can we not look at this as an optimization problem? We can try looking at a bunch of probability densities (which are essentially functions with specific constraints) and compare that with the posterior via KL-Divergence. Any density that has the least possible KL-Divergence with the true posterior can be considered as an approximation of that posterior. This paradigm of viewing inference as an optimization problem is called Variational Inference, which will be the focus of our next blog! 

>MCMC is reported to work well on lower dimensions, and seems to be less efficient for higher dimensions. Variational Inference, powered by efficient optimization algorithms, is considered more efficient than MCMC in both lower and higher dimensions.
{: .prompt-info}


## REFERENCES

> Sayed, A. H. (2022). *Inference and Learning from Data: Inference*. Cambridge University Press.

> Bishop, C. M. (2007). Pattern Recognition and Machine Learning (Information Science and Statistics). Springer. ISBN: 0387310738

> https://rpubs.com/Deb2024/1347161 - Helped me understand the balance equation in Gibbs Sampling.

> https://youtu.be/C3p2wI4RAi8?si=0KyfouO8gUd0_zLd - Acted as a reference for my implementation.


## AI USAGE DISCLOSURE

None of the above content was written using AI. The codes were also written manually. ChatGPT was used as a reference aid for syntax lookup, but not for code generation. Claude helped in creating the slide deck that contains nearly 16 plots in it for the Metropolis-Hastings algorithm, and also created the algorithm box in which the pseudocodes are written. 