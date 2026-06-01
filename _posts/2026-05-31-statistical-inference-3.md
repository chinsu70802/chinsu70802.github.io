---

layout: post
title: "Variational Inference"
date: 2026-05-31
categories: [Statistics]
tags: [Statistics]
math: True
---

## PROLOGUE

In the previous article, we discussed ways to work around the intractability of posterior distributions. In the context of mean square inference, the problem was in the inability to compute the conditional mean estimate. As seen through an example on bayesian mixture of gaussians, evaluating the posterior took exponential time (particularly the "evidence" - denominator of the posterior function). Under the umbrella term of "Approximate Inference", one of the earliest solution was the Laplace method, where a gaussian approximation for the intractable posterior is identified via a second-order taylor series expansion. Given the problems it faces with multimodal and skewed distributions, we moved towards Monte Carlo techniques. Importance Sampling re-oriented the goal of the problem we started with. It asked the following question : "Why tinker with approximating the posterior when you can try approximating the function of the posterior itself (expectation for example)?". The trick was to introduce a proposal distribution $\mathbb{\pi}$ into the expectation, and write our intended expectation as a function of $\mathbb{\pi}$ instead of the posterior. $\mathbb{\pi}$ should be chosen such that it is easy to sample from, and also must be informed of the behavior of the function for which the expectation is computed. Later, we wanted to have the ability to draw samples from the intractable posterior and did not want to settle with just Importance Sampling. Two techniques - Metropolis Hastings and Gibbs Sampling - were discussed. The former warrants a proposal distribution that can ensure the construction of an ergodic markov chain of the samples whose unique stationary distribution is the intractable posterior. Over time, we end up sampling from the intractable posterior with good guarantees. With these samples, any function of the posterior can be approximated empirically. Gibbs Sampling is a special case of Metropolis Hastings that facilitates sampling from a joint posterior. Though we don't need to worry about the proposal distribution, there is still a need to know full conditional distributions of hidden variables. It has been observed that these MCMC techniques do not scale well at higher dimensions, and larger datasets. Is there a way to resolve this?

The Laplace method functioned in a restricted setting - any posterior was approximated as a gaussian alone. What if a bit more flexibility is introduced here? We instead allow the approximation of the intractable posterior to come from a family of densities $\mathcal{D}$. Since we want the approximated density to be as close as possible to the posterior, a "distance" metric (KL-Divergence in this case) can be used to compute the closeness of the former to the latter. Denoting $\mathbf{q^*}$ as the best approximation of the posterior from $\mathcal{D}$, $\mathbf{p}$ as the intractable posterior, and $x$ as the observation, we have the objective as : 

$$
\begin{align}
q^{*}(z) = \arg \min_{q(.) \in \mathcal{D}} \mathbf{KL}(q(z)||p(z|x)) \label{obj}
\end{align}
$$

Notice that the KL-Divergence is a functional — it is a function that takes two functions (probability densities) as input. Our objective is to vary $\mathbf{q}$ so as to minimize the KL-Divergence. We can use the calculus of variations (ordinary calculus extended to functionals) to perform this minimization. This paradigm is called "variational" because the optimization is carried out over functions (probability distributions). Since we use this paradigm to make inference easier by approximating an intractable posterior distribution, the method is called Variational Inference.

In comparison to MCMC techniques where the sole focus was on sampling, variational inference views inference (the part where intractable posterior has to be approximated to enable inferencing) as an optimization problem. Given that efficient algorithms exist for optimization, variational inference is expected to be much faster and scalable than MCMC in many cases. But it cannot give good guarantees like MCMC - we are just finding the density closest to the intractable posterior! 

> Variational Inference also falls under the broad family of "Approximate Inference". 
{: .prompt-info}

## EVIDENCE LOWER BOUND (ELBO)

Continuing from $\eqref{obj}$, we can rewrite KL-Divergence as : 

$$
\begin{align}
\mathbf{KL}(q(z)||p(z|x)) &= \mathbb{E}_{z \sim q}[\log\frac{q(z)}{p(z|x)}] \\
                 &= \mathbb{E}_{z \sim q}[\log q(z)] - \mathbb{E}_{z \sim q}[\log p(z|x)] \\
                 &= \mathbb{E}_{z \sim q}[\log q(z)] - \mathbb{E}_{z \sim q}[\log p(z,x)] + \mathbb{E}_{z \sim q}[\log p(x)] \qquad \text{(Expanding $p(z|x)$ as ratio of joint density of $z$ and $x$, and marginal of $x$)} \\
                 &= \mathbb{E}_{z \sim q}[\log q(z)] - \mathbb{E}_{z \sim q}[\log p(z,x)] + \log p(x) \qquad \text{($\log p(x)$ is a constant in the expectation over $z$)} \\
\underbrace{-(\mathbb{E}_{z \sim q}[\log q(z)] - \mathbb{E}_{z \sim q}[\log p(z,x)])}_{\text{ELBO(q)}} &= -\mathbf{KL}(q(z)||p(z|x)) + \log p(x) \label{elbo} \qquad \text{(Swapping places of KL-Divergence and -$\mathbf{ELBO}(q)$ in the previos equation)}\\
\end{align}
$$

The term to the left in $\eqref{elbo}$ is called Evidence Lower Bound (ELBO). It is a function that takes in input as the function $q$. The reason why it is named so comes from $\eqref{elbo}$ itself. As KL-Divergence is always non-negative, we can see that $\mathbf{ELBO}(q) \leq \log p(x)$. 

In the objective seen in $\eqref{obj}$, we wanted to minimize the KL-Divergence. Doing so is equivalent to maximizing the $\mathbf{ELBO}$ in $\eqref{elbo}$ ($\log p(x)$ is a constant as the optimization is with respect to $q$). 

Let us a take a close look at $\mathbf{ELBO}(q)$:

$$
\begin{align}
\mathbf{ELBO}(q) &= \mathbb{E}_{z \sim q}[\log p(x,z)] - \mathbb{E}_{z \sim q}[\log q(z)] \\
                 &= \mathbb{E}_{z \sim q}[\log p(x|z) + \log p(z)] - \mathbb{E}_{z \sim q}[\log q(z)] \qquad \text{(Writing joint as product of likelihood and marginal)} \\
                 &= \mathbb{E}_{z \sim q}[\log p(x|z)] + \mathbb{E}_{z \sim q}[\log p(z)] - \mathbb{E}_{z \sim q}[\log q(z)] \\
                 &= \mathbb{E}_{z \sim q}[\log p(x|z)] - \underbrace{\mathbb{E}_{z \sim q}[\log \frac{q(z)}{p(z)}]}_{\text{$\mathbf{KL}(q(z)||p(z))$}} \label{lik_prior}\\
\end{align}
$$

In order to maximize $\mathbf{ELBO}(q)$ with respect to $q$, we need to maximize the first term and minimize the second term in $\eqref{lik_prior}$. The first term expresses the expected likelihood. Since we are maximizing it, $q$ places more density on $z$ that fit more to the observed data $x$. The second term is the KL-Divergence between $q(z)$ and the prior $p(z)$. Minimizing this means wanting the variational distribution ($q$) to not deviate much from the the prior on the latent variables unless supported by observed data $x$. If there exists a certain $q$ that fits the data well, but fails to match the prior $p(z)$, then the $\mathbf{ELBO}(q)$ is penalized according to the second term. We may not consider such candidates. Equation $\eqref{lik_prior}$ illustrates that the optimization algorithm must find the right $q$ on $z$ that can fit the data well while being reasonably close to the prior. 

It is now clear that we will be maximizing the $\mathbf{ELBO}(q)$ with respect to $q$. But $q \in \mathcal{D}$. From what family of densities $\mathcal{D}$ does $q$ come from?

## MEAN-FIELD VARIATIONAL FAMILY

In this article, we will be working with the mean-field variational family. The nature of any density $q$ in this family is such that:

$$
\begin{align}
q(z) = \prod_{n = 1}^{M} q_{j}(z_j) \qquad \text{Consider $z$ to be a vector of $M$ random variables $z_1, z_2, \cdots, z_M$}
\end{align}
$$

In other words, the variational distribution $q$ assumes that latent variables are independent of each other. Each $z_j$ is governed by its own distinct variational density $q_j$. This family makes derivations significantly easier, but may not be a practical choice. 

If dependencies are introduced between latent variables, then the family of densities are dubbed as the structured mean-field variational family. Variational inference methods that use structured mean-field variational family are collectively referred to as structured variational inference. I guess it will be better to discuss the latter in an applied setting (which I will be doing in my next blog article related to an application from speech processing). 

We now have everything setup to begin the optimization procedure!

> A few definitions: Variational density $q(z)$ refers to the joint density of all latent variables. Variational factor $q_i(z_i)$ refers to the distribution followed by a particular latent variable $z_i$.
{: .prompt-info}

## COORDINATE-ASCENT VARIATIONAL INFERENCE (CAVI)

<div style="font-family:monospace;border:1.5px solid #000;border-radius:4px;overflow:hidden;max-width:680px;margin:1.5rem 0;background:#fff;color:#000">
  <div style="background:#f5f5f5;border-bottom:1.5px solid #000;padding:8px 16px;font-family:sans-serif;font-size:13px;font-weight:600;text-align:center;color:#000">
    Algorithm 1 — Coordinate Ascent Variational Inference (CAVI)
  </div>
  <div style="padding:12px 16px 14px;font-size:13.5px;line-height:1.9;color:#000">
    <span style="display:block"><b>Input:</b> A model <i>p(<b>x</b>, <b>z</b>)</i>, a data set <b>x</b></span>
    <span style="display:block"><b>Output:</b> A variational density <i>q(<b>z</b>) = ∏<sub>j=1</sub><sup>m</sup> q<sub>j</sub>(z<sub>j</sub>)</i></span>
    <span style="display:block"><b>Initialize:</b> Variational factors <i>q<sub>j</sub>(z<sub>j</sub>)</i></span>
    <hr style="border:none;border-top:0.5px solid #999;margin:6px 0">
    <span style="display:block"><b>while</b> <i>the ELBO has not converged</i> <b>do</b></span>
    <span style="display:block;padding-left:2em"><b>for</b> <i>j ∈ {1, …, m}</i> <b>do</b></span>
    <span style="display:block;padding-left:4em">
      Set &nbsp;<i>q<sub>j</sub>(z<sub>j</sub>) ∝ exp{𝔼<sub>−j</sub>[log p(z<sub>j</sub> | <b>z</b><sub>−j</sub>, <b>x</b>)]}</i>
    </span>
    <span style="display:block;padding-left:2em"><b>end</b></span>
    <span style="display:block;padding-left:2em">
      Compute &nbsp;ELBO(<i>q</i>) = 𝔼[log <i>p</i>(<b>z</b>, <b>x</b>)] − 𝔼[log <i>q</i>(<b>z</b>)]
    </span>
    <hr style="border:none;border-top:0.5px solid #999;margin:6px 0">
    <span style="display:block"><b>end</b></span>
    <span style="display:block;margin-top:4px"><b>return</b> <i>q</i>(<b>z</b>)</span>
  </div>
</div>

In the algorithm above, the notation $z_{-j}$ means every latent variable other than $z_j$. 
$$\mathbb{E}_{-j}$$ means expectation over every other variable except 
$z_j$ with respect to the variational density 
$$q(z_1,z_2,\cdots,z_{j-1},z_{j+1},\cdots,z_M)$$.

With the mean-field assumption, the algorithm aids in learning the variational factor $q_j$ for each $z_j$. The steps will make sense once we see how the optimal variational factor for $z_j$ ($q_j^{*}(z_j)$) is obtained in the algorithm. 

Let us try expressing $ELBO(q)$ in terms of $q_j$ alone. Consider all other variables other than $z_j$ (and their variational factors) to be constants.

$$
\begin{align}
\mathbf{ELBO}(q) &= \mathbb{E}_{z \sim q}[\log p(x,z)] - \mathbb{E}_{z \sim q}[\log q(z)] \\
                 &= \mathbb{E}_{z_{j},z_{-j} \sim q}[\log p(x,z_j,z_{-j})] - \mathbb{E}_{z \sim q}[\log \prod_{j = 1}^{M} q_{j}(z_j)] \qquad \text{(Using the mean field assumption imposed on the variational density)} \\
                 &= \mathbb{E}_{z_j}[\mathbb{E}_{z_{-j}}[\log p(x,z_j,z_{-j})]] - \mathbb{E}_{z \sim q}[\sum_{j = 1}^{M} \log q_{j}(z_j)] \qquad \text{(Applying the law of iterated expectation for the first term)} \\
                 &= \mathbb{E}_{j}[\mathbb{E}_{-j}[\log p(x, z_j, z_{-j})]] - \sum_{j = 1}^{M} \mathbb{E}_{z \sim q}[\log q_{j}(z_j)] \qquad \text{(To avoid clutter in the expectation subscript, $z_j \rightarrow j$ and $z_{-j} \rightarrow -j$)} \\
                 &= \mathbb{E}_{j}[\mathbb{E}_{-j}[\log p(x, z_j, z_{-j})]] - \mathbb{E}_{j}[\log q_{j}(z_j)] + const \qquad \text{(As other variational factors and latent variables except $q_j$ and $z_j$ are constant)} \\
                 &= \mathbb{E}_{j}[\log \exp \mathbb{E}_{-j}[\log p(x, z_j, z_{-j})]] - \mathbb{E}_{j}[\log q_{j}(z_j)] + const \qquad \text{(Using the fact that $\log \exp x = x$)} \\
\mathbf{ELBO}(q_j) &= - \underbrace{\mathbb{E}_{j}[\log \frac{q_{j}(z_j)}{\exp \mathbb{E}_{-j}[\log p(x, z_j, z_{-j})]}]}_{\text{$\mathbf{KL}(q_{j}(z_j)||\exp \mathbb{E}_{-j}[\log p(x, z_j, z_{-j})])$}} + const
\end{align}
$$

To maximize the $$\mathbf{ELBO}(q_j)$$, we need to minimize the KL-Divergence between $$q_{j}(z_j)$$ and $$\exp \mathbb{E}_{-j}[\log p(x, z_j, z_{-j})]$$.
For that, we need to set $$q_{j}(z_j) = \exp \mathbb{E}_{-j}[\log p(x, z_j, z_{-j})] \propto \exp \mathbb{E}_{-j}[\log p(z_j|x,z_{-j})]$$. As $\mathbf{ELBO}(q_j)$ gets maximized at this value, we can say that the optimal variational factor for $z_j$ is $$q_j^{*}(z_j) = \exp \mathbb{E}_{-j}[\log p(x, z_j, z_{-j})] \propto \exp \mathbb{E}_{-j}[\log p(z_j|x,z_{-j})]$$. 

For the $M$ latent variables, we individually get the optimal variation factors, compute $q(z)$ by multiplying the $M$ individual variational factors, and then compute the $\mathbf{ELBO}(q)$. Since the $\mathbf{ELBO}(q)$ is upper bounded by $\log p(x)$, and also the fact that the individual optimal variational factors ensures that the $\mathbf{ELBO}(q)$ increases (or stays same) across iterations, we can say that $\mathbf{ELBO}(q)$ will converge after several iterations.    

## EXAMPLE : BAYESIAN MIXTURE OF GAUSSIANS

Recall the generative model for the bayesian mixture of gaussians discussed in the previous article:

$$
\begin{align}
&\mu_k \sim \mathcal{N}(0, \sigma^2) \qquad k = 1, 2, \cdots, M \label{eqn1}\\
&c_i \sim Cat(\frac{1}{M},\frac{1}{M},\cdots,\frac{1}{M}), \qquad i = 1, 2, \cdots, N \label{eqn2}\\
&x_i \sim \mathcal{N}(c_{i}^T\mathbf{\mu}, 1), \qquad i = 1, 2, \cdots, N  \label{eqn3}
\end{align}
$$

We saw that the evidence 
$p(x)$ is intractable as it contains exponential number of terms. Now, we will be using variational inference to get an approximation for the joint posterior 
$p(\mathbf{\mu},\mathbf{c}|\mathbf{x})$.

We can start with setting up the variational density $q(\mathbf{\mu},\mathbf{c})$ for this by following the mean-field assumption:

$$
\begin{align}
q(\mathbf{\mu},\mathbf{c}) = \prod_{k = 1}^M q_k(\mu_k) \prod_{i = 1}^N q_i(c_i) \\
\end{align}
$$

Assume that $q_k(\mu_k) = \mathcal{N}(m_k, \sigma_{k}^2)$ and 
$q_i(c_i) = \mathrm{Cat}(\phi_i)$, where $\phi_i$ is a $M$-dimensional vector that contains probability of observation $x_i$ belonging to the $m^{th}$ cluster (denoted as $q_i(c_i = m) = \phi_{im}$ for each valid $m$).

The $\mathbf{ELBO}(q)$ will look like:

$$
\begin{align}
\mathbf{ELBO}(\mathbf{\mu},\mathbf{c}) &= \mathbb{E}[\log p(\mathbf{x},\mathbf{\mu},\mathbf{c})] - \mathbb{E}[\log q(\mathbf{\mu},\mathbf{c})] \\
                                       &= \mathbb{E}[\log p(\mathbf{\mu},\mathbf{c}) + \log p(\mathbf{x}|\mathbf{\mu},\mathbf{c})] - \mathbb{E}[\log q(\mathbf{\mu},\mathbf{c})] \\
                                       &= \mathbb{E}[\log p(\mathbf{\mu}) + \log p(\mathbf{c}) + \log p(\mathbf{x}|\mathbf{\mu},\mathbf{c})] - \mathbb{E}[\log q(\mathbf{\mu}) + \log q(\mathbf{c})] \\
                                       &= \mathbb{E}_{q_{\mathbf{\mu}}}[\log p(\mathbf{\mu})] + \mathbf{E}_{q_{\mathbf{c}}}[\log p(\mathbf{c})] + \mathbb{E}_{q_{\mathbf{\mu}},q_{\mathbf{c}}}[\log p(\mathbf{x}|\mathbf{\mu},\mathbf{c})] - \mathbb{E}_{q_{\mathbf{\mu}}}[\log q(\mathbf{\mu})] - \mathbb{E}_{q_{\mathbf{c}}}[\log q(\mathbf{c})] \\
\end{align}
$$

Let us express the $\mathbf{ELBO}$ in terms of $\mu_k$ alone (Consider all the other variational factors and its parameters as constants):

$$
\begin{align}
\mathbf{ELBO}(\mu_k)  &= \mathbb{E}_{q_{\mu_k}}[\log p(\mu_k)] + \mathbb{E}_{q_{\mu_k}}[\mathbb{E}_{q_{\mathbf{c}}}[\log p(\mathbf{x}|\mu_k)]] - \mathbb{E}_{q_{\mu_k}}[\log q(\mu_k)] \qquad \text{(In the second term, everything other than $q(\mu_k)$ is a constant. Since we already know the component of mean for all observations is $k$ as in the ELBO input, we can remove conditioning on $\mathbf{c}$)}\\
                      &= \mathbb{E}_{q_{\mu_k}}[\log p(\mu_k)] + \mathbb{E}_{q_{\mu_k}}[\mathbb{E}_{q_{\mathbf{c}}}[\sum_{i = 1}^N \log p(x_i|\mu_k)]] - \mathbb{E}_{q_{\mu_k}}[\log q(\mu_k)] \\
                      &= \mathbb{E}_{q_{\mu_k}}[\log p(\mu_k)] + \mathbb{E}_{q_{\mu_k}}[\sum_{i = 1}^N \mathbb{E}_{q_{\mathbf{c}}}[\log p(x_i|\mu_k)]] - \mathbb{E}_{q_{\mu_k}}[\log q(\mu_k)] \\
                      &= \mathbb{E}_{q_{\mu_k}}[\log p(\mu_k)] + \mathbb{E}_{q_{\mu_k}}[\sum_{i = 1}^N q(c_i =  k)\log p(x_i|\mu_k)] - \mathbb{E}_{q_{\mu_k}}[\log q(\mu_k)] \\
                      &= \mathbb{E}_{q_{\mu_k}}[\log p(\mu_k)] + \mathbb{E}_{q_{\mu_k}}[\sum_{i = 1}^N \phi_{im} \log p(x_i|\mu_k)] - \mathbb{E}_{q_{\mu_k}}[\log q(\mu_k)] \\
                      &= - \mathbb{E}_{q_{\mu_k}}[\log\frac{q(\mu_k)}{\exp (\log p(\mu_k) + \sum_{i = 1}^N \phi_{im} \log p(x_i|\mu_k))}] \\
q_k^{*}(\mu_k) &\propto \exp (\log p(\mu_k) + \sum_{i = 1}^N \phi_{im} \log p(x_i|\mu_k))
\end{align}
$$

> Continue from the last step in the previous derivation and try getting the parameters for the variational factor $q_k(\mu_k)$. You know $p(\mu_k)$ and $p(x_i|\mu_k)$ from the bayesian model. For verifying the final result obtained, you can refer to the python code given below.
{: .prompt-exercise}

> Keeping the above derivation for ELBO as reference, try expressing ELBO in terms of the variational factors for the cluster assigment $q_i(c_i)$ and obtain the parameters for the same. Though not mandatory to solve, you can write $p(x_i|\mathbf{\mu},c_i) = \prod_{k = 1}^K p(x_i|\mu_k)^{c_{ik}}$ for ease of derivation. You can refer to the code below to verify your final result/
{: .prompt-exercise}
```python
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

elbo_history = [] # Tracking ELBO values for the purpose of plotting

while ((abs(elbo - elbo_prev) >= 1e-3) and (iters < max_iters)): # Coordinate Ascent Variational Inference
    phi_i_list = []

    for i in range(N):
        phi_i_logit = variational_means*x[i] - ((variational_var + (variational_means)**2)/2)
        phi_i_logit -= np.max(phi_i_logit)
        phi_i = np.exp(phi_i_logit)
        phi_i /= phi_i.sum()   # Final answer for the second exercise
        phi_i_list.append(phi_i)
    
    variational_cat = np.array(phi_i_list)
    
    for k in range(K):
        variational_means[k] = np.sum(variational_cat[:,k] * x)
        variational_means[k] /= ((1/var_mu) + variational_cat[:, k].sum()) # Final answer for the first exercise (mean parameter)

        variational_var[k] = 1
        variational_var[k] /= ((1/var_mu) + variational_cat[:, k].sum()) # Final answer for the first exercise (variance parameter)
    
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
```

![Data histogram vs full mixture from CAVI](/assets/images/variational_inference/obs_vs_variational.png)

CAVI seems to have done a good job in fitting well to the data. The underlying model here is simple due to conjugacy (the prior and the posterior are from the same family of distribution)

![ELBO plot](/assets/images/variational_inference/elbo_plot.png)

As argued in the earlier derivation, the ELBO is increasing across iterations. 

![Comparison of true posterior density with variational factors](/assets/images/variational_inference/posterior_mean_comp.png)

Here, the fit with the true posteriors is almost perfect. This is specific to this model. The true posterior over the means happens to factorize, so the mean-field independence assumption is fine enough. The moment the true posterior has correlation between latent variables, mean-field can no longer capture it, and the variational approximation will deviate from the true posterior. The visualizations for this case can be reserved for later, and kept as a separate blog post. I will have to create a different generative model and that will take a bit more time.

## EPILOGUE

This marks the end of the primer on inference. With this background, we can comfortably read research papers that apply statistical concepts seen so far. In sometime, I will be writing an article on a research paper that applies variational inference for acoustic unit discovery. 

## REFERENCES

> Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). Variational Inference: A Review for Statisticians. Journal of the American Statistical Association, 112(518), 859–877. https://doi.org/10.1080/01621459.2017.1285773

> Sayed, A. H. (2022). *Inference and Learning from Data: Inference*. Cambridge University Press.

## AI USAGE DISCLOSURE  

None of the textual content (with the exception of the pseudocode) was written using AI. The codes were written manually for most of the part (except for a few places where proper credit has been given to the bot used). ChatGPT was used as a reference aid for syntax lookup. Claude aided in creating the box for writing algorithms, and also wrote the pseudocodes by referring to the aforementioned review paper. 