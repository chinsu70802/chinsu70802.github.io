---

layout: post
title: "Expectation Maximization"
date: 2026-06-27
categories: [Statistics]
tags: [Statistics]
math: True
---

## PROLOGUE

Recall the noisy-channel model of speech recongition discussed as an example to motivate the act of inference [Mean Square Inference](https://chinsu70802.github.io/posts/statistical-inference-1/). In that example, we described a world where people cannot hear what others say, but can see the acoustic waveform of the word uttered. The best possible guess of the word was mentioned as 
$\arg\max\limits_{x}p(x \| y) = \arg\max\limits_{x} p(y \| x) \cdot p(x)$, where $x$ and $y$ refer to the word and the observed waveform respectively. Let the waveform correspond to the word 'feign' (a common word that means to pretend). The pronounciation of this word is similar to 'fane' and 'fain' (both of which are old english words that mean temple and happy respectively). Using prior knowledge (the fact that old english words are rarely used in conversation), it is easy to eliminate 'fane' and 'fain' as $p(x)$ is going to be very less for these words. Hence, estimated $x$ will be 'feign'. This estimate is called the maximum a-posteriori (MAP) estimate. What if people do not form any prior belief (or consider anything and everything to be equally likely apriori) in this world? The best guess now becomes $\arg\max\limits_{x} p(y \| x)$ (The conditional probability in the objective is called likelihood). They need to rely on the observation $y$ to infer $x$. It is possible that 'fain' best explains $y$ compared to others. Hence, estimated $x$ is 'fain'. This estimate of $x$ is called as the maximum likelihood (ML) estimate. 

Notice that there can be a difference between ML and MAP estimates. This was a very high-level example. So, it may not be possible to fully appreciate it. Let us turn to a simple mathematical example surrounding coin tosses. We observe a coin with bias $k$ (the probability of heads is $k$) being tossed $n$ times independently. The $n$ observations are denoted as $y$ (which is the sequence 
$y_1, y_2, \cdots, y_n$). We are now in the world where prior belief on anything is a crime. If we observe $n$ heads at a stretch, the likelihood 
$p(y|k) = \prod_{i = 1}^{n} p(y_i|k)$. The conditioning on bias $k$ is universal across all independent tosses as the coin is the same across trials. As $p(y_i|k) = k$ considering each trial as a bernoulli, $p(y|k) = k^n$. As $k \in [0,1]$, $\arg\max\limits_{k \in [0,1]} p(y \| k) = \arg\max\limits_{k \in [0,1]} k^n$. The likelihood gets maximized when $k$ is $1$. Based on the observation alone, and with no prior belief, we made an extreme statement that the coin is always biased towards heads using the ML estimate. But such coins rarely exist in reality (unless we are talking about doubled-headed coins, a coin that has heads on both sides). If the world now allows people to have prior beliefs, we can consider a beta prior on $k$. For example, let the prior be $p(k;2,2) = \frac{\Gamma(4)}{\Gamma(2)\Gamma(2)} k(1-k)$. 

$$
\begin{align}
\arg\max\limits_{k \in [0,1]}p(k | y) &= \arg\max\limits_{k \in [0,1]} p(y | k) \cdot p(k;2,2) \\
                                       &= \arg\max\limits_{k \in [0,1]} k^n \frac{\Gamma(4)}{\Gamma(2)\Gamma(2)} k(1-k) \\
                                       &= \arg\max\limits_{k \in [0,1]} k^{n + 1} (1 - k) \qquad \text{I(gnoring the constants)}
\end{align}
$$

Differentiating the above objective with respect to $k$ and setting the result to $0$ yields the best possible guess of the bias as $\frac{n + 1}{n + 2}$. This estimate is not as extreme as $1$, and may be a bit more realistic. 

> More generally, consider the prior to be a $\beta(a,b)$. Derive MAP estimate for it and compare with the ML estimate.
{: .prompt-exercise}

When the world allows enforcement of a prior belief on the hidden factor, we call that world as Bayesian. The estimation or inference performed by people (the citizens are called bayesians) in this bayesian world is called Bayesian Inference. We have been dealing with this in the previous blog posts. 

In this blog post, we enter into the world with no prior beliefs. People living in this world are called empiricists/frequentists (we will be called that for the duration of this blog), and any sort of estimation or inference done by the people in this world is called Frequentist Inference. They will have to do several rounds of coin toss to understand the uncertainity surrounding the best estimate $k$ (They infer based on experiments and observations, hence the name empiricist). In one round of $n$ tosses, they may have seen all heads and estimated $k$ according to it. In another round, they may see only a few heads and a lot of tails. Based on this, their best estimate will change accordingly (mostly $k$ reduces from its earlier value of $1$). Their best estimates can have significant variance across rounds when $n$ is small. The bayesians, on the other hand, often have reduced variance in their estimates when compared with frequentists for small $n$. Their prior aids in making an informed choice, pushing the estimates towards values deemed plausible by it. If repeated experiments is the bottleneck that empiricists face, the design of a good prior is a difficulty that the bayesians face. If the real world coin is biased towards heads, and the bayesians design a prior that favors tails, their estimation will not be good. But as more and more observations are made, the bad design of the prior is forgiven by the data, and the MAP estimate acts more like ML estimate in this example. In conclusion, frequentists may have to become bayesians when observation is scarce (we saw that ML estimates tend to be extreme at times), and bayesians may become frequentists once a lot of observations are available (MAP and ML estimates are similar most of the times in presence of more observations). As a final touch for the narrative around bayesians and frequentists, we realise that any citizen from either world must hold visas to both the worlds!  

```python
"""
VARIANCE OF MAXIMUM LIKELIHOOD AND MAXIMUM A-POSTERIORI ESTIMATES FOR COIN TOSSING EXPERIMENT   
"""
import numpy as np
import matplotlib.pyplot as plt

rounds = 10000          # Number of rounds of coin toss
n = 100                 # Number of tosses with a round
p = 0.7                 # Bias of the coin
ml_estimate_across_rounds = [] 
map_estimate_across_rounds = []
a = 3                   # alpha parameter of the beta distribution                
b = 2                   # beta parameter of the beta distribution

for i in range(rounds):
    obs = np.random.choice([1, 0], size = n, p = [p, 1 - p])        # Generating n-length sequence of observations
    n_heads = np.sum(obs)                                           # Number of heads within a round
    ml_estimate = np.mean(obs)                                      # The derived ML estimate
    ml_estimate_across_rounds.append(ml_estimate)
    map_estimate = (n_heads + a - 1) / (a + b + n - 2)              # The derived MAP estimate
    map_estimate_across_rounds.append(map_estimate)

ml_estimate_across_rounds = np.array(ml_estimate_across_rounds)
map_estimate_across_rounds = np.array(map_estimate_across_rounds)
var_ml_estimate = np.var(ml_estimate_across_rounds)                 # Getting variance of ML estimates across rounds
var_map_estimate = np.var(map_estimate_across_rounds)               # Getting variance of MAP estimates across rounds

print("Variance of ML estimate: ", var_ml_estimate)
print("Variance of MAP estimate: ", var_map_estimate)
```

> To understand the intuition explained in the previous paragraph, tweak the hyperparameters in the python code above and relate your observations to the arguments encountered in the explanations.
{: .prompt-exercise}

> (Optional) Mathematically derive the variance of ML and MAP estimate for the above scenario. You will unravel a relationship between them that explains the variance reduction faced by MAP when the observations are scarce!
{: .prompt-exercise}

People living in the Bayesian world faced issues with intractability of the posterior distributions. We introduced approximate inference to help the bayesians in that regard. Do frequentists also face such issues with the likelihood function that they rely on? Unfortunately, they do!

## EXAMPLE: GAUSSIAN MIXTURE MODEL

Consider a set of $K$ distinct gaussian distributions, where $k^{th}$ gaussian has mean as $\mu_k$ and covariance as $\Sigma_k$. Let there be an observation $y_i$ which is drawn from the $k^{th}$ gaussian with probability $\pi_k$. We have the observations with us, but we do not know which gaussian component generated any particular observation $y_i$. Then, we can write the likelihood as (assume that the observations are conditionally independent of each other given the parameters $\mathbf{\mu} = \mu_1, \mu_2, \ldots, \mu_K$ and $\mathbf{\Sigma} = \Sigma_1, \Sigma_2, \ldots, \Sigma_K$):

$$
\begin{align}
f(y|\mathbf{\mu},\mathbf{\Sigma}) &= \prod_{i = 1}^{N} f(y_i|\mathbf{\mu}, \mathbf{\Sigma}) \\
                                  &= \prod_{i = 1}^{N} \sum_{k} f(y_i, k | \mathbf{\mu}, \mathbf{\Sigma}) \qquad \text{(Applying law of total probability. k denotes the cluster assignment for y_i)}\\
                                  &= \prod_{i = 1}^{N} \sum_{k} f(y_i | \mathbf{\mu}, \mathbf{\Sigma}, k) \pi_k \qquad \text{(Splitting joint as product of conditional and marginal)} \\
                                  &= \prod_{i = 1}^{N} \sum_{k} f(y_i | \mu_k, \Sigma_k) \pi_k \\
f(y|\mathbf{\mu}, \mathbf{\Sigma}, \mathbf{\pi}) &= \prod_{i = 1}^{N} \underbrace{\sum_{k} \pi_k \mathcal{N}(y_i;\mu_k,\Sigma_k)}_{\text{Mixture of Gaussians}} \qquad \text{($\mathbf{\pi}$ is defined similar to $\mathbf{\mu}$ and $\mathbf{\Sigma}$)} \label{eqn67}\\
\end{align}
$$

Since we are in the frequentist world, we need to infer $\mathbf{\mu}$, $\mathbf{\Sigma}$ and $\mathbf{\pi}$ by maximizing the likelihood expressed above in equation $\ref{eqn67}$ with respect to all of them. Taking logarithm of the likelihood yields:

$$
\begin{align}
\log f(y|\mathbf{\mu}, \mathbf{\Sigma}, \mathbf{\pi}) = \sum_{i = 1} ^ {N} \log \sum_{k} \pi_k \mathcal{N}(y_i;\mu_k,\Sigma_k) \label{eqn10}\\
\end{align}
$$

Let us isolate the effect of change in $\mu_k$ on the logarithm of likelihood by keeping all other parameters fixed i.e. compute the gradient of function in $\ref{eqn10}$ with respect to $\mu_k$:

$$
\begin{align}
\frac{\mathrm{d}}{\mathrm{d\mu_k}} \log f(y|\mathbf{\mu}, \mathbf{\Sigma}, \mathbf{\pi}) = \sum_{i = 1}^{N} \underbrace{\frac{\pi_k \mathcal{N}(y_i;\mu_k,\Sigma_k)}{\sum_{k} \pi_k \mathcal{N}(y_i;\mu_k,\Sigma_k)}}_{\text{Responsibility $r(k,y_i)$}}\Sigma_k^{-1}(y_i - \mu_k) \\ 
\end{align}
$$

Responsibility denotes the posterior probability of cluster $k$ being assigned to $y_i$ given the latter. Set the above derivative to zero at $\mu_k = \hat{\mu_k}$ to obtain the optimal mean estimate that best explains the observations:

$$
\begin{align}
\left. \frac{\mathrm{d}}{\mathrm{d\mu_k}} \log f(y|\mathbf{\mu}, \mathbf{\Sigma}, \mathbf{\pi}) \right\vert_{\mu_k = \hat{\mu_k}} = 0 \\
\sum_{i = 1}^{N} r(k,y_i) [\Sigma_k^{-1}(y_i - \hat{\mu_k})] = 0 \\
\hat{\mu_k} = \frac{\sum_{i = 1}^{N} r(k,y_i)y_i}{\sum_{i = 1}^{N} r(k,y_i)} \label{eqn65}\\
\end{align}
$$

Note that the responsibility factor contains $\hat{\mu_k}$. The form in $\ref{eqn65}$ resembles $x = cos(x)$ in spirit, which doesn't have a closed-form solution for $x$ (we cannot express $x$ in terms of algebraic entities like +, - , *, / etc.). This is where intractability of computing ML estimates come into picture. There is no closed-form solution for $\hat{\mu_k}$ (so is the case with $\hat{\Sigma_k}$ and $\hat{\pi_k}$), which makes estimation harder. We neither know the clusters assigned to each observation $y_i$ (Had this been known, estimation of mean, covariance and the cluster probabilities is straightforward via computation of sample statistics) nor the other parameters (Knowing this enables one to identify the posterior distribution of cluster assigned to observations, which is also straightforward). As a way to work around this problem, Expectation Maximization was introduced to handle intractability of likelihood functions for ML estimation. 

> It is not recommended to attempt the above derivation for optimal covariance matrix $\hat{\Sigma_k}$ of the $k^{th}$ gaussian. That requires knowing how to differentiate the inverse of a matrix with respect to another matrix. I personally spent an entire afternoon working that out, and wouldn't recommend it. If you are still interested, refer to standard formula sheets for the matrix calculus and work your way through it! 
{: .prompt-info}

## EXPECTATION MAXIMIZATION

From $\ref{eqn65}$, we can deduce that $\hat{\mu_k} = f(\hat{\mu_k})$. While differentiating with respect to $\hat{\mu_k}$, we fixed all other unknown parameters. One can consider these fixed parameters to be their respective best estimates ($\hat{\mu_j}$ for all $j \neq k$, $\hat{\Sigma_k}$ and $\hat{\pi_k}$ for all $k$). This does no harm as the best estimates maximizes the likelihood. But we do not know what the best estimates are. Nevertheless, assume for now that it is sorted. A naive way to solve this is to consider an iterative method where the unknown parameters are randomly initialized, and plugged into $\ref{eqn65}$ (and other unwritten equations related to other parameter estimates) to obtain new estimates. Considering the new estimate as current, we plug those into the same equation and continue this process until the parameters do not vary further. If an input $x$ to a function $f$ remains invariant to the transformation i.e. $f(x) = x$, we call $x$ as the fixed point of the function $f$. Here, $\hat{\mu_k}$ behaves like the fixed point of $f$. Had $f$ been a linear function (or algebraically invertible - something easy to invert and obtain a function with algebraic entities like addition, subtraction etc.), solving for $\hat{\mu_k}$ is straightforward (as easy as solving for $x$ in $x = ax + b$). But, we know that $f$ is a complicated function (called as transcendental - any function that cannot be expressed using just addition, subtraction, other basic algebraic objects etc.) that involves ratio of an exponential to sum of exponentials. To solve for $\hat{\mu_k}$ in $\ref{eqn65}$, we need to employ fixed-point iterative methods that can bring us close to $\hat{\mu_k}$. Expectation Maximization (EM) is one such method to get the ML estimate of the unknown parameters in the presence of a hidden variable $z$. In our example, $z$ is the cluster assignment for each observation. This hidden variable introduced further complications in our log-likelihood function as we had to deal with logarithm of summation, which is considered a symptom of computational intractability. Without its presence, computing the ML etimate is not hard as is. We do not mostly require EM to get ML estimates in those cases.

More abstractly, $\ref{eqn67}$ can be written:

$$
\begin{align}
\mathbb{P}(y|\theta) &= \prod_{i = 1}^{N} \sum_{z} \mathbb{P}(z) \mathbb{P}(y_i | \theta, z) \label{eqn91}
\end{align}
$$

where $\theta$ encapsulates all the unknown parameters in it, and $z$ is the hidden variable. Since the proposed workaround is an iterative method, let us consider $\theta_m$ and $\theta_{m - 1}$ to be the estimated parameters in the $m^{th}$ and $(m - 1)^{th}$ iteration. Let us take the logarithm of LHS of $\ref{eqn91}$, which is a function of $\theta$ as $y$ is fixed, and call it $l(\theta)$. A bare minimum expectation would be to want the estimate in the next iteration to give atleast as good a likelihood as the previous iteration. In other words, we want $l(\theta_m) \geq l(\theta_{m - 1})$. With this, we will atleast reach a local maxima in the logarithm of the likelihood. That is guaranteed to happen theoretically as proven below:

$$
\begin{align}
l(\theta_m) &= \sum_{i = 1}^{N} \log \sum_{z} \mathbb{P}(z) \mathbb{P}(y_i | \theta_m, z) \\
            &= \sum_{i = 1}^{N} \log \sum_{z} \mathbb{P}(z|y_i, \theta_{m - 1}) \frac{\mathbb{P}(y_i | \theta_m, z)}{\mathbb{P}(z|y_i, \theta_{m - 1})} \mathbb{P}(z) \qquad \text{(Multiplying and dividing by $\mathbb{P}(z|y_i, \theta_{m - 1})$)}\\
            &\geq \sum_{i = 1}^{N} \sum_{z} \underbrace{\mathbb{P(z|y_i, \theta_{m - 1})}}_{\alpha} \log \frac{\mathbb{P}(y_i | \theta_m, z)}{\mathbb{P}(z|y_i, \theta_{m - 1})} \mathbb{P}(z) \qquad \text{($\log$ is a concave function and the inner summation is a convex combination, where $\alpha$ being a probability, is atmost $1$, and all possible $\alpha$'s sum to $1$)} \label{eqn92}\\

l(\theta_{m - 1}) &= \sum_{i = 1}^{N} \log \mathbb{P}(y_i | \theta_{m - 1}) \\
                  &= \sum_{i = 1}^{N} \sum_{z} \mathbb{P}(z | y_i, \theta_{m - 1}) \log \mathbb{P}(y_i | \theta_{m - 1}) \qquad \text{($\sum_{z} \mathbb{P}(z | y_i, \theta_{m - 1})$ is anyways just $1$)} \label{eqn93}\\
\end{align}
$$

Observe that subtracting the RHS of $\ref{eqn93}$ from the RHS of $\ref{eqn92}$ yields :

$$
\begin{align}
\sum_{i = 1}^{N} \sum_{z} \underbrace{\mathbb{P}(z|y_i, \theta_{m - 1})}_{\alpha} \log \frac{\mathbb{P}(y_i | \theta_m, z)}{\mathbb{P}(z|y_i, \theta_{m - 1})} \mathbb{P}(z) - l(\theta_{m - 1}) &= \sum_{i = 1}^{N} \sum_{z} \mathbb{P}(z | y_i, \theta_{m - 1}) \log \frac{\mathbb{P}(y_i, z | \theta_m)}{\mathbb{P}(y_i, z | \theta_{m - 1})} \\
\sum_{i = 1}^{N} \sum_{z} \underbrace{\mathbb{P}(z|y_i, \theta_{m - 1})}_{\alpha} \log \frac{\mathbb{P}(y_i | \theta_m, z)}{\mathbb{P}(z|y_i, \theta_{m - 1})} \mathbb{P}(z) &= l(\theta_{m - 1}) + \sum_{i = 1}^{N} \sum_{z} \mathbb{P}(z | y_i, \theta_{m - 1}) \log \frac{\mathbb{P}(y_i, z | \theta_m)}{\mathbb{P}(y_i, z | \theta_{m - 1})} \label{eqn94}
\end{align}
$$

We can substitute the RHS of $\ref{eqn94}$ to the RHS of the inequality in $\ref{eqn92}$ and obtain a summary stated below:

$$
\begin{align}
l(\theta_m) \geq l(\theta_{m - 1}) + \sum_{i = 1}^{N} \sum_{z} \mathbb{P}(z | y_i, \theta_{m - 1}) \log \frac{\mathbb{P}(y_i, z | \theta_m)}{\mathbb{P}(y_i, z | \theta_{m - 1})} \label{eqn95}
\end{align}
$$

To maximize $l(\theta_m)$ with respect to $\theta_m$, it is sufficient to maximize the second term of the RHS in $\ref{eqn95}$ with respect to $\theta_m$ as all other terms are fixed (we get $\theta_{m - 1}$ from the previous iteration). So, the best estimate of parameter in the $m^{th}$ iteration ($\hat{\theta_{m}}$) is given by the following expression:

$$
\begin{align}
\hat{\theta_m} &= \arg\max_{\theta_m} \sum_{i = 1}^{N} \sum_{z} \mathbb{P}(z | y_i, \theta_{m - 1}) \log \frac{\mathbb{P}(y_i, z | \theta_m)}{\mathbb{P}(y_i, z | \theta_{m - 1})} \\
               &= \arg\max_{\theta_m} \sum_{i = 1}^{N} \mathbb{E}_{z | y_i, \theta_{m - 1}}[\log \frac{\mathbb{P}(y_i, z | \theta_m)}{\mathbb{P}(y_i, z | \theta_{m - 1})}] \qquad \text{(The term within the outer sum is just expectation of the logarithm term with respect to $z$ conditioned on $y_i$ and $\theta_{m - 1}$)} \label{eqn77}\\
\end{align}
$$

We can isolate two specific events happening above into two separate equations:

$$
\begin{align}
Q(y_i, \theta_m) = \mathbb{E}_{z | y_i, \theta_{m - 1}}[\log \frac{\mathbb{P}(y_i, z | \theta_m)}{\mathbb{P}(y_i, z | \theta_{m - 1})}] \qquad \text{(Famously called E-step or Expectation-step which is self-explanatory)} \\
\hat{\theta_m} = \arg\max_{\theta_m} \sum_{i = 1}^{N} Q(y_i, \theta_m) \qquad \text{(Famously called the M-step or Maximization-step which is again self-explanatory)}
\end{align}
$$

> Note that we can ignore the denominator within the expectation as that is a constant (we know $\theta_{m - 1}$ from the previous iteration). 
{: .prompt-info}

That explains the name given for this algorithm as Expectation-Maximization. The pseudocode for EM is:

<div style="font-family:monospace;border:1.5px solid #000;border-radius:4px;overflow:hidden;max-width:700px;margin:1.5rem 0;background:#fff;color:#000"> <div style="background:#f5f5f5;border-bottom:1.5px solid #000;padding:8px 16px;font-family:sans-serif;font-size:13px;font-weight:600;text-align:center;color:#000"> Algorithm 1 — Expectation-Maximization (EM) Algorithm </div> <div style="padding:12px 16px 14px;font-size:13.5px;line-height:1.9;color:#000"> <span style="display:block"> <b>Input:</b> Observation vectors <i>{y<sub>n</sub>}<sub>n=1</sub><sup>N</sup></i>, hidden variable <i>z</i> </span> <span style="display:block"> <b>Output:</b> Maximum likelihood estimate <i>θ̂</i> </span> <span style="display:block"> <b>Initialize:</b> Initial parameter estimate <i>θ<sup>(0)</sup></i> </span> <hr style="border:none;border-top:0.5px solid #999;margin:6px 0"> <span style="display:block"> <b>while</b> <i>the log-likelihood has not converged</i> <b>do</b> </span> <span style="display:block;padding-left:2em"> <b>// E-step</b> </span> <span style="display:block;padding-left:2em"> Compute <i> Q(y<sub>n</sub>; θ) = 𝔼<sub>z|y<sub>n</sub>,θ<sup>(m−1)</sup></sub> [log p(y<sub>n</sub>,z | θ)] </i> </span> <span style="display:block;padding-left:2em"> using the conditional distribution <i>p(z | y<sub>n</sub>, θ<sup>(m−1)</sup>)</i> </span> <span style="display:block;padding-left:2em"> <b>// M-step</b> </span> <span style="display:block;padding-left:2em"> Update the parameter estimate </span> <span style="display:block;padding-left:2em"> <i> θ<sup>(m)</sup> = arg&nbsp;max<sub>θ</sub> Σ<sub>n=1</sub><sup>N</sup> Q(y<sub>n</sub>; θ) </i> </span> <hr style="border:none;border-top:0.5px solid #999;margin:6px 0"> <span style="display:block"><b>end</b></span> <span style="display:block;margin-top:4px"> <b>return</b> <i>θ̂ ← θ<sup>(m)</sup></i> </span> </div> </div>

The second term in the RHS of $\ref{eqn95}$ is guaranteed to be non-negative by the behavior of EM algorithm. Consider $\theta_m$ to momentarily be $\theta_{m-1}$. This makes that term vanish to $0$. Given that we are talking about a maximizer of the second term, the value of the second term at the maxima must be atleast $0$! 

## RETURNING TO THE EXAMPLE ON GAUSSIAN MIXTURE MODEL

Recall from $\ref{eqn65}$ the issue of $\hat{\mu_k}$ not having a closed-form solution. It ended up having parameters of other gaussians in its expression via the responsibility factor. Let us see how EM overcomes this issue!

We must first begin by computing the joint 
$$f(y_i, z | \mathbf{\mu}_{m}, \mathbf{\Sigma}_{m}, \mathbf{\pi}_{m})$$
, where $z$ is one of the $K$ gaussian components:

$$
\begin{align}
f(y_i, z = k | \mathbf{\mu}_{m }, \mathbf{\Sigma}_{m}, \mathbf{\pi}_{m}) &= \mathbb{P}(z = k | \mathbf{\pi}_{m}) f(y_i | \mu_{m}^{(k)}, \Sigma_{m}^{(k)}) \qquad \text{(Splitting joint as product of marginal and likelihood; superscript denotes the $k^{th}$ cluster)}\\
                                                                                    &= \pi_{m}^{(k)} \mathcal{N}(y_i ; \mu_{m}^{(k)}, \Sigma_{m}^{(k)}) \\
                                                                                    &= \prod_{j = 1}^{K} {(\pi_{m}^{(j)})}^{\mathbb{I}[j = k]} (\mathcal{N}(y_i ; \mu_{m}^{(j)}, \Sigma_{m}^{(j)}))^{\mathbb{I}[j = k]} \qquad \text{($\mathbb{I}$ is the indicator function)}\\ 
\end{align}
$$

Let $\mathbb{I}[j = k]$ be denoted as $z_j$. $z_j$ is a bernoulli random variable that can take either $0$ or $1$. We can put all $z_j$ into a $K$-dim vector and call that as $z$. It will be a one-hot random vector. The event {$z = k$} denotes the standard basis vector {$z = e_k$}. Rewriting the previous equation, we get:

$$
\begin{align}
f(y_i, z = e_k | \mathbf{\mu}_{m}, \mathbf{\Sigma}_{m}, \mathbf{\pi}_{m}) &=  \prod_{j = 1}^{K} {(\pi_{m}^{(j)})}^{z_j} (\mathcal{N}(y_i ; \mu_{m}^{(j)}, \Sigma_{m}^{(j)}))^{z_j} \\
\log f(y_i, z = e_k | \mathbf{\mu}_{m}, \mathbf{\Sigma}_{m}, \mathbf{\pi}_{m}) &= \sum_{j = 1}^{K} (z_j \log(\pi_{m}^{(j)}) + z_j \log (\mathcal{N}(y_i ; \mu_{m}^{(j)}, \Sigma_{m}^{(j)}))) \\
\mathbb{E}_{z | y_i, \mathbf{\mu}_{m - 1}, \mathbf{\Sigma}_{m - 1}, \mathbf{\pi}_{m - 1}}[\log f(y_i, z = e_k | \mathbf{\mu}_{m}, \mathbf{\Sigma}_{m}, \mathbf{\pi}_{m})] &= \sum_{j = 1}^{K} \log(\pi_{m}^{(j)} \mathcal{N}(y_i; \mu_{m}^{(j)}, \Sigma_{m}^{(j)})) \mathbb{E}_{z | y_i, \mathbf{\mu}_{m - 1}, \mathbf{\Sigma}_{m - 1}, \mathbf{\pi}_{m - 1}}[z_j] \label{eqn100}\\
\end{align}
$$

> Convince yourself that 
>$$\mathbb{E}_{z | y_i, \mathbf{\mu}_{m - 1}, \mathbf{\Sigma}_{m - 1}, \mathbf{\pi}_{m - 1}}[z_j] = \mathbb{P}(z = j | y_i, \mathbf{\mu}_{m - 1}, \mathbf{\Sigma}_{m - 1}, \mathbf{\pi}_{m - 1}) = r(j, y_i)$$
> and arrive at an expression for $r(j, y_i)$ in terms of the parameters from the previous iteration ($z = j$ is where $z$ is a scalar that denotes cluster assigned to an observation).
{: .prompt-exercise}

We can denote the expectation in the exercise, which is the responsibility factor computed at $m^{th}$ iteration, as $r_{m}(j, y_i)$. Substituting this result to $\ref{eqn100}$, we get:

$$
\begin{align}
Q(y_i; \mathbf{\mu}_{m}, \mathbf{\Sigma}_{m}, \mathbf{\pi}_{m}) = \mathbb{E}_{z | y_i, \mathbf{\mu}_{m - 1}, \mathbf{\Sigma}_{m - 1}, \mathbf{\pi}_{m - 1}}[\log f(y_i, z = e_k | \mathbf{\mu}_{m}, \mathbf{\Sigma}_{m}, \mathbf{\pi}_{m})] = \sum_{j = 1}^{K} r_{m}(j, y_i)\log(\pi_{m}^{(j)} \mathcal{N}(y_i; \mu_{m}^{(j)}, \Sigma_{m}^{(j)})) \qquad \text{(Expectation-step)} \\

\hat{\mu_{m}^{(j)}}, \hat{\Sigma_{m}^{(j)}}, \hat{\pi_{m}^{(j)}} = \arg\max_{\mu_m^{(j)}, \Sigma_m^{(j)}, \pi_m^{(j)}} \sum_{i = 1}^{N} Q(y_i; \mathbf{\mu}_{m}, \mathbf{\Sigma}_{m}, \mathbf{\pi}_{m}) = \arg\max_{\mu_m^{(j)}, \Sigma_m^{(j)}, \pi_m^{(j)}} \sum_{i = 1}^{N} \sum_{j = 1}^{K}r_{m}(j, y_i)\log(\pi_{m}^{(j)} \mathcal{N}(y_i; \mu_{m}^{(j)}, \Sigma_{m}^{(j)})) \qquad \text{(Maximization-step)} \\
\end{align}
$$

Notice the inner sum in the maximization step - Each of the terms include parameters of one component only. Differentating the maximization objective with respect to some parameter of a particular component will lend you a term that contains that component's parameters alone. This is unlike $\ref{eqn65}$, where parameters from other components were also involved. To be more precise, $r_{m}(j, y_i)$ is a term that involves estimated parameters from the previous iteration as seen in the exercise. This is a fixed term with respect to $\mu_m^{(j)}, \Sigma_m^{(j)}, \pi_m^{(j)}$. To estimate best parameters for the current iteration, there is no tight coupling netween current iteration parameters as seen in $\ref{eqn65}$. Let us derive the best estimates for the mean of the $k^{th}$ component $\mu_m^{(k)}$ and the mixing parameter of the the same component $\pi_m^{(k)}$. We will be directly stating the result for the best estimate of the covariance of the $k^{th}$ component. Deriving that is a tedious process and requires one to know matrix calculus. I wouldn't recommend deriving that unless you have the time!

$$
\begin{align}
\frac{\mathrm{d}}{\mathrm{d\mu_m^{(j)}}} \sum_{i = 1}^{N} \sum_{j = 1}^{K}r_{m}(j, y_i)\log(\pi_{m}^{(j)} \mathcal{N}(y_i; \mu_{m}^{(j)}, \Sigma_{m}^{(j)})) &= 0 \\
\sum_{i = 1}^{N} r_{m}(j, y_i) {\Sigma_{m}^{(j)}}^{-1}(y_i - \hat{\mu_{m}^{(j)}}) &= 0 \\
\hat{\mu_{m}^{(j)}} &= \frac{\sum_{i = 1}^{N} r_{m}(j, y_i) y_i}{\sum_{i = 1}^{N} r_{m}(j, y_i)}
\end{align}
$$

For the mixing parameter of the $k^{th}$ component, we need to bear in mind that there are constraints. For instance, $\sum_{j = 1}^{K} \pi_{m}^{(j)} = 1$. We need to solve a constrained optimization problem using lagragian multipliers:

$$
\begin{align}
\frac{\mathrm{d}}{\mathrm{d\pi_m^{(j)}}} (\sum_{i = 1}^{N} \sum_{j = 1}^{K}r_{m}(j, y_i)\log(\pi_{m}^{(j)} \mathcal{N}(y_i; \mu_{m}^{(j)}, \Sigma_{m}^{(j)})) + \lambda (\sum_{j = 1}^{K} \pi_{m}^{(j)} - 1)) &= 0 \\
\sum_{i = 1}^{N} \frac{r_{m}(j, y_i)}{\pi_{m}^{(j)}} + \lambda &= 0 \\
\sum_{i = 1}^{N} r_{m}(j, y_i) + \lambda \pi_{m}^{(j)} &= 0 \label{eqn101}\\
\sum_{j = 1}^{K} \sum_{i = 1}^{N} r_{m}(j, y_i) + \lambda \sum_{j = 1}^{K} \pi_{m}{(j)} &= 0 \qquad \text{(Summing up both sides over the cluster indices)} \\
\sum_{i = 1}^{N} \sum_{j = 1}^{K} r_{m}(j, y_i) + \lambda &= 0 \qquad \text{(The term after lambda sums to $1$)}
\sum_{i = 1}^{N} 1 + \lambda &= 0 \qquad \text{(The inner sum within the first term is the sum over cluster indices of posterior cluster assignment probabilities given a particular observation, which sums to $1$)} \\
\lambda &= -N \\
\end{align}
$$

Substituting for $\lambda$ in $\ref{eqn101}$, we get:

$$
\begin{align}
\hat{\pi_{m}^{(j)}} = \frac{\sum_{i = 1}^{N} r_{m}(j, y_i)}{N} = \frac{N_{m}^{(j)}}{N} \qquad \text{($N_{m}^{(j)}$ is the effective number of observations within cluster $j$ in the $m^{th}$ iteration)}
\end{align}
$$

The best covariance estimate in the $m^{th}$ iteration is given by :

$$
\begin{align}
\hat{\Sigma_{m}^{(j)}} = \frac{\sum_{i = 1}^{N} r_{m}(j, y_i) (y_i - \mu_{m}^{(j)})(y_i - \mu_{m}^{(j)})^{T}}{\sum_{i = 1}^{N} r_{m}(j, y_i)}
\end{align}
$$

The best mean estimate in the $m^{th}$ iteration can be interpreted as the weighted average of observations (with the weights reflecting how strongly a particular observations falls into a particular cluster). The weights are obtained from the best estimates of the parameters from the previous iteration. The best covariance estimate can be similarly interpreted from an empirical perspective. The best mixing parameter estimate is essentially the fraction of all observations lying in a particular cluster. 

```python
"""
EXPECTATION-MAXIMIZATION AS APPLIED TO GAUSSIAN MIXTURE MODELS

TO RUN THIS CODE LOCALLY, YOU NEED TO CLONE THE GITHUB REPOSITORY LINKED IN THE "GITHUB REPO CREDITS" SECTION OF THIS BLOG.
CREATE A VIRTUAL ENV WITH PYTHON 3.10 AND PIP INSTALL THE REQUIREMENTS AS IN THE GITHUB REPOSITORY'S "requirements.txt"
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.special import logsumexp, softmax 
from gaussian_contour_levels.contourlevels import make_ellipse_parameter_dict  
from matplotlib.patches import Ellipse

np.random.seed(20)

def log_likelihood(random_mixture_probabilities, random_cluster_means, random_cluster_covariances, N , K):
    log_probs = np.stack(
        [multivariate_normal.logpdf(y, mean=random_cluster_means[k], cov=random_cluster_covariances[k])
         for k in range(K)],
        axis=-1
    )                                           # Thanks to Claude for identifying an optimization in this function
    return np.sum(logsumexp(np.log(random_mixture_probabilities)[None, :] + log_probs, axis=-1))

K = 3           # Number of gaussian components
N = 1000        # Number of observations
D = 2           # Dimension of the observations
alphas = np.random.uniform(0,1,K)           # Sampling the hyperparameters of the dirichlet distribution
mixture_probabilities = np.random.dirichlet(alphas)                         # Getting mixing probabilities from dirichlet which guarantees that they sum to 1
cluster_ids = np.arange(K)          
cluster_assignments = np.random.choice(cluster_ids, N, p = mixture_probabilities)         # Assigning cluster ids to each observation
cluster_means = np.random.normal(10,20,size=(K,D))                # Generating cluster means
random_data = np.random.normal(10, 7, size = (K, D + 10, D))            #   Generating random data to get a proper covariance matrix using np.cov()
cluster_covariances = []
for i in range(K):
    covariance = np.cov(random_data[i], rowvar=False)
    cluster_covariances.append(covariance)
cluster_covariances = np.array(cluster_covariances) # Getting the covariance matrix for each cluster
y = np.array([np.random.multivariate_normal(mean = cluster_means[c], cov = cluster_covariances[c]) for c in cluster_assignments])    # Generating observations from mixture of gaussians

# Expectation-Maximization to estimate the parameters of the mixture of gaussians given the observations generated earlier
random_cluster_means = np.random.uniform(10, 12, size=(K,D))        # Cluster means random initialization
random_data_for_random_init = np.random.normal(2,6, size = (K, D + 10, D))
random_cluster_covariances = []
for i in range(K):
    random_covariance = np.cov(random_data_for_random_init[i], rowvar=False) + 1e-6 * np.eye(D)
    random_cluster_covariances.append(random_covariance)
random_cluster_covariances = np.array(random_cluster_covariances)       # Cluster covariances random initialization
random_alphas = np.random.uniform(5,9,K)
random_mixture_probabilities = np.random.dirichlet(random_alphas)       # Mixture probabilities random initialization

max_iter = 1000
log_likelihood_init = 0
curr_log_likelihood = log_likelihood(random_mixture_probabilities, random_cluster_means, random_cluster_covariances, N, K)  # Computing initial log-likelihood with random initialization of parameters
start = curr_log_likelihood
count = 0
eps = 1e-4

state_tracker = []
while ((count < max_iter) and (np.abs(log_likelihood_init - curr_log_likelihood) >= eps)):
    log_probs = np.stack(
        [multivariate_normal.logpdf(y, mean=random_cluster_means[k], cov=random_cluster_covariances[k])
         for k in range(K)],
        axis=-1
    )                   # Thanks to Claude for suggesting a similar optimization as in the log_likelihood function for speedup in responsibility computation
    responsibilities = softmax(np.log(random_mixture_probabilities)[None, :] + log_probs, axis=-1).T
    common_denom = np.sum(responsibilities, axis = -1)
    random_cluster_means = (responsibilities @ y) / common_denom[:, None] # Updating cluster means
    A = np.expand_dims((np.expand_dims(y, axis = 1) - random_cluster_means), axis = -1)
    random_cluster_covariances = np.sum((A @ A.transpose((0,1,3,2))) * np.expand_dims(responsibilities.transpose((1,0)), axis = (2,3)), axis = 0) /  common_denom[:, None][:, None]     # Updating cluster covariances
    random_cluster_covariances += 1e-6 * np.eye(D)[None, :, :]
    random_mixture_probabilities = common_denom / N         # Updating mixing probabilities
    log_likelihood_init = curr_log_likelihood
    curr_log_likelihood = log_likelihood(random_mixture_probabilities, random_cluster_means, random_cluster_covariances, N, K)
    state_tracker.append((random_cluster_means, random_cluster_covariances, random_mixture_probabilities, curr_log_likelihood))
    count += 1

sample = [0, int(count / 4), int(count / 2), int((3*count) / 4), int(count - 1)]        # State tracking only for these indices in state_tracker
#Plotting log-likelihood across iterations
likelihoods = np.array([start] + [i[-1] for i in state_tracker])
plt.figure()
plt.title('Evolution of log-likelihood')
plt.plot(likelihoods)
plt.xlabel('Iterations')
plt.ylabel('log-likelihood')

# Plotting the true gaussian contour
fig, ax = plt.subplots()
ax.scatter(y[:,0], y[:,1], c = cluster_assignments, cmap='tab10')
confidence = [0.5,0.9,0.99]
cmap = plt.get_cmap('tab10')
for i in range(K):
    ell_props = make_ellipse_parameter_dict(cluster_covariances[i], confidence_list = confidence)
    ax.scatter(cluster_means[i][0], cluster_means[i][1], marker = 'x', color = cmap(i))
    for conf in confidence:
        ax.add_patch(
            Ellipse(
                (cluster_means[i][0], cluster_means[i][1]),
                width = 2*ell_props[f'major_axis_{str(conf)[2:]}'],
                height = 2*ell_props[f'minor_axis_{str(conf)[2:]}'],
                angle = ell_props[f'angle_{str(conf)[2:]}'] * 180 / np.pi,
                facecolor = "none",
                edgecolor = cmap(i),
                linestyle ='--',
                linewidth = 1.5,
            )
        )
ax.set_title('True Gaussian Contours')
ax.set_aspect("equal")

# Plotting the estimated gaussian contour
for idx in sample:
    fig, ax = plt.subplots()
    random_cluster_means = state_tracker[idx][0]
    random_cluster_covariances = state_tracker[idx][1]
    ax.scatter(y[:,0], y[:,1], c = cluster_assignments, cmap='tab10')
    confidence = [0.5,0.9,0.99]
    for i in range(K):
        ell_props = make_ellipse_parameter_dict(random_cluster_covariances[i], confidence_list = confidence)
        ax.scatter(random_cluster_means[i][0], random_cluster_means[i][1], marker = 'x', color = cmap(i + 4))
        for conf in confidence:
            ax.add_patch(
                Ellipse(
                    (random_cluster_means[i][0], random_cluster_means[i][1]),
                    width = 2*ell_props[f'major_axis_{str(conf)[2:]}'],
                    height = 2*ell_props[f'minor_axis_{str(conf)[2:]}'],
                    angle = ell_props[f'angle_{str(conf)[2:]}'] * 180 / np.pi,
                    facecolor = "none",
                    edgecolor = cmap(i + 4),
                    linestyle ='--',
                    linewidth = 1.5,
                )
            )
    ax.set_title(f'Estimated Gaussian Contours for state {idx}')
    ax.set_aspect("equal")

plt.tight_layout()
plt.show()
```

![Log-Likelihood Convergence](/assets/images/expectation_maximization/log_likelihood.png)

As observed in the plot above, the log-likelihood is monotonically increasing as conveyed by the theoretical derivation earlier. This is a sanity check for the correctness of our implementation. If the log-likelihood even remotely decreases in this plot, then there is some implementation error somewhere in the code.

![True Gaussian Contour](/assets/images/expectation_maximization/true_contour.png)

For the illustration, well separated gaussian components were used to generate observations. I could have chosen overlapping gaussians (or poorly separated observations), but that was not visually pleasing. If you wish to observe the EM behavior in those cases, you can modify the distribution of the cluster_means variable in the code and analyze the plots. You will notice that it takes more iterations to converge than the current setup (Nevertheless, the code execution is fast for the latter case!). 

Let us see how the EM algorithm evolved over iterations, and tried adapting to the observed data! (Note that the word 'state' in the figure refers to EM iteration index)

![Evolution of EM - 0](/assets/images/expectation_maximization/state_0.png)
![Evolution of EM - 4](/assets/images/expectation_maximization/state_4.png)
![Evolution of EM - 9](/assets/images/expectation_maximization/state_9.png)
![Evolution of EM - 14](/assets/images/expectation_maximization/state_14.png)
![Evolution of EM - 18](/assets/images/expectation_maximization/state_18.png)

## EPILOGUE

In this article, we covered the difficulties faced by empiricists in ML estimation. Analogous to approximate inference techniques that bayesians use, Expectation-Maximization was introduced as a tool that the frequentists can employ to work around computational intractability. But, the discussion on EM doesn't end here...

There is an interesting connect between EM and Variational Inference!

Recall that 
$ELBO(q) = \mathbb{E}_{z \sim q}[\log \frac{p(y,z)}{q(z)}]$. In the previous article on variational inference, we saw that best approximation of the intractable posterior $p(z|y)$ is the function $q$ that maximizes $ELBO(q)$ or equivalently minimizes the KL-Divergence between $p(z|y)$ and $q(z)$. In the setting of EM, $p(y,z)$ is parameterized by $\theta$ and is written as $p(y,z|\theta)$. We neither know $q$ nor $\theta$. We also do not have any informative prior on $\theta$. So, the $ELBO$ becomes a function of $q$ and $\theta$. Furthermore, $ELBO(q) \leq \log p(y|\theta)$. In order to maximize the RHS of the inequality (i.e. Likelihood), it is sufficient to maximize the LHS i.e. $ELBO$. As explained in the <a href="https://mbernste.github.io/posts/em/">blog post</a> by Matthew Bernstein on EM, it can be seen that EM is just a coordinate ascent on $ELBO$, where we alternate between finding maximizers $q$ and $\theta$! For more details, you can read the blog post. I do not wish to plunge into more details right now. In future blog posts, we will explore this connection more. For now, we can halt the discussions on theoretical aspects and see where the theory developed so far in the blog posts get applied. The next article will be focussing on a speech application (a research paper) that makes use of concepts seen for acoustic unit discovery in speech signals.

## REFERENCES

> Sayed, A. H. (2022). *Inference and Learning from Data: Inference*. Cambridge University Press.

## GITHUB REPO CREDITS   

Credits are due to <a href = "https://github.com/herzphi">herzphi</a>, whose <a href = "https://github.com/herzphi/2DGaussianContourLevels">github repository </a> aided in plotting gaussian contours.

## AI USAGE DISCLOSURE

None of the textual content (with the exception of the pseudocode) was written using AI. The codes were written manually for most of the part (except for a few places where proper credit has been given to the bot used). ChatGPT was used as a reference aid for syntax lookup. Claude aided in creating the box for writing algorithms, and also wrote the pseudocodes by referring to the aforementioned book. 