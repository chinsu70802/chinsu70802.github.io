---

layout: post
title: "Mean Square Inference"
date: 2026-05-19
categories: [Statistics]
tags: [Statistics]
math: True
---

## PROLOGUE

We must have all had experience playing dumb charades. The actor has a word in his mind, but he is not allowed to utter that word to the other contestants. He can just use his body gestures to act out the word without giving any verbal cues. The others are supposed to "estimate" or "infer" the word that is being acted out. 

Jumping into a more technical example, we can look at the analogue of the above. We have a person who thinks up a word in his mind. The others must guess what that word is. So far, the setup is the same as dumb charades. Except that what the others see is not his gestures, but the acoustic waveform of the actual word (Imagine that in this world, people can only see the other person's waveform but not hear what the others speak). 

The word in the person's mind undergoes a series of transformations - starting from conversion of neuronal signals to mechanical actions that help coordinate the body gestures (articulators) to act out the word (generate an acoustic waveform). In both the cases, what is hidden is the word and what is observed is a manifestation of the hidden object - either as bodily gestures or acoustic waveform.

Essentially, we can think of the literal word that exists in the person's brain as a hidden variable (something that is not directly observable to the outside world) $x$. The others are making a series of observations, his gestures (acoustic waveform), which we can label as $y$. The task is to estimate $x$ given $y$ as accurately as possible, where $x$ and $y$ are related to each other in some way.

To make it more mathematically precise, we have a distribution over the set of words that can exist in a given language. In dumb charades, you have the actor point out a language using his fingers. In the waveform example, assume that the language of the word is known. We already have a prior knowledge $p(x)$ on $x$. We know that some certain words are more probable than the others as a prior belief. Let us take one candidate $x$ from $p(x)$ and understand how likely are the body gestures (waveform) given that person has $x$ in his mind, the likelihood $p(y\|x)$. We can use Bayes' rule to characterize the posterior distribution $p(x\|y) = p(y\|x) \cdot p(x)$, which says how probable is $x$ given $y$. We are more interested in a point estimate at the moment - $\arg\max\limits_{x}p(x \| y) = \arg\max\limits_{x} p(y \| x) \cdot p(x)$. We want the most probable word $x$ to be chosen from the posterior.

This is what is meant by inference - estimate the unknown using observable manifestations of it!

> We may have stopped with just the posterior distribution itself - That is also a form of inference as we have estimated a distribution of the unknown given the observation. Inference doesn't always mean a point estimate obtained from a distribution. It is a process through which one can quantitatively ascertain the uncertainty of any unknown random variable (e.g. getting to know that an unknown random variable has so and so mean and variance is itself an appreciable inference). For most of the discussions here, we will be interested in point estimates (More importantly, we more or less know the distribution of the hidden variable, but we do not know what the hidden variable gets realized as at any given moment. Our efforts will be directed towards estimating that unknown realization from a known distribution of the hidden variable).
{: .prompt-tip }

> The example involving the acoustic waveform is an interesting way of viewing the problem of automatic speech recognition. It is called as the 'noisy-channel model' of speech recognition. You can refer to Jurafsky & Martin's book on Speech and Natural Language Processing to learn more.
{: .prompt-info }

## MEAN SQUARE RISK

What is desired is an optimal estimate of the unknown random variable. But optimal in what sense? If there is an estimation, we need to know how good of an estimate that is. To facilitate this, we can begin by finding the squared difference between the groundtruth $x$ and its estimation $\hat{x}$. As $x$ and $\hat{x}$ can be random variables, we are interested in $\mathbb{E}[(x - \hat{x})^2]$ (If the subscript of $\mathbb{E}$ is empty, then we are taking expectation over all quantities of randomness in the expression). This is the mean square risk. Our objective is to minimize the mean square risk with respect to $\hat{x}$, the only quantity that is under our control. $x$ is decided by nature, and beyond our control. In effect, $$\hat{x}_{o} = \arg\min\limits_{\hat{x}} \mathbb{E}[(x - \hat{x})^2]$$, where $\hat{x}_{o}$ is the optimal/best estimate. On an average, we want the estimate $\hat{x}$ to be as close as possible to $x$. We will be using the mean square risk for the rest of this blog as a means to measure optimality of an estimator.

> The quantity $x - \hat{x}$ is called the residual ($\tilde{x}$). We will be frequently commenting about this in the future. Ideally, we want the best estimate of $x$ which is as close as possible to $x$. No matter what $x$ nature decides to use and the estimate $\hat{x}$ we derive from the observable manifestations of $x$, the residual should be close to zero for the best estimate. On an average, the residual should be close to zero ($\mathbb{E}[\tilde{x}] \approx 0$). For each $x$, we would have obtained different estimates. The set of residuals must be compactly packed around zero. For example, if x can take values like {$36.025, 64.0001, \cdots$} and the estimates are {$36, 64, \ldots$}, the residuals will be {$0.025, 0.0001, \cdots$}. As can be noticed, the variance of the residuals is close to zero. Ideally, it is expected that $\tilde{x}$ takes the value $0$ almost surely. In practice, that is not possible (unless we are talking about a deterministic process that nature decides to use). Residuals can be thought of as a part of $x$ that remains after removing the predictable aspect $\hat{x}$ from it. We ideally want to predict everything about $x$. We will see more on this later.
{: .prompt-tip } 

## INFERENCE WITHOUT OBSERVATIONS

What if we do not have access to any observation $y$ related to $x$, apart from the mean $\bar{x} = \mathbb{E}[x]$ of $x$? $\hat{x}$ is not a random variable in this case. If we have observations, then everytime natures generates $x$ (which is not observable by the world), there is also a corresponding observation. We can estimate $x$ based on the $y$ that we observe. We get different estimates for different observations, making $\hat{x}$ a random variable. Without observations, there is no scope for us to change our estimates. Since $x$ is hidden, once we estimate it, we remain fixed in our guess. So, $\hat{x}$ doesn't have any randomness here. Imagine a scenario where a person cannot sense the temperature of the surrounding, and has no access to thermometer readings. For him, the temperature is $x$, a hidden variable. All he knows at present is the average temperature. He makes a guess and remains stuck with it as he is not given information related to the temperature at any point in time. What should his best guess be?

$$
\begin{align}
\hat{x}_{o} = \arg\min\limits_{\hat{x}} \mathbb{E}[(x - \hat{x})^2] \label{eq1}
\end{align}
$$

As per $\eqref{eq1}$, we need to minimize the expectation with respect to $\hat{x}$.

$$
\begin{align}
\mathbb{E}[(x - \hat{x})^2] &= \mathbb{E}[(x - \bar{x} + \bar{x} - \hat{x})^2] \\
                            &= \mathbb{E}[(x - \bar{x})^2] + \mathbb{E}[(\bar{x} - \hat{x})^2] + 2\mathbb{E}[(x - \bar{x})(\bar{x} - \hat{x})] \\
                            &= \mathbb{Var}[x] + \mathbb{E}[(\bar{x} - \hat{x})^2] + 2(\bar{x} - \hat{x})\mathbb{E}[x - \bar{x}] \\
                            &= \mathbb{Var}[x] + (\bar{x} - \hat{x})^2 \qquad \text{($\mathbb{E}[x - \bar{x}] = 0$ by linearity of expectations)} \\
                            &\geq \mathbb{Var}[x] \qquad \text{(As $(\bar{x} - \hat{x})^2$ is non-negative, the minimum we can achieve with it is $0$ when $\hat{x} = \bar{x}$)} \label{eq2}
\end{align}
$$

From the above derivation, it can be observed that the mean square risk is minimized for the case with no observations when the best estimate $\hat{x}_{o} = \bar{x}$. So, the best guess we can make is whatever is given about the hidden $x$, which is the mean $\bar{x}$ itself. 

On the other hand, notice that $$\mathbb{E}[\hat{x}_{o}] = \mathbb{E}[x] = \bar{x}$$. The best estimate happens to be an unbiased estimate (the previous sentence is the definition of an unbiased estimate). What does it tell us about the residue $\tilde{x}$? $\mathbb{E}[\tilde{x}] = \mathbb{E}[x - \hat{x}_{o}] = 0$. As expected, the mean of the residue is $0$. We can write the minimum risk as :

$$
\begin{align}
\mathbb{E}[(x - \hat{x}_{o})^2] &= \mathbb{E}[\tilde{x}^2] \\
                                &= \mathbb{E}[\tilde{x}^2] - 0^2 \\
                                &= \mathbb{E}[\tilde{x}^2] - (\mathbb{E}[\tilde{x}])^2 \qquad \text{(As seen in the preceding paragraph)} \\
                                &= \mathbb{Var}[\tilde{x}]
\end{align}
$$

The minimum risk with the best guess $$\hat{x}_{o}$$ is the variance of the residue. As seen in the tip provided earlier, the variance of the residue must be as small as possible. Is that the case here? We get the minimum risk as variance of the input as well from $\eqref{eq2}$. So, under the optimal estimate $\hat{x}_{o} = \bar{x}$, the variance of the residue is the same as variance of the input. This is alarming! If the input has very high variance, so will the residue - it increases the likelihood of large residues. The residue in this case, $x - \bar{x}$, is just the mean centered version of $x$. The distribution of the residue remains as broad as the input. This estimator is "poor" in the sense that the residue and the input share the same variance. You can refer to the figure below to understand better in the context of gaussians. We should graduate towards "good" estimators - ones where $\mathbb{Var}[\tilde{x}] \lt \mathbb{Var}[x]$. Can having observations $y$ help in this?

> Note that we made no assumptions about the distribution of the hidden variable $x$ in the derivation above. Irrespective of the nature of distribution of $x$ (whether iid or not), $\bar{x}$ is the best guess that can be made without any observations!
{: .prompt-info }

```python
"""
VISUALIZING INFERENCE WITH NO OBSERVATIONS USING GAUSSIANS
"""
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
We are given the mean of the gaussian in this case (we are also told that it is a gaussian). But no other information is given to you. Below, we are computing the residue using the best possible estimate, the mean itself.
"""
x_tilde = x - 3 
y_tilde = gaussian(x_tilde, 0, 2)

plt.plot(x_tilde, y_tilde, label=r'Distribution of the residue ($\hat{X} = X - \bar{X}$)')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.grid(True)

plt.show()
```

![Inferencing a gaussian random variable with no observations related to it](assets/images/inference_without_observations.png)

## INFERENCE WITH OBSERVATIONS

Now, consider observation $y$ that is related in some way to $x$. Recall the first two examples from the beginning of this article to understand $y$. Here, the estimate will be a random variable. It depends on what $y$ is observed. Elements of randomness in this case include $x$, $y$ and consequentially $\hat{x} = c(y)$, where $c$ is some function of $y$. The risk minimization becomes $\arg\min\limits_{\hat{x}} \mathbb{E}[(x - \hat{x})^2] = \arg\min\limits_{c(.)} \mathbb{E}[(x - c(y))^2]$ with respect to any function $c$ of $y$. The expectation is over both $x$ and $y$. 

> The derivation of the optimal estimate for this case is quite similar to the case without observations. As an exercise, you can derive this by referring to the previous section.
> <details>
> <summary>Hint</summary>
> Add and subtract $\mathbb{E}[x|y]$ just like $\bar{x}$ in the previous section. You may not know what to do with $2\mathbb{E}[(x - \mathbb{E}[x|y])(\mathbb{E}[x|y] - \hat{x})]$. Derive an alternate expression for $\mathbb{E}[xg(y)]$, where the expectation is over both $x$ and $y$ and $g$ is a function. Apply your result to $2\mathbb{E}[(x - \mathbb{E}[x|y])(\mathbb{E}[x|y] - \hat{x})]$ by observing its similarity with $\mathbb{E}[xg(y)]$.
> </details>
{: .prompt-exercise }

You must have got $\hat{x}_{o} = \mathbb{E}[x\|y]$ - the optimal estimate is the conditional mean estimate! This is also an unbiased estimator. Let us come to the part that is of utmost concern: the residue.

> Similar to the previous section, you can derive the variance of the residue. Refer to the previous section. 
> <details>
> <summary>Hint</summary>
> Start with the minimum risk formulation and substitute $\mathbb{E}[x|y]$ for the optimal estimate. Do not forget the identity related to $\mathbb{E}[xg(y)]$.
> </details>
{: .prompt-exercise }

You will notice that $\mathbb{Var}[\tilde{x}] = \mathbb{Var}[x] -\mathbb{Var}[\hat{x}_{o}]$. 

> (a) Rearrange the previous expression such that you have an expression for $\mathbb{Var}[x]$ in terms of the other two quantities in the expression. Substitute the values of $\tilde{x}$ and $\hat{x}_{o}$, and obtain a compact expression.\\
> (b) Is it still possible for the variance of the residue to be equal to variance of the input in the case with observations?
> <details>
> <summary>Hint</summary>
> (a) You will end up with the law of total variance - $\mathbb{Var}[x] = \mathbb{E}[\mathbb{Var}[x|y]] + \mathbb{Var}[\mathbb{E}[x|y]]$.
> <br><br>
> (b) Yes. It happens when $x$ and $y$ are uncorrelated. But didn't we say that $x$ and $y$ are related to each other in some way? The issue is when both of them share a non-linear relationship that has zero correlation. For example, consider the case where $y = x^2 + n$, where $x \sim Cat(1/5,1/5,1/5,1/5,1/5)$ with values belonging to {$-2, -1, 0, 1, 2$} and $n \sim \mathcal{N}(0,1)$. Trivially, if $x$ and $y$ are independent, then the variance of residue will be same as variance of input. But this is not very interesting as we know that both the variables are related to each other in some way.
> </details>
{: .prompt-exercise }

> Let $x \sim Ber(1/2)$ where it is either realized as $+1$ or $-1$. Assume there exists an additive noise $n \sim \mathcal{N}(0,\mathbb{Var}[n])$. We observe $y = x + n$. Get the optimal estimate for $x$ using MSE risk.
{: .prompt-exercise}

As a follow-up to the previous exercise, we can visually see the variance of the residue being less than the variance of $x$ in the figure below.

```python
"""
VISUALIZING INFERENCE WITH OBSERVATIONS BASED ON THE PREVIOUS EXERCISE
"""
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
```

![Inferencing, with observations, a bernoulli hidden variable under additive gaussian noise](assets/images/inference_with_observations.png)

> Modify the code pasted above to understand the effect of noise (by tweaking std_dev_n - making it too large or too small) on the variance of residue.
{: .prompt-exercise}

There is one important step that you may have come across while finding the optimal estimate for the case with observations. In the derivation,
$$\mathbb{E}[(x - \mathbb{E}[x|y])(\mathbb{E}[x|y] - \hat{x})]$$ would have been $0$. It is because the second operand in the inner expression is a function of $y$. Application of distributive property to the expression (by considering the second operand as $g(y)$ and taking it into the first operand) gives you $0$. We can also inspect $\mathbb{E}[(x - \mathbb{E}[x|y])]$ and $\mathbb{E}[\mathbb{E}[x|y] - \hat{x}]$ separately. The former expectation is $0$ as $\mathbb{E}[\mathbb{E}[x|y]] = \mathbb{E[x]}$. The product of these two expectations will also be zero. In effect, $\mathbb{E}[(x - \mathbb{E}[x|y])(\mathbb{E}[x|y] - \hat{x})] = \mathbb{E}[(x - \mathbb{E}[x|y])] \cdot \mathbb{E}[\mathbb{E}[x|y] - \hat{x}] = 0$. This implies that both the expressions are uncorrelated. More specifically, the residue with the optimal estimate $x - \mathbb{E}[x|y]$ is orthogonal (uncorrelated, denoted by $\perp$) to any function $g(y)$. No matter how much I modify $y$, I cannot gain any information about the residue. This is practical - there will always be some amount of uncertainity that cannot be inferred/predicted from any observation or its function. This is called as the **Orthogonality** property.

### ORTHOGONALITY DEFINES OPTIMALITY

There is an interesting result that pops up around orthogonality that it deserves a separate section. 

> Theorem: 
> <br><br>
> $\hat{x}$ is an optimal estimator in the mean square sense iff $\hat{x}$ is unbiased and $(x - \hat{x}) \perp g(y)$ for any function $g$ of $y$.

**Proof:**

For the forward part, we know that the optimal estimator in the mean square sense (risk) is 
$\hat{x}_{o} = \mathbb{E}[x|y]$. We need to prove that this estimate is unbiased and the residue is orthogonal to any function of $y$. We have already proved both these results previously. So, the forward part holds.

For the backward part, assume we have an estimate $\hat{x}$ which is both unbiased and whose residue is orthogonal to any function of $y$. Consider a variable 
$z = \hat{x} - \mathbb{E}[x|y]$. $\mathbb{E}[z] = \mathbb{E}[\hat{x}] - \mathbb{E}[\mathbb{E}[x|y]]$. As $\hat{x}$ is unbiased as was assumed, and the fact that $\mathbb{E}[\mathbb{E}[x|y]] = \mathbb{E[x]}$, $\mathbb{E}[z] = 0$. From our earlier discussions, we know that $(x - \mathbb{E}[x|y]) \perp g(y)$ for any function $g$ of $y$. It is also given that $(x - \hat{x}) \perp g(y)$ for any function $g$ of $y$. It can be seen that $(\hat{x} - \mathbb{E}[x|y]) \perp g(y)$.

> Show that if $(x - \mathbb{E}[x|y]) \perp g(y)$ and $(x - \hat{x}) \perp g(y)$ for any function $g$ of $y$, then $(\hat{x} - \mathbb{E}[x|y]) \perp g(y)$
> <details>
> <summary>Hint</summary>
> If $X$ and $Y$ are two random variables that are uncorrelated, then $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$. Use this formulation for the two uncorrelatedness conditions in the exercise and proceed to prove.
{: .prompt-exercise }

We know that 
$z = \hat{x} - \mathbb{E}[x|y]$. Rewriting the previous result, 
$\mathbb{E}[zg(y)] = 0$ for any function $g$ of $y$. Note that $z$ is itself a function of $y$. We can set $g(y) = z$ as the previous holds for any function of $y$. This implies that $\mathbb{E}[z^2] = 0$. As $\mathbb{E}[z] = 0$, $\mathbb{Var}[z] = 0$. It means that $z = \hat{x} - \mathbb{E}[x|y]$ takes the value $0$ almost surely. In other words, 
$\hat{x} = \mathbb{E}[x|y]$ almost surely. If the assumptions in the backward part are satisfied by some estimate $\hat{x}$, then that estimate has no other choice than being the conditional mean estimate $\mathbb{E}[x|y]$.

We have proved that orthogonality is a defining property for an optimal estimator in the mean square sense!

For scalar random variables $x$ and $y$, the discussion so far holds. We can generalize the above to vector random variables, particularly for gaussian random vectors $\vec{x}$ and $\vec{y}$

### CONDITIONAL MEAN ESTIMATE FOR GAUSSIAN RANDOM VECTORS

For mean square risk, we can derive the optimal estimator, the conditional mean estimate, when both the hidden and the observation variables are gaussian random vectors. This section aims to introduce an interesting algebraic manipulation (as seen in the book by Ali Sayed) that can be used to derive the conditional density function, 
$f(x|y)$, without having to complete squares (as is standard in books like Christopher Bishop's PRML).

$$
\begin{align}
\vec{x} \in \mathbb{R}^{p x 1}, f_{\vec{x}}(\vec{x}) = \frac{1}{\sqrt{2\pi^p}} \frac{1}{\sqrt{\det{R_{\vec{x}}}}} \exp{(-\frac{1}{2}(\vec{x} - \mu_{\vec{x}})^TR_{vec{x}}^{-1}(\vec{x} - \mu_{\vec{x}}))} \\
\vec{y} \in \mathbb{R}^{q x 1}, f_{\vec{y}}(\vec{y}) = \frac{1}{\sqrt{2\pi^q}} \frac{1}{\det{\sqrt{R_{\vec{y}}}}} \exp{(-\frac{1}{2}(\vec{y} - \mu_{\vec{y}})^TR_{\vec{y}}^{-1}(\vec{y} - \mu_{\vec{y}}))} \\
f(\vec{x},\vec{y}) = \frac{1}{\sqrt{2\pi^{(p+q)}}} \frac{1}{\sqrt{\det{R}}} \exp{(-\frac{1}{2}(\begin{pmatrix} \vec{x} \\ \vec{y} \end{pmatrix} - \begin{pmatrix} \mu_{\vec{x}} \\ \mu_{\vec{y}} \end{pmatrix})^TR^{-1}(\begin{pmatrix} \vec{x} \\ \vec{y} \end{pmatrix} - \begin{pmatrix} \mu_{\vec{x}} \\ \mu_{\vec{y}} \end{pmatrix}))} \label{imp} \\
R = \begin{bmatrix} R_{\vec{x}} & R_{\vec{x}\vec{y}} \\ R_{\vec{y}\vec{x}} & R_{\vec{y}} \end{bmatrix} \label{eqR}
\end{align}
$$

Note that $R$ can be decomposed as a product of three matrices:

$$
\begin{align}
R = \begin{bmatrix} I_p & R_{\vec{x}\vec{y}}R_{\vec{y}}^{-1} \\ 0 & I_q \end{bmatrix} \begin{bmatrix} \Sigma_{\vec{x}} & 0 \\ 0 & R_{\vec{y}} \end{bmatrix} \begin{bmatrix} I_p & 0 \\ R_{\vec{y}}^{-1}R_{\vec{y}\vec{x}} & I_q \end{bmatrix} \\
\Sigma_{\vec{x}} = R_{\vec{x}} - R_{\vec{x}\vec{y}}R_{\vec{y}}^{-1}R_{\vec{y}\vec{x}}
\end{align}
$$

> Substitute the above decomposition of $R$ to equation $\eqref{imp}$, and compute the form of 
>$f(\vec{x}|\vec{y})$ by simple matrix algebra. Also find the mean and covariance of the distribution. You will notice that without the need for completion of squares, the above decomposition enables a smooth factorization of $f(\vec{x},\vec{y})$ into $f(\vec{y})$ and $f(\vec{x}|\vec{y})$!
{: .prompt-exercise }

The conditional distribution will also be a gaussian! The computation of the conditional mean estimate is quite simple. We just need to find the term that accompanies $x$ in the difference computation that is there in gaussian form. You will end up with:

$$
\begin{align}
\mu_{\vec{x}|\vec{y}} = \mu_{\vec{x}} + R_{\vec{x}\vec{y}}R_{\vec{y}}^{-1}R_{\vec{y}\vec{x}} \\
R_{\vec{x}|\vec{y}} = \Sigma_{\vec{x}} 
\end{align}
$$

> To internalize this decomposition idea, try computing $f(y|x)$ (along with its mean and covariance) using the same method as in the previous exercise. You need to start with decomposing $R$ in a different way, but as a factorization into three matrices as before!
> <details>
> <summary>Hint</summary>
> The block diagonal matrix in between becomes $\begin{bmatrix} R_{\vec{x}} & 0 \\ 0 & \Sigma_{\vec{y}} \end{bmatrix}$ where $\Sigma_{\vec{y}} = R_{\vec{y}} - R_{\vec{y}\vec{x}}R_{\vec{x}}^{-1}R_{\vec{x}\vec{y}}$. Try inferring the other two matrices to get $R$ and proceed from thereon.
{: .prompt-exercise }

> (a) Show $R_{\vec{x}} = R_{\tilde{x}} + R_{\hat{x}}$, where $\tilde{x} = \vec{x} - \hat{x}$, and $\hat{x} = \mathbb{E}[\vec{x}|\vec{y}]$ for jointly gaussian case. Notice that it is analogous to the law of total variance! \\
> (b) What is $R_{\tilde{x}}$ equal to? Do some basic algebra with the expression from (a) and apply the results derived for conditional gaussian pdf to obtain an equivalent expression for the covariance of the residue. Compare with variance of residue in the scalar case (with observations).
{: .prompt-exercise }



## BIAS-VARIANCE RELATION

We could comfortably identify the optimal estimates by computing the conditional mean estimates under the MSE risk. This was possible as we had to access to the distribution of $x$ (this helped in computing 
$f(x|y)$ using Bayes', and then we computed the expectation of the posterior distribution). We just wanted to estimate realizations of $x$ given that we observed a related quantity $y$. In real world, you will not have access to the distribution of the hidden variable $x$ (The optimal estimates obtained so far can be termed "theoretical" in this sense). Instead, we assume a prior on $x$ and proceed to estimate the posterior (and hence the conditional mean estimate). That may not give you optimal estimates. Based on how bad the MSE risk is, you will have to re-design the prior to inch towards reducing the MSE risk. This is an iterative process, and also considered an art! 

What if we had a $K$-size dataset $D$ of the realizations from $f(x)$, instead of $f(x)$ itself (or any assumptions of it thereof)? From this, we can compute $y$ for each $x$ in $D$ as they are related to each other. $D$ inherits stochasticity from $x$ and $y$. We can write the optimal "empirical" estimate in this case as 
$$\hat{x}_{d} = c_{D}(y)$$. 
With access to the underlying distribution of $x$, the minimum expected risk was 
$\mathbb{E}[(x - \mathbb{E}[x|y])^2]$ (MMSE). 
Using 
$$\hat{x}_{d}$$ in place of the "theoretical" optimal estimate, the expected risk will be atleast $$\mathbb{E}[(x - \mathbb{E}[x|y])^2]$$. An insightful question to ask here is what the expected risk will be, 
with $$\hat{x}_{d}$$ as the optimal estimate, $$\mathbb{E}[(x - c_{D}(y))^2]$$, where the expectation is over $x$, $y$, and $D$. 

> Theorem: \\
> $$\mathbb{E}[(x - c_{D}(y))^2] = \underbrace{\mathbb{E}[(x - \mathbb{E}[x|y])^2]}_{\text{MMSE}} + \underbrace{\mathbb{E}[(\mathbb{E}[x|y] - \mathbb{E}[c_{D}(y)])^2]}_{\text{Bias}} + \underbrace{\mathbb{E}[(c_{D}(y) - \mathbb{E}[c_{D}(y)])^2]}_{\text{Variance}}$$

In the theorem, it is shown that by using the optimal empirical estimate obtained via $D$, the theoretical MMSE is increased by two quantities: Bias and Variance. Intuitively, bias tells us how far away from the theoretical optimal estimate 
$\mathbb{E}[x|y]$ is the expected estimate 
$\mathbb{E}[c_{D}(y)]$ on an average. Variance indicates the spread of the obtained estimates across datasets $D$.

**Proof:**

$$
\begin{align}
\mathbb{E}_{x,D}[(x - c_{D}(y))^2|y = y] &= \mathbb{E}[(x - \mathbb{E}[x|y] + \mathbb{E}[x|y] - c_{D}(y))^2|y=y] \qquad \text{(Adding and subtracting $\mathbb{E}[x|y]$)}\\
                                         &= \mathbb{E}_x[(x - \mathbb{E}[x|y])^2|y=y] + \mathbb{E}_D[(\mathbb{E}[x|y] - c_{D}(y))^2|y=y] + 2\underbrace{\mathbb{E}_{x,D}[(x - \mathbb{E}[x|y])(\mathbb{E}[x|y] - c_{D}(y))|y=y]}_{\text{0}} \label{eq_ex_1}\\
                                         &= \mathbb{E}_x[(x - \mathbb{E}[x|y])^2|y=y] + \mathbb{E}_D[(\mathbb{E}[x|y] - c_{D}(y))^2|y=y] \\
                                         &= \mathbb{E}_x[(x - \mathbb{E}[x|y])^2|y=y] + \mathbb{E}_D[((\mathbb{E}[x|y] - \mathbb{E}[c_{D}(y)]) - (c_{D}(y) - \mathbb{E}[c_{D}(y)]))^2|y=y] \qquad \text{(Adding and subtracting $\mathbb{E}[c_{D}(y)]$ in the second term)}\\
                                         &= \mathbb{E}_x[(x - \mathbb{E}[x|y])^2|y=y] + \mathbb{E}_D[(\mathbb{E}[x|y] - \mathbb{E}[c_{D}(y)])^2|y=y] + \mathbb{E}_D[(c_{D}(y) - \mathbb{E}[c_{D}(y)])^2|y=y] - 2\underbrace{\mathbb{E}_D[(\mathbb{E}[x|y] - \mathbb{E}[c_{D}(y)])(c_{D}(y) - \mathbb{E}[c_{D}(y)])|y=y]}_{\text{0}} \label{eq_ex_2} \\
\mathbb{E}_{x,D}[(x - c_{D}(y))^2|y = y] &= \mathbb{E}_x[(x - \mathbb{E}[x|y])^2|y=y] + \mathbb{E}_D[(\mathbb{E}[x|y] - \mathbb{E}[c_{D}(y)])^2|y=y] + \mathbb{E}_D[(c_{D}(y) - \mathbb{E}[c_{D}(y)])^2|y=y] \label{imp_step}\\
\end{align}
$$

Taking outer expectation over $y$ on both sides of $\eqref{imp_step}$, we get:

$$
\begin{align}
\mathbb{E}[(x - c_{D}(y))^2] = \underbrace{\mathbb{E}[(x - \mathbb{E}[x|y])^2]}_{\text{MMSE}} + \underbrace{\mathbb{E}[(\mathbb{E}[x|y] - \mathbb{E}[c_{D}(y)])^2]}_{\text{Bias}} + \underbrace{\mathbb{E}[(c_{D}(y) - \mathbb{E}[c_{D}(y)])^2]}_{\text{Variance}}
\end{align}
$$

> Show how the third and fourth term in equations $\eqref{eq_ex_1}$ and $\eqref{eq_ex_2}$ vanish to $0$.
{: .prompt-exercise }

```python
"""
VISUALIZING BIAS-VARIANCE RELATION USING THE EXERCISE ON ESTIMATING CONDITIONAL MEAN ESTIMATE OF BERNOULLI HIDDEN VARIABLE UNDER ADDITIVE GAUSSIAN NOISE
"""

import matplotlib.pyplot as plt
import numpy as np

test_dataset_size = 1000
n_datasets = 50                                                             # Number of different trials/datasets to obtain different estimators
n_entries_per_dataset = 100
std_dev_noise = 1                                                           # Std Dev of noise under consideration
degree_polynomial_estimators = [1, 3, 5, 7]                                 # Using odd degree polynomial estimators as tanh is an odd function
x_test = np.random.choice([-1,+1], size=test_dataset_size)
n_test = np.random.normal(0, std_dev_noise, size=test_dataset_size)
y_test = x_test + n_test
x_hat_theoretical = np.tanh(y_test/(std_dev_noise**2))                      # Theoretical optimal estimate (conditional mean estimate) on the test data

train_datasets = []
for i in range(n_datasets):                                                 # Generating different training datasets from the same distribution for obtaining different estimators 
    x = np.random.choice([-1,+1], size = n_entries_per_dataset)
    n = np.random.normal(0, std_dev_noise, size = n_entries_per_dataset)
    y = x + n
    train_datasets.append(np.column_stack((np.ones(y.shape[0]),y,x)))

theoretical_optimal_mmse = np.mean((x_test - x_hat_theoretical)**2)         # The first term in the bias-variance relation

bias_across_estimators = []
variance_across_estimators = []
mse_across_estimators = []
weight_variance_across_estimators = []
residues_across_estimators = []
for i in degree_polynomial_estimators:
    weights_across_datasets = []
    mse_across_datasets = []
    if i == 1:
        for j in range(n_datasets):
            cols = train_datasets[j]
            X = cols[:, :-1]
            y = cols[:, -1]
            lmse_weights_poly_estimator = np.linalg.pinv(X) @ y             # Using the LMSE solution to get the weights of the polynomial estimator
            weights_across_datasets.append(lmse_weights_poly_estimator)
    else:        
        for j in range(n_datasets):
            y_new = np.array([train_datasets[j][:,1]**m for m in range(2,i+1)]).T
            cols = np.column_stack((train_datasets[j][:,:-1], y_new, train_datasets[j][:, -1]))
            X = cols[:, :-1]
            y = cols[:, -1]
            lmse_weights_poly_estimator = np.linalg.pinv(X) @ y
            weights_across_datasets.append(lmse_weights_poly_estimator)
    average_estimator_across_datasets = np.mean(np.array(weights_across_datasets), axis=0)      
    y_new = np.array([y_test**m for m in range(i+1)])
    mean_x_hat_estimator = y_new.squeeze().T @ average_estimator_across_datasets  # This approximates the expectation of the estimators (E[c_D(y)])
    bias = np.mean((x_hat_theoretical - mean_x_hat_estimator)**2)
    bias_across_estimators.append(bias)
    weights_across_datasets = np.array(weights_across_datasets).squeeze()
    weight_variance_across_estimators.append(np.mean(np.var(weights_across_datasets, axis=0)))
    estimates_across_datasets = y_new.T @ weights_across_datasets.T
    variance = np.mean(np.var(estimates_across_datasets, axis = 1))
    variance_across_estimators.append(variance)
    mse = np.mean((estimates_across_datasets - x_test[:, None])**2)                # MSE reported here is the sum total of the theoretical optimal MSE, bias and variance
    mse_across_estimators.append(mse)
    residue = x_test[:, None] - estimates_across_datasets
    residues_across_estimators.append(residue) 
    
bias_across_estimators = np.array(bias_across_estimators)
variance_across_estimators = np.array(variance_across_estimators)
mse_across_estimators = np.array(mse_across_estimators)
weight_variance_across_estimators = np.array(weight_variance_across_estimators)
plt.xticks(degree_polynomial_estimators)
x = degree_polynomial_estimators
plt.figure()
plt.plot(x, bias_across_estimators.squeeze(), label='Bias as computed using a test set', marker='o')
plt.plot(x, variance_across_estimators.squeeze(), label='Variance as computed using a test set', marker='s')
plt.plot(x, mse_across_estimators.squeeze(), label='MSE as computed using a test set', marker='s')
plt.plot(x, np.array([theoretical_optimal_mmse,]*len(x)).squeeze(), label=f'Theoretical Optimal MMSE : {theoretical_optimal_mmse}', marker='*')
plt.xlabel('Order of estimator')
plt.ylabel('Bias-Variance relation component values')
plt.legend()

plt.figure()
plt.xticks(degree_polynomial_estimators)
plt.plot(x, weight_variance_across_estimators.squeeze(), label='Variance of learned weights for different estimators', marker='s')
plt.xlabel('Order of estimator')
plt.ylabel('Variance of learned weights')
plt.legend()

plt.figure()
for idx, order in enumerate(degree_polynomial_estimators):
    residue_vals = residues_across_estimators[idx].flatten()
    y_coords = np.full(residue_vals.shape, order)
    plt.scatter(residue_vals, y_coords, s=5)
plt.yticks(degree_polynomial_estimators)
plt.xlabel('real axis')
plt.ylabel('order')
plt.title('scatter diagrams of estimation errors')

plt.show()
```

![Bias-Variance Relation Visualization](assets/images/bias_variance.png)

It can be observed that as the order of the estimator increases, the value of bias decreases and the value of variance increases. The total MSE, which is a sum of the theoretical MMSE, bias, and variance, is dominated by the variance value at higher orders. In other words, with a more "complex" model, we are able to get closer to the theoretical estimate, $\tanh(\frac{y}{\mathbb{Var}[n]})$, on an average. But different estimators do not seem to agree more with each other on an average. The reason why the variance increases with order of the estimator is patent when we observe the figure below. As the order increases, the variance of the learned weights increases a lot. As the learned weights differ significantly across different datasets $D$, the estimators will also be different significantly. The learned weights vary a lot in higher orders (more than 2) when compared with linear estimator. 

> In the code above, the test dataset is independent of the train_datasets. Wherever $D$ is mentioned in the explanation, it refers to the train_datasets. As $D$ changes, so will the estimators change. The test dataset is fixed.
{: .prompt-info}

![Variance of learned weights](assets/images/variance_of_learned_weights.png)

We can now come to the point of concern with data-driven estimators, the residue. Instead of a single theoretical estimator, we now have a group of estimators for different $D$. Given an observation $y$ in the test set, there will $L$ (which in the code is n_datasets) estimators. We need to find the inferred $\hat{x}$ for a given estimator across all observations. From thereon, the residue $x - \hat{x}$ can be computed. The same computation must be carried out for all $L$ estimators of a particular order. In the scatter figure below, the residues on the test set are plotted across the aforementioned estimators for different orders. With the number of samples per dataset set in the code, we can observe large variance in the residue as we increase order of the estimator. 

![Variance of residue across estimators](assets/images/residue_scatter.png)

> Play with the code above to dig deeper into the bias-variance relations. Answer questions like 'Will bias always decrease as variance increases?' by making empirical observations (tweaking the values of relevant variables in the code above). Try to reason out the behavior of the residue across estimators of different orders (Maybe increase number of samples per dataset (n_entries_per_dataset)).
{: .prompt-exercise}

## EPILOGUE

Even if we have access to the true distribution of $x$, it is not always easy to analytically evaluate (may not have a closed-form expression) the posterior distribution (and hence the conditional mean estimate). With Gaussians, we were fortunate as seen in this post. In the next article, we will cover smart techniques like Laplace, Importance Sampling, Metropolis-Hastings and Gibbs Sampling — methods that allow us to sample from complex, intractable posteriors and approximate their expectations.

## REFERENCES

> Sayed, A. H. (2022). *Inference and Learning from Data: Inference*. Cambridge University Press.

## AI USAGE DISCLOSURE

None of the above content was written using AI. The codes were also written manually. ChatGPT was used as a reference aid for syntax lookup, but not for code generation. Credits are due to Claude for aiding in construction of example related to the variance of residue being same as the variance of input despite there being dependence between the hidden and observable variable.