---

layout: post
title: "Expectation Maximization"
date: 2026-06-10
categories: [Statistics]
tags: [Statistics]
math: True
---

## PROLOGUE

Recall the noisy-channel model of speech recongition discussed as an example to motivate the act of inference [Mean Square Inference](https://chinsu70802.github.io/posts/statistical-inference-1/). In that example, we described a world where people cannot hear what others say, but can see the acoustic waveform of the word uttered. The best possible guess of the word was mentioned as 
$\arg\max\limits_{x}p(x \| y) = \arg\max\limits_{x} p(y \| x) \cdot p(x)$, where $x$ and $y$ refer to the word and the observed waveform respectively. Let the waveform correspond to the word 'feign' (which means to pretend). The pronounciation of this word is similar to 'fane' and 'fain' (both of which are old english words that mean temple and happy respectively). Using prior knowledge (the fact that old english words are rarely used in conversation), it is easy to eliminate 'fane' and 'fain' as $p(x)$ is going to be very less. Hence, estimated $x$ will be 'feign'. This estimate is called the maximum a-posteriori (MAP) estimate. What if people do not form any prior belief (or consider anything and everything to be equally likely apriori) in this world? The best guess now becomes $\arg\max\limits_{x} p(y \| x)$ (The conditional probability in the objective is called likelihood). They need to rely on the observation $y$ to infer $x$. It is possible that 'fain' best explains $y$ compared to others. Hence, estimated $x$ is 'fain'. This estimate of $x$ is called as the maximum likelihood (ML) estimate. 

Notice that there can be a dramatic difference between ML and MAP estimates. This was a very high-level example. So, it may not be possible to fully appreciate it. Let us turn to a simple mathematical example surrounding coin tosses. We observe a coin with bias $k$ (the probability of heads is $k$) being tossed $n$ times independently. The $n$ observations are denoted as $y$ (which is the sequence 
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

> Vary the beta prior and derive $k$.
{: .prompt-exercise}

When the world allows enforcement of a prior belief on the hidden factor, we call that world as Bayesian. The estimation or inference performed by people (the citizens are called bayesians) in this bayesian world is called Bayesian Inference. We have been dealing with this in the previous blog posts. 

In this blog post, we enter into the world with no prior beliefs. People living in this world are called empiricists/frequentists (we will be called that for the duration of this blog), and any sort of estimation or inference done by the people in this world is called Frequentist Inference. They will have to repeatedly toss coins to understand the uncertainity surrounding the best estimate $k$ (They infer based on experiments and observations, hence the name empiricist). In one round of $n$ tosses, they may have seen all heads and estimated $k$ according to it. In another round, they may see only a few heads and a lot of tails. Based on this, their best estimate will change accordingly (mostly $k$ reduces from its earlier value of $1$). Their best estimates will probably have significant variance across rounds until realistic sequences of heads/tails start showing up. The bayesians do not suffer this trouble as they intuitively come up with a prior belief, and reduce experiment repetions as much as possible. If repeated experiments is the bottleneck that empiricists face, the design of a good prior is a difficulty that the bayesians face. If the real world coin is biased towards heads, and the bayesians design a prior that favors tails, their estimation will not be good (though they will stay consistent with the wrong estimate across rounds, hence reduced variance). 

People living in the Bayesian world faced issues with intractability of the posterior distributions. We introduced Approximate Inference to help the bayesians (people living in that world) solve the problem. Do frequentists also face such issues?

## EXAMPLE: GAUSSIAN MIXTURE MODEL



## EPILOGUE

## REFERENCES

> Sayed, A. H. (2022). *Inference and Learning from Data: Inference*. Cambridge University Press.



## AI USAGE DISCLOSURE