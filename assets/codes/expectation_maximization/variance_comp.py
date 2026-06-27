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
a = 3                   # Parameter of the beta distribution                
b = 2                   # Parameter of the beta distribution

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

