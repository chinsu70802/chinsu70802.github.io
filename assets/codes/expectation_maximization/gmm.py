"""
EXPECTATION-MAXIMIZATION AS APPLIED TO GAUSSIAN MIXTURE MODELS
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
    )
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
random_mixture_probabilities = np.array([1/K,]*K)      # Mixture probabilities random initialization

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
    )
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
diff_list = []
diff_cov_list = []
diff_mix_list = []
for i in range(1, len(state_tracker)):
    diff = state_tracker[i][0] - state_tracker[i - 1][0]
    diff_cov = state_tracker[i][1] - state_tracker[i - 1][1]
    diff_mix = state_tracker[i][2] - state_tracker[i - 1][2]
    diff_list.append(np.linalg.norm(diff))
    diff_cov_list.append(np.linalg.norm(diff_cov))
    diff_mix_list.append(np.linalg.norm(diff_mix))

plt.figure()
plt.plot(diff_list, label='Means')
plt.plot(diff_cov_list, label='Covariance')
plt.plot(diff_mix_list, label='Mixture prob')
plt.legend()

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