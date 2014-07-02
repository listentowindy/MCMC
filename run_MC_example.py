# EXAMPLE usage for the MCMC engine

import numpy as np
import toolbox

# Define the log-likelihood metric (chi square), should return some positive scalar 
def lnprob(param_vector):
	return   sum(param_vector)

# initial parameter vector, your best starting guess
initial_vector= [some values]
# variance for the initial parameter vector. Will be used to draw 
# new vectors in a way that the acceptance fraction converges to the desired value
sigma_vector = [some other values]
target_acceptance_fraction = 0.3
# number of steps in the chain
niterations = 1e6
# burn-in number
nburn_in = 1e3

# Initialize MCMC object
mc_obj = MC_engine.MC(lnprob, initial_vector,  target_acceptance_fraction, niterations, nburn_in, sigma_vector, 'save_path')
# Run chains
positions, probabilities = mc_obj.makechain()

# Fetch acceptance fraction
facc = mc_obj.acc()
print("Mean acceptance fraction:", facc)


# Plot the chains
toolbox.mcplot(positions,probs=probabilities,  ranges=3, labels=None, nticks = 4, smooth = 3, bins=(25,25))



