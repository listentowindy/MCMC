# THIS SCRIPT INCLUDES A CLASS THAT BUILDS A METROPOLIS-HASTINGS 
# MCMC METHOD.


import numpy as np
import os
import time
from scipy import optimize 
import toolbox


class MC:
	def __init__(self, chi, X0, target_ratio, niter, nburn, sigmas, save_path=None, verbose=0):
		self.X0 = np.array(X0)
		self.chi = chi
		self.p0 = chi(self.X0)
		self.ndim = self.X0.size
		self.pos_chain = np.array([self.X0])
		self.prob_chain = np.array([self.p0])
		self.acceptance_index = np.zeros(0)
		self.target_ratio = target_ratio
		self.niter = niter
		self.nburn  = nburn
		self.sigmas = np.array([sigmas])
		self.reject_count = 0
		self.save_path = save_path
		self.verbose = verbose
	#
	def acc(self):
		return (self.acceptance_index.size*1.) / self.pos_chain.shape[0]
	#
	def accept(self,i,p1):
		self.pos_chain = np.vstack((self.pos_chain, self.proposal)); 
		self.acceptance_index =np.append(self.acceptance_index,i)
		self.prob_chain = np.append(self.prob_chain, p1)
		self.reject_count = 0
	#
	def reject(self, Xc, pc):
		self.pos_chain = np.vstack((self.pos_chain, Xc))
		self.prob_chain = np.append(self.prob_chain, pc)
		self.reject_count += 1
	#
	def draw_prop(self, Xc, sigmasc):
		return np.random.normal(Xc, sigmasc) # this returns a vector if the arguments are vectors
	#
	def makechain(self):
		for i in range(self.niter):
			Xc = self.pos_chain[-1]
			sigmasc = self.sigmas[-1]
			pc = self.prob_chain[-1]
			self.proposal = self.draw_prop(Xc, sigmasc) 
			p1 = self.chi(self.proposal )
			if self.verbose==1: print('trial chi2 is '+str(p1)+' and the current good one is '+str(pc))
			logr = p1-pc
			if logr < 0 : 
				self.accept(i,p1)
			else: 
				u = np.random.random()
				if u<np.exp(-logr): 
					self.accept(i,p1)
				else: 
					self.reject(Xc, pc)
			#
			if i==self.nburn or i==2*self.nburn:
				argmin_x = np.argmin(self.prob_chain)
				min_x = self.prob_chain[argmin_x]
				yy = np.exp(- self.prob_chain[-self.nburn:] + min_x)
				var = np.zeros(self.ndim)
				for k in range(self.ndim): 
					xx = self.pos_chain[:,k]
					mean_x = np.mean(xx)
					def err_func(sig): 
						return sum( (toolbox.a_gaussian(xx,1,self.pos_chain[argmin_x,k],sig) - yy)**2 )
					param_space_size = max( max(xx)-min(xx) , 1e-2)
					var[k] = np.var(xx)
				#
				stddev = np.sqrt(var)
				print('New stddev is '+str(stddev))
				self.acc_f = (np.where(self.acceptance_index>(i-self.nburn))[0].size+1.0) / self.pos_chain[-self.nburn:].shape[0]
				fact = (self.acc_f/self.target_ratio)
				new_stddev = stddev * fact
				self.sigmas = np.vstack((self.sigmas, new_stddev))
			#
			if i%self.nburn==0 and i>2*self.nburn:
				self.acc_f = (np.where(self.acceptance_index>(i-self.nburn))[0].size+1.0) / self.pos_chain[-self.nburn:].shape[0]
				self.sigmas = np.vstack((self.sigmas, self.sigmas[-1]*(self.acc_f/self.target_ratio)   ))
				if self.verbose==1:  print self.acc_f/self.target_ratio
			#
			if self.save_path != None:
				np.save(self.save_path, np.hstack((self.pos_chain, self.prob_chain.reshape(self.prob_chain.size,1))) )
			#
		return self.pos_chain, self.prob_chain





