"""
Pricing Analysis: create plots for different serverless,spot and on-demand VMs scenario

VM and spot : M/G/1 model

USAGE: python generate_plots_mg1_spot.py <exp_type>

Author: Kunal Mahajan
mkunal@cs.columbia.edu
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy
import math

markers = ['s', 'h', '^', '*', 'o', 'p', '+', 'x', '<', 'D', '>', 'v', 'd', 0, 5, 2, 7, 1, 4, 3, 6, '1', '2', '3', '4', '8']

def solve_quadratic_eq(a,b,c):
	d = (b**2)-(4*a*c) # discriminant
	# string = "a: " + str(a) + "\tb: " + str(b) + "\tc: " + str(c) + "\td: " + str(d)
	# print string
	if d < 0:
		# print "This equation has no real solution"
		return [] 
	elif d == 0:
		x = (-b+math.sqrt(d))/(2*a)
		# print "This equation has one solutions: ", x
		if x >= 0:
			return [x]
		else:
			return []
	else:
		x1 = (-b+math.sqrt(d))/(2*a)
		x2 = (-b-math.sqrt(d))/(2*a)
		# print "This equation has two solutions: ", x1, " and", x2
		return [min(x1,x2)]

# Vary serverless to on-demand cost
def AO_vary_price_ratio():
	max_lambda = 29
	lambdas = [.01*x for x in range(0,((max_lambda+1)*100))]

	# Number of spot VMs:
	N_SPOT = 2
	# Max capacity of VM allowed:
	beta = 0.9 		

	# Cost
	alpha_v = 1.0
	omega_savings_percent = 0.75
	alpha_spot = (1-omega_savings_percent) * alpha_v

	# price_ratios = [10,10,10,10]
	price_ratios = [1,1.3,3,10]

	# Service rates
	mu_v = 8.0
	mu_sl = 10.0
	delta = 1
	mu_spot = delta*mu_v

	# Current Number of VMs:
	n_spot = 0
	n_v = 0
	
	# Transition points for VMs
	L_v = beta * mu_v
	L_spot = beta * mu_spot
	prev_transition_point = 0

	# Previous transition cost:
	prev_transition_cost = 0

	results = []
	message = ""

	# prev_transition_cost = n_spot * alpha_spot
	for ratio in price_ratios:
		n_spot = 0
		# print message
		message = ""
		print "price ratio = %f" % ratio
		results_price = []
		alpha_sl = ratio * alpha_v
		L_sl_v = (alpha_v/alpha_sl)*mu_sl
		L_sl_spot = (alpha_spot/alpha_sl)*mu_sl
		prev_transition_point = N_SPOT*L_spot
		prev_transition_cost = N_SPOT*alpha_spot
		# Ls = ((alpha_v*mu_s)/(alpha_s*mu_v))*(mu_v + gamma) - gamma
		for val_lambda in lambdas:
 			cost = 0
			if val_lambda > prev_transition_point:		# spot instances are all USED UP. Start using on-demand VMs
				val_lambda -= prev_transition_point
				if L_v <= L_sl_v:						# Case 1: spot service capacity is lower than transition point from Serverless to Spot
					cost = prev_transition_cost + alpha_sl*(val_lambda/mu_sl)
				else:										# Case 2: identify how many spots to use
					k = math.floor(val_lambda/L_v)
					S = 0			# number of VMs, workload to serverless
					if val_lambda-(math.floor(val_lambda/L_v)*L_v) < L_sl_v:	
						n_v,S = k,val_lambda-(math.floor(val_lambda/L_v)*L_v)
						cost = prev_transition_cost + (alpha_v*n_v) + (alpha_sl*(S/mu_sl))
					else:
						n_v,S = k+1,0
						cost = prev_transition_cost + (alpha_v*n_v) + (alpha_sl*(S/mu_sl))
			else:							# spot instances are available for use
				if L_spot <= L_sl_spot:						# Case 1: spot service capacity is lower than transition point from Serverless to Spot
					cost = alpha_sl*(val_lambda/mu_sl)
				else:										# Case 2: identify how many spots to use
					k = math.floor(val_lambda/L_spot)
					S = 0			# number of VMs, workload to serverless
					if val_lambda-(math.floor(val_lambda/L_spot)*L_spot) < L_sl_spot:	
						n_spot,S = k,val_lambda-(math.floor(val_lambda/L_spot)*L_spot)
						cost = (alpha_spot*n_spot) + (alpha_sl*(S/mu_sl))
					else:
						n_spot,S = k+1,0
						cost = (alpha_spot*n_spot) + (alpha_sl*(S/mu_sl))
													
			results_price.append(cost)
		results.append(results_price)
	filename = '../graphs/mg1_spot/AO_vary_price_ratios'  + '.png'
	fig = plt.figure()
	legends = []
	for ratio in price_ratios:
		key = r'$\alpha_s$=' + str(ratio) + r'$\alpha_v$'
		legends.append(key)
	plt.plot(lambdas[::200], results[0][::200], 'c*', markersize=7)
	plt.plot(lambdas[::200], results[1][::200], 'ro', markersize=7)
	plt.plot(lambdas[::200], results[2][::200], 'g^', markersize=7)
	plt.plot(lambdas[::200], results[3][::200], 'bs', markersize=7)
	plt.plot(lambdas, results[0], 'c', linewidth='2')
	plt.plot(lambdas, results[1], 'r', linewidth='2')
	plt.plot(lambdas, results[2], 'g', linewidth='2')
	plt.plot(lambdas, results[3], 'b', linewidth='2')

	plt.legend(legends, loc='upper left', fontsize=21)
	plt.ylabel('Cost', fontsize=25)
	# title = "Multiple VMs, different price ratios"
	# fig.suptitle(title)
	plt.xlabel(r'$\lambda$', fontsize=25)
	plt.savefig(filename)



def main():
	if len(sys.argv) != 2:
		print "USAGE: python generate_plots_mg1_spot.py <exp_type>"
		print "<exp_type> : AO_vary_price_ratio/multipleVMs_vary_mu/multipleVMonoff_vary_price_ratio/multipleVMonoff_vary_mu/multipleVMonoff_vary_gamma"
		print "<exp_type> : costserv_to_vm_ON/costserv_to_vm_ON_OFF"
		print "<exp_type> : mus_to_muv_ON_OFF"
		return
	exp_type = sys.argv[1]
	if exp_type == 'AO_vary_price_ratio':
		AO_vary_price_ratio()
	elif exp_type == 'multipleVMs_vary_mu':
		plotmultipleVMs_vary_mu()
	elif exp_type == 'multipleVMonoff_vary_price_ratio':
		plotmultipleVMonoff_vary_price_ratio()
	elif exp_type == 'multipleVMonoff_vary_mu':
		plotmultipleVMonoff_vary_mu()
	elif exp_type == 'multipleVMonoff_vary_gamma':
		plotmultipleVMonoff_vary_gamma()
	elif exp_type == 'costserv_to_vm_ON':
		plotcostserv_to_vm_ON()
	elif exp_type == 'costserv_to_vm_ON_OFF':
		plotcostserv_to_vm_ON_OFF()
	elif exp_type == 'mus_to_muv_ON_OFF':
		plotmus_to_muv_ON_OFF()
	else:
		print "Wrong <exp_type>"
		print "<exp_type> : AO_vary_price_ratio/multipleVMs_vary_mu/multipleVMonoff_vary_price_ratio/multipleVMonoff_vary_mu/multipleVMonoff_vary_gamma"
		print "<exp_type> : costserv_to_vm_ON/costserv_to_vm_ON_OFF"
		print "<exp_type> : mus_to_muv_ON_OFF"

if __name__ == '__main__':
	main()