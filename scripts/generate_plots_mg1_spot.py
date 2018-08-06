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

# AO, no failure for spot instances, Vary serverless to on-demand cost
def AO_vary_price_ratio_no_fail():
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
	filename = '../graphs/mg1_spot/AO_vary_price_ratios_no_fail'  + '.png'
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

# AO, failure rate for spot instances
def AO_vary_failure_rate():
	max_lambda = 19
	lambdas = [.01*x for x in range(0,((max_lambda+1)*100))]

	# Number of spot VMs:
	N_SPOT = 3
	# Max capacity of VM allowed:
	beta = 0.9 		

	# Cost
	alpha_v = 1.0
	omega_savings_percent = 0.75
	alpha_spot = (1-omega_savings_percent) * alpha_v
	alpha_sl = 3*alpha_v	

	# Failure rate
	failure_rates = [0,0.1,0.2,0.4]

	# Service rates
	mu_v = 8.0
	mu_sl = 10.0
	delta = 0.8
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
	for fail_rate in failure_rates:
		n_spot = 0
		print "failure rate = %f" % fail_rate
		results_price = []

		L_sl_v = (alpha_v/alpha_sl)*mu_sl
		L_sl_spot = (alpha_spot/alpha_sl)*mu_sl
		prev_transition_point = N_SPOT*L_spot
		prev_transition_cost = N_SPOT*alpha_spot
	
		for val_lambda in lambdas:
			cost = 0
			val_lambda += val_lambda*fail_rate
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
					prev_transition_point = 0
					prev_transition_cost = 0
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
	filename = '../graphs/mg1_spot/AO_vary_failure_rate'  + '.png'
	fig = plt.figure()
	legends = []
	for rate in failure_rates:
		key = 'f = ' + str(rate)
		legends.append(key)
	plt.plot(lambdas[::100], results[0][::100], 'c*', markersize=7)
	plt.plot(lambdas[::100], results[1][::100], 'ro', markersize=7)
	plt.plot(lambdas[::100], results[2][::100], 'g^', markersize=7)
	plt.plot(lambdas[::100], results[3][::100], 'bs', markersize=7)
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

# OO, no failure for spot instances, Vary serverless to on-demand cost
def OO_vary_price_ratio_no_fail():
	max_lambda = 29
	lambdas = [.01*x for x in range(0,((max_lambda+1)*100))]

	# Number of spot VMs:
	N_SPOT = 3
	# Max capacity of VM allowed:
	beta = 0.9 		
	# Startup delay
	gamma = 1.0
	
	# Cost
	alpha_v = 1.0
	omega_savings_percent = 0.5
	alpha_spot = (1-omega_savings_percent) * alpha_v

	# price_ratios = [1.5,1.5,1.5,1.5]
	price_ratios = [1,1.5,2.5,10]

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
		L_sl_v = ((alpha_v*mu_sl)/(alpha_sl*mu_v))*(mu_v + gamma) - gamma
		L_sl_spot = ((alpha_spot*mu_sl)/(alpha_sl*mu_spot))*(mu_spot + gamma) - gamma
		prev_transition_point = N_SPOT*L_spot
		prev_transition_cost = N_SPOT*alpha_spot
		
		for val_lambda in lambdas:
 			cost = 0
			if val_lambda > prev_transition_point:		# spot instances are all USED UP. Start using on-demand VMs
				val_lambda -= prev_transition_point
				if L_v <= L_sl_v:						# Case 1: spot service capacity is lower than transition point from Serverless to Spot
					cost = prev_transition_cost + alpha_sl*(val_lambda/mu_sl)
				else:										# Case 2: identify how many spots to use
					k = math.floor(val_lambda/L_v)
					S = 0			# number of VMs, workload to serverless
					extra_lambda = val_lambda-(math.floor(val_lambda/L_v)*L_v)
					if extra_lambda < L_sl_v:
						C_v = alpha_v*((L_v*(mu_v + gamma))/((mu_v*(L_v + gamma))))*k
						cost = prev_transition_cost + C_v + (alpha_sl*(extra_lambda/mu_sl))
					else:
						C_v = alpha_v*((L_v*(mu_v + gamma))/((mu_v*(L_v + gamma))))*k
						cost = prev_transition_cost + C_v + (alpha_v*((extra_lambda*(mu_v + gamma))/((mu_v*(extra_lambda + gamma)))))
			else:							# spot instances are available for use
				if L_spot <= L_sl_spot:	# Case 1: spot service capacity is lower than transition point from Serverless to Spot
					cost = alpha_sl*(val_lambda/mu_sl)
					prev_transition_cost = 0
					prev_transition_point = 0
				else:										# Case 2: identify how many spots to use
					k = math.floor(val_lambda/L_spot)
					S = 0			# number of VMs, workload to serverless
					extra_lambda = val_lambda-(math.floor(val_lambda/L_spot)*L_spot)
					if extra_lambda < L_sl_spot:
						C_spot = alpha_spot*((L_spot*(mu_spot + gamma))/((mu_spot*(L_spot + gamma))))*k
						cost = C_spot + (alpha_sl*(extra_lambda/mu_sl))
					else:
						C_spot = alpha_spot*((L_spot*(mu_spot + gamma))/((mu_spot*(L_spot + gamma))))*k
						cost = C_spot + (alpha_spot*((extra_lambda*(mu_spot + gamma))/((mu_spot*(extra_lambda + gamma)))))
													
			results_price.append(cost)
		results.append(results_price)
	filename = '../graphs/mg1_spot/OO_vary_price_ratios_no_fail'  + '.png'
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

# OO, failure rate for spot instances
def OO_vary_failure_rate():
	max_lambda = 19
	lambdas = [.01*x for x in range(0,((max_lambda+1)*100))]

	# Number of spot VMs:
	N_SPOT = 3
	# Max capacity of VM allowed:
	beta = 0.9 		
	# startup delay
	gamma = 1

	# Cost
	alpha_v = 1.0
	omega_savings_percent = 0.75
	alpha_spot = (1-omega_savings_percent) * alpha_v
	alpha_sl = 2.5*alpha_v	

	# Failure rate
	failure_rates = [0,0.1,0.2,0.4]

	# Service rates
	mu_v = 8.0
	mu_sl = 10.0
	delta = 0.8
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
	for fail_rate in failure_rates:
		n_spot = 0
		print "failure rate = %f" % fail_rate
		results_price = []

		L_sl_v = ((alpha_v*mu_sl)/(alpha_sl*mu_v))*(mu_v + gamma) - gamma
		L_sl_spot = ((alpha_spot*mu_sl)/(alpha_sl*mu_spot))*(mu_spot + gamma) - gamma		
		prev_transition_point = N_SPOT*L_spot
		prev_transition_cost = N_SPOT*alpha_spot
		
		for val_lambda in lambdas:
 			cost = 0
 			val_lambda += val_lambda*fail_rate
			if val_lambda > prev_transition_point:		# spot instances are all USED UP. Start using on-demand VMs
				val_lambda -= prev_transition_point
				if L_v <= L_sl_v:						# Case 1: spot service capacity is lower than transition point from Serverless to Spot
					cost = prev_transition_cost + alpha_sl*(val_lambda/mu_sl)
				else:										# Case 2: identify how many spots to use
					k = math.floor(val_lambda/L_v)
					S = 0			# number of VMs, workload to serverless
					extra_lambda = val_lambda-(math.floor(val_lambda/L_v)*L_v)
					if extra_lambda < L_sl_v:
						C_v = alpha_v*((L_v*(mu_v + gamma))/((mu_v*(L_v + gamma))))*k
						cost = prev_transition_cost + C_v + (alpha_sl*(extra_lambda/mu_sl))
					else:
						C_v = alpha_v*((L_v*(mu_v + gamma))/((mu_v*(L_v + gamma))))*k
						cost = prev_transition_cost + C_v + (alpha_v*((extra_lambda*(mu_v + gamma))/((mu_v*(extra_lambda + gamma)))))
			else:							# spot instances are available for use
				if L_spot <= L_sl_spot:	# Case 1: spot service capacity is lower than transition point from Serverless to Spot
					cost = alpha_sl*(val_lambda/mu_sl)
					prev_transition_cost = 0
					prev_transition_point = 0
				else:										# Case 2: identify how many spots to use
					k = math.floor(val_lambda/L_spot)
					S = 0			# number of VMs, workload to serverless
					extra_lambda = val_lambda-(math.floor(val_lambda/L_spot)*L_spot)
					if extra_lambda < L_sl_spot:
						C_spot = alpha_spot*((L_spot*(mu_spot + gamma))/((mu_spot*(L_spot + gamma))))*k
						cost = C_spot + (alpha_sl*(extra_lambda/mu_sl))
					else:
						C_spot = alpha_spot*((L_spot*(mu_spot + gamma))/((mu_spot*(L_spot + gamma))))*k
						cost = C_spot + (alpha_spot*((extra_lambda*(mu_spot + gamma))/((mu_spot*(extra_lambda + gamma)))))
													
			results_price.append(cost)
		
		results.append(results_price)
	filename = '../graphs/mg1_spot/OO_vary_failure_rate'  + '.png'
	fig = plt.figure()
	legends = []
	for rate in failure_rates:
		key = 'f = ' + str(rate)
		legends.append(key)
	plt.plot(lambdas[::100], results[0][::100], 'c*', markersize=7)
	plt.plot(lambdas[::100], results[1][::100], 'ro', markersize=7)
	plt.plot(lambdas[::100], results[2][::100], 'g^', markersize=7)
	plt.plot(lambdas[::100], results[3][::100], 'bs', markersize=7)
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
		print "<exp_type> : AO_vary_price_ratio_no_fail/AO_vary_failure_rate"
		print "<exp_type> : OO_vary_price_ratio_no_fail/OO_vary_failure_rate"
		return
	exp_type = sys.argv[1]
	if exp_type == 'AO_vary_price_ratio_no_fail':
		AO_vary_price_ratio_no_fail()
	elif exp_type == 'AO_vary_failure_rate':
		AO_vary_failure_rate()
	elif exp_type == 'OO_vary_price_ratio_no_fail':
		OO_vary_price_ratio_no_fail()
	elif exp_type == 'OO_vary_failure_rate':
		OO_vary_failure_rate()
	else:
		print "Wrong <exp_type>"
		print "<exp_type> : AO_vary_price_ratio_no_fail/AO_vary_failure_rate"
		print "<exp_type> : OO_vary_price_ratio_no_fail/OO_vary_failure_rate"

if __name__ == '__main__':
	main()