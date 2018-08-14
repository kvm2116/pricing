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
from matplotlib.patches import Rectangle


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
			# val_lambda = calculate_lambda(val_lambda, fail_rate, mu_spot, mu_v, N_SPOT, beta)
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

def find_min_cost(val_lambda, num_instances, alpha_instance, mu_instance, alpha_sl, mu_sl, beta, gamma):
	L_instance = beta * mu_instance
	min_num_spot = num_instances-1
	min_num_spot_serverless = val_lambda - (min_num_spot*L_instance)
	min_spot_serv_cost = (alpha_instance*((L_instance*(mu_instance + gamma))/((mu_instance*(L_instance + gamma))))*min_num_spot) + (alpha_sl*(min_num_spot_serverless/mu_sl))		
	min_spot_cost = (alpha_instance*((L_instance*(mu_instance + gamma))/((mu_instance*(L_instance + gamma))))*min_num_spot)
	min_serv_cost = (alpha_sl*(min_num_spot_serverless/mu_sl))

	spot_serv_cost_k_1 = (alpha_instance*((L_instance*(mu_instance + gamma))/((mu_instance*(L_instance + gamma))))*min_num_spot) + (alpha_instance*((min_num_spot_serverless*(mu_instance + gamma))/((mu_instance*(min_num_spot_serverless + gamma)))))
	if min_spot_serv_cost > spot_serv_cost_k_1:
		min_num_spot += 1
		min_num_spot_serverless = 0
		min_spot_serv_cost = spot_serv_cost_k_1
		min_spot_cost = spot_serv_cost_k_1
		min_serv_cost = 0
	# print "min_num_spot=%f\tmin_num_spot_serverless=%f\min_spot_serv_cost=%f" % \
					 # (min_num_spot, min_num_spot_serverless, min_spot_serv_cost)
	return min_num_spot, min_num_spot_serverless, min_spot_serv_cost, min_serv_cost, min_spot_cost

def inst_cost(val_lambda, num_instances, alpha_instance, mu_instance, beta, gamma):
	if num_instances == 0:
		return 0
	if val_lambda != 0:
		val_lambda = val_lambda/num_instances
	return (alpha_instance*((val_lambda*(mu_instance + gamma))/((mu_instance*(val_lambda + gamma))))*num_instances)

# return load served at full service capacity, load served at partial service capacity, load not served at all
def get_num_load(load,num_inst, mu_inst, beta):
	max_load = num_inst*mu_inst*beta
	if load >= max_load:
		return max_load, 0, load - max_load 
	for count in range(1,num_inst+1):
		# print "count=%f" %count 
		can_serve = count * beta * mu_inst
		if load < can_serve:
			return (count-1)*beta*mu_inst, load-((count-1)*beta*mu_inst),0
	# return load,0,0

def calculate_vm_dist_cost(arrival_lambda, fail_rate, mu_spot, mu_v, mu_sl, N_SPOT, beta, alpha_v, alpha_sl, alpha_spot, gamma):
	L_v = beta * mu_v
	L_spot = beta * mu_spot

	new_lambda_only_spots = arrival_lambda/(1-fail_rate)
	num_spot_only_required = int(math.ceil(new_lambda_only_spots/L_spot))
	actual_spots = min(num_spot_only_required+1,N_SPOT+1)
	num_vm_only_required = int(math.ceil(arrival_lambda/L_v))

	if num_spot_only_required <= N_SPOT:
		# compare spot +serverless with OD + serverless
		min_num_spot, min_num_spot_serverless, min_spot_serv_cost, min_serv_cost, min_spot_cost = find_min_cost(new_lambda_only_spots, num_spot_only_required, alpha_spot, mu_spot, alpha_sl, mu_sl, beta, gamma)
		min_num_vm, min_num_vm_serverless, min_vm_serv_cost, min_servvm_cost, min_vm_cost = find_min_cost(arrival_lambda, num_vm_only_required, alpha_v, mu_v, alpha_sl, mu_sl, beta, gamma)
		serverless_only_cost = alpha_sl * arrival_lambda / mu_sl
		if min_vm_serv_cost <= min_spot_serv_cost:					# OD + serverless is cheaper than spot + serverless
			if serverless_only_cost <= min_vm_serv_cost:
				return 0,0,arrival_lambda, serverless_only_cost, serverless_only_cost, 0, 0
			else:
				return 0,min_num_vm, min_num_vm_serverless, min_vm_serv_cost, min_servvm_cost, 0, min_vm_cost
		else:
			if serverless_only_cost <= min_spot_serv_cost:
				return 0,0,arrival_lambda, serverless_only_cost, serverless_only_cost, 0, 0
			else:
				return min_num_spot, 0, min_num_spot_serverless, min_spot_serv_cost, min_serv_cost, min_spot_cost, 0

	num_vm = 0
	num_spot = 0
	num_serv = 0
	min_cost = 999999.0

	v,s = 0,0
	for v in range(0, num_vm_only_required+1):
		for s in range(0, actual_spots):
			val_lambda = arrival_lambda
			spot_load = 0
			serv_load = 0
			if s != 0:
				val_lambda = calc_lambda(arrival_lambda, fail_rate, v, s)
				spot_load = (float(s)/(float(v)+float(s)))*val_lambda 			
			vm_load = val_lambda - spot_load							
			spot_load_full,spot_load_part,excess_spot = get_num_load(spot_load, s, mu_spot, beta)
			# OPTION 1: putting excess load of spot and VM on serverless
			# vm_load_full,vm_load_part,excess_vm = get_num_load(vm_load, v, mu_v, beta)
			# serv_load = excess_spot + excess_vm		
			# OPTION 2: put excess load of spot on VM, then excess load of VM on serverless
			vm_load_full,vm_load_part,excess_vm = get_num_load(vm_load+excess_spot, v, mu_v, beta)
			serv_load = excess_vm
			if v != 0:
				vm_cost = inst_cost(vm_load_full, vm_load_full/(beta*mu_v), alpha_v, mu_v, beta, gamma) + \
							inst_cost(vm_load_part, 1.0, alpha_v, mu_v, beta, gamma)
				# vm_cost = inst_cost(vm_load_full, v, alpha_v, mu_v, beta, gamma)
			else:
				vm_cost = 0
			if s != 0:
				spot_cost = inst_cost(spot_load_full, spot_load_full/(beta*mu_spot), alpha_spot, mu_spot, beta, gamma) + \
								inst_cost(spot_load_part, 1.0, alpha_spot, mu_spot, beta, gamma)
				# spot_cost = inst_cost(spot_load_full, s, alpha_spot, mu_spot, beta, gamma)
			else:
				spot_cost = 0
			serv_cost = alpha_sl * serv_load / mu_sl
			if v > 0 and vm_load_full + vm_load_part <= (v-1)*beta*mu_v:
				continue
			if s > 0 and spot_load_full + spot_load_part <= (s-1)*beta*mu_spot:
				continue
			if fail_rate == 0.4:
				print "v=%f\ts=%f\tval_lam=%f\tv_f=%f\tv_p=%f\ts_f=%f\ts_p=%f\tsl=%f\tcost=%f\tmin_cost=%f\t" % \
					 (v,s,val_lambda,vm_load_full, vm_load_part,spot_load_full,spot_load_part, serv_load,serv_cost + spot_cost + vm_cost,min_cost)
			
			if min_cost > serv_cost + spot_cost + vm_cost:
				min_cost = serv_cost + spot_cost + vm_cost
				min_serv_cost = serv_cost
				min_spot_cost = spot_cost
				min_vm_cost = vm_cost
				num_vm = v
				num_spot = s
				num_serv = serv_load
			# if vm_load <= v*L_v:
			# 	break
	return num_spot, num_vm, num_serv, min_cost, min_serv_cost, min_spot_cost, min_vm_cost

def calculate_vm_dist_cost_old(arrival_lambda, fail_rate, mu_spot, mu_v, mu_sl, N_SPOT, beta, alpha_v, alpha_sl, alpha_spot, gamma):
	L_v = beta * mu_v
	L_spot = beta * mu_spot

	new_lambda_only_spots = arrival_lambda/(1-fail_rate)
	num_spot_only_required = math.ceil(new_lambda_only_spots/L_spot)
	num_vm_only_required = math.ceil(arrival_lambda/L_v)

	if num_spot_only_required <= N_SPOT:
		# compare spot +serverless with OD + serverless
		min_num_spot, min_num_spot_serverless, min_spot_serv_cost = find_min_cost(new_lambda_only_spots, num_spot_only_required, alpha_spot, mu_spot, alpha_sl, mu_sl, beta, gamma)
		min_num_vm, min_num_vm_serverless, min_vm_serv_cost = find_min_cost(arrival_lambda, num_vm_only_required, alpha_v, mu_v, alpha_sl, mu_sl, beta, gamma)
		serverless_only_cost = alpha_sl * arrival_lambda / mu_sl
		if min_vm_serv_cost <= min_spot_serv_cost:					# OD + serverless is cheaper than spot + serverless
			if serverless_only_cost <= min_vm_serv_cost:
				return 0,0,arrival_lambda, serverless_only_cost
			else:
				return 0,min_num_vm, min_num_vm_serverless, min_vm_serv_cost
		else:
			if serverless_only_cost <= min_spot_serv_cost:
				return 0,0,arrival_lambda, serverless_only_cost
			else:
				return min_num_spot, 0, min_num_spot_serverless, min_spot_serv_cost
	else:												# Mix of spot, vm, serverless
		# return 0,0,0,0
		if fail_rate == 0.4:
			print "hello"
		num_vm = 0.0
		prev_cost = 99999999.0		
		min_cost = 9999999.0
		prev_serv_load = 0		
		# if fail_rate == 0.4:
		# 	print "prev_cost = %f\tmin_cost=%f" % (prev_cost,min_cost)
		while True:
			serv_load = 0
			
			new_lambda = calc_lambda(arrival_lambda, fail_rate, num_vm, N_SPOT)
			if fail_rate == 0.4:
				print "prev_cost = %f\tmin_cost=%f\tarrival_lambda=%f" % (prev_cost,min_cost,arrival_lambda)
			spot_load = (N_SPOT/(N_SPOT+num_vm))*new_lambda # Split this into Max_Spot + serverless
			vm_load = new_lambda - spot_load			# How to optimally split VM load into VM and serverless
			max_spot_load = L_spot* N_SPOT
			max_vm_load = L_v * num_vm
			# OPTION 1: Max_spot + serv + VM (putting excess load of spot on serverless)
			if vm_load > max_vm_load:
				serv_load = vm_load - max_vm_load
				vm_load = max_vm_load
			if spot_load > max_spot_load:
				serv_load += spot_load - max_spot_load
				spot_load = max_spot_load
			if fail_rate == 0.4:
				print "new_load=%f\tarrival_load=%f\tnum_vm=%f\tServ_load=%f\tvm_load=%f\tspot_load=%f\t" % \
					(new_lambda, arrival_lambda,num_vm, serv_load, vm_load, spot_load)

			spot_cost = inst_cost(spot_load, N_SPOT, alpha_spot, mu_spot, beta, gamma) 	# compute Spot cost
			vm_cost = inst_cost(vm_load, num_vm, alpha_v, mu_v, beta, gamma) 		# compute VM cost
			serv_cost = alpha_sl * serv_load / mu_sl 						# compute serv cost
			new_cost = spot_cost + vm_cost + serv_cost
			if fail_rate == 0.4:
				print "spot_cost=%f\tvm_cost=%f\tserv_cost=%f\tmin_cost=%f\tnew_cost=%f" % \
						(spot_cost,vm_cost,serv_cost,min_cost,new_cost)
			if min_cost > new_cost:
				print "here"
				min_cost = new_cost
				num_vm += 1
				prev_serv_load = serv_load
			else:
				return N_SPOT, num_vm, prev_serv_load, min_cost
			
			# OPTION 2: Max_spot + serv + VM (putting excess load of spot on VM)			
			# if spot_load > max_spot_load:
			# 	vm_load += spot_load - max_spot_load
			# 	spot_load = max_spot_load
			# if vm_load > max_vm_load:
			# 	serv_load = vm_load - max_vm_load
			# 	vm_load = max_vm_load
			# spot_cost = inst_cost(spot_load, N_SPOT, alpha_spot, mu_spot, beta, gamma) 	# compute Spot cost
			# vm_cost = inst_cost(vm_load, num_vm, alpha_v, mu_v, beta, gamma) 		# compute VM cost
			# serv_cost = alpha_sl * serv_load / mu_sl 						# compute serv cost
			# new_cost = spot_cost + vm_cost + serv_cost
			# if min_cost < new_cost:
			# 	min_cost = new_cost
			# 	num_vm += 1
			# 	continue
			# else:
			# 	return N_SPOT, num_vm, serv_load, min_cost

def calc_lambda(arrival_lambda, f, num_vm, num_spot):
	# if f == 0.4:
		# print arrival_lambda, f, num_vm, num_spot
	return arrival_lambda/(1.0-(f*(float(num_spot)/(float(num_spot) + float(num_vm)))))

# OO, failure rate for spot instances
def OO_vary_failure_rate():
	max_lambda = 27
	lambdas = [.01*x for x in range(1,((max_lambda+1)*100))]

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
	failure_rates = [0.1,0.2,0.4]
	# failure_rates = [0.1,0.1,0.1,0.1]
	# Service rates
	mu_v = 8.0
	mu_sl = 10.0
	delta = 0.8
	mu_spot = delta*mu_v

	# Current Number of VMs:
	n_spot = 0
	n_v = 0
	
	# Transition points for VMs
	
	prev_transition_point = 0

	# Previous transition cost:
	prev_transition_cost = 0

	results = []
	message = ""

	# prev_transition_cost = n_spot * alpha_spot
	count = 0
	for fail_rate in failure_rates:
		n_spot = 0
		print "failure rate = %f" % fail_rate
		results_price = []
		quants = []
		count += 1
		
		for val_lambda in lambdas:
 			num_spot, num_vm, num_serverless, cost = calculate_vm_dist_cost(val_lambda, fail_rate, mu_spot, mu_v, mu_sl, N_SPOT, beta, alpha_v, alpha_sl, alpha_spot, gamma)
 			if count == 1:
 				print "Spot = %f\tOD = %f\tServ = %fval_lambda=%f\n\n" % (num_spot, num_vm, num_serverless,val_lambda)
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
	# plt.plot(lambdas[::100], results[3][::100], 'bs', markersize=7)
	plt.plot(lambdas, results[0], 'c', linewidth='2')
	plt.plot(lambdas, results[1], 'r', linewidth='2')
	plt.plot(lambdas, results[2], 'g', linewidth='2')
	# plt.plot(lambdas, results[3], 'b', linewidth='2')

	plt.legend(legends, loc='upper left', fontsize=21)
	plt.ylabel('Cost', fontsize=25)
	# title = "Multiple VMs, different price ratios"
	# fig.suptitle(title)
	plt.xlabel(r'$\lambda$', fontsize=25)
	plt.savefig(filename)

# OO, failure rate for spot instances, plot servers info
def OO_vary_failure_rate_servers(fail_rate):
	min_lambda = 1
	max_lambda = 27
	lambdas = [.01*x for x in range(min_lambda,((max_lambda+1)*100))]

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
	failure_rates = [fail_rate]
	# failure_rates = [0.1,0.1,0.1,0.1]
	# Service rates
	mu_v = 8.0
	mu_sl = 10.0
	delta = 0.8
	mu_spot = delta*mu_v

	# Current Number of VMs:
	n_spot = 0
	n_v = 0
	
	# Transition points for VMs
	
	prev_transition_point = 0

	# Previous transition cost:
	prev_transition_cost = 0

	results = []
	message = ""
	spots = []
	vms = []
	serv = []
	spot_cost = []
	serv_cost = []
	vm_cost = []

	# prev_transition_cost = n_spot * alpha_spot
	count = 0
	for fail_rate in failure_rates:
		n_spot = 0
		print "failure rate = %f" % fail_rate
		results_price = []
		quants = []
		count += 1
		
		for val_lambda in lambdas:
 			num_spot, num_vm, num_serverless, cost, min_serv_cost, min_spot_cost, min_vm_cost = calculate_vm_dist_cost(val_lambda, fail_rate, mu_spot, mu_v, mu_sl, N_SPOT, beta, alpha_v, alpha_sl, alpha_spot, gamma)
 			if count == 1:
 				print "Spot = %f\tOD = %f\tServ = %fval_lambda=%f\n\n" % (num_spot, num_vm, num_serverless,val_lambda)
			spots.append(num_spot)
			vms.append(num_vm)
			serv_cost.append(min_vm_cost + min_spot_cost + min_serv_cost)
			spot_cost.append(min_vm_cost + min_spot_cost)
			vm_cost.append(min_vm_cost)
			if num_serverless > 0:
				serv.append(1)
			else:
				serv.append(0)
			results_price.append(cost)
		
		results.append(results_price)
	filename = '../graphs/mg1_spot/OO_failure_rate_' + str(fail_rate) + '_servers'  + '.png'
	legends = ['OD cost', 'Spot cost', 'SC cost']
	ylabel = 'Cost'
	title = "Spot failure rate = %d%%" % int(fail_rate*100)
	location = "upper left"
	xlabel = r'$\lambda$'
	plot_graph(filename, legends, lambdas, vm_cost, spot_cost, serv_cost, xlabel, ylabel, title, location)

def plot_graph(filename, legends, xaxis, vm_cost, spot_cost, serv_cost, xlabel, ylabel, title, location):
	fig = plt.figure()
	# plt.plot(xaxis[::100], vm_cost[::100], 'g^', markersize=7)
	# plt.plot(xaxis[::100], spot_cost[::100], 'ro', markersize=7)
	# plt.plot(xaxis[::100], serv_cost[::100], 'bs', markersize=7)
	# plt.plot(xaxis, vm_cost, 'yellowgreen', linewidth='2')
	# plt.plot(xaxis, spot_cost, 'lightblue', linewidth='2')
	# plt.plot(xaxis, serv_cost, 'black', linewidth='2')
	yg = Rectangle((0, 0), 1, 1, fc="black", alpha=0.05)
	lb = Rectangle((0, 0), 1, 1, fc="black", alpha=0.5)
	b = Rectangle((0, 0), 1, 1, fc="black", alpha=1)
	new_legends = [yg,lb,b]
	plt.legend(new_legends, legends, loc=location, fontsize=21)
	fig.suptitle(title)
	# plt.autoscale(enable=False, axis='x')
	plt.fill_between(xaxis, 0, vm_cost, color='black', alpha='0.05')
	plt.fill_between(xaxis, vm_cost, spot_cost, color='black', alpha='0.5')
	plt.fill_between(xaxis, spot_cost, serv_cost, color='black', alpha='1')
	plt.xlim(0,xaxis[-1])
	plt.xlabel(xlabel, fontsize=25)
	plt.ylabel(ylabel, fontsize=25)
	plt.savefig(filename)

# OO, vary spot/VM price ratio
def OO_price_ratio_spot_od():
	val_lambda = 20

	omega_savings_percent = [.01*x for x in range(1,91)]

	# Number of spot VMs:
	N_SPOT = 3
	# Max capacity of VM allowed:
	beta = 0.9 		
	# startup delay
	gamma = 1

	# Cost
	alpha_v = 1.0
	alpha_sl = 2.5*alpha_v	

	# Failure rate
	fail_rate = 0.1
	# Service rates
	mu_v = 8.0
	mu_sl = 10.0
	delta = 0.8
	mu_spot = delta*mu_v

	# Current Number of VMs:
	n_spot = 0
	n_v = 0
	
	# Transition points for VMs
	
	prev_transition_point = 0

	# Previous transition cost:
	prev_transition_cost = 0

	results = []
	message = ""
	spots = []
	vms = []
	serv = []
	spot_cost = []
	serv_cost = []
	vm_cost = []

	# prev_transition_cost = n_spot * alpha_spot
	count = 0
	for omega_savings_ratio in omega_savings_percent:
		n_spot = 0
		# print "failure rate = %f" % fail_rate
		alpha_spot = (1-omega_savings_ratio) * alpha_v
		# results_price = []
		quants = []
		count += 1
		
 		num_spot, num_vm, num_serverless, cost, min_serv_cost, min_spot_cost, min_vm_cost = calculate_vm_dist_cost(val_lambda, fail_rate, mu_spot, mu_v, mu_sl, N_SPOT, beta, alpha_v, alpha_sl, alpha_spot, gamma)
		# 	if count == 1:
		# 		print "Spot = %f\tOD = %f\tServ = %fval_lambda=%f\n\n" % (num_spot, num_vm, num_serverless,val_lambda)
		spots.append(num_spot)
		vms.append(num_vm)
		serv_cost.append(min_vm_cost + min_spot_cost + min_serv_cost)
		spot_cost.append(min_vm_cost + min_spot_cost)
		vm_cost.append(min_vm_cost)
		if num_serverless > 0:
			serv.append(1)
		else:
			serv.append(0)
		# results_price.append(cost)
		
		results.append(cost)
	filename = '../graphs/mg1_spot/OO_price_ratio_spot_OD' + str(fail_rate) + '_servers'  + '.png'
	legends = ['OD cost', 'Spot cost', 'SC cost']
	ylabel = 'Cost'
	location = "upper right"
	xlabel = r'$\alpha_{spot}/\alpha_v$'
	title = ""
	plot_graph(filename, legends, omega_savings_percent, vm_cost, spot_cost, serv_cost, xlabel, ylabel, title, location)

# OO, vary spot/VM price ratio
def OO_failure_rate_impact(val_lambda):

	failure_rates = [.01*x for x in range(1,99)]

	# Number of spot VMs:
	N_SPOT = 3
	# Max capacity of VM allowed:
	beta = 0.9 		
	# startup delay
	gamma = 1

	# Cost
	alpha_v = 1.0
	alpha_sl = 2.5*alpha_v	
	omega_savings_percent = 0.75
	alpha_spot = (1-omega_savings_percent) * alpha_v

	# Failure rate
	fail_rate = 0.1
	# Service rates
	mu_v = 8.0
	mu_sl = 10.0
	delta = 0.8
	mu_spot = delta*mu_v

	# Current Number of VMs:
	n_spot = 0
	n_v = 0
	
	# Transition points for VMs
	
	prev_transition_point = 0

	# Previous transition cost:
	prev_transition_cost = 0

	results = []
	message = ""
	spots = []
	vms = []
	serv = []
	spot_cost = []
	serv_cost = []
	vm_cost = []

	# prev_transition_cost = n_spot * alpha_spot
	count = 0
	for fail_rate in failure_rates:
		n_spot = 0
		# print "failure rate = %f" % fail_rate
		
		# results_price = []
		quants = []
		count += 1
		
 		num_spot, num_vm, num_serverless, cost, min_serv_cost, min_spot_cost, min_vm_cost = calculate_vm_dist_cost(val_lambda, fail_rate, mu_spot, mu_v, mu_sl, N_SPOT, beta, alpha_v, alpha_sl, alpha_spot, gamma)
		print "Spot = %f\tOD = %f\tServ = %fval_lambda=%f\n\n" % (num_spot, num_vm, num_serverless,val_lambda)
		spots.append(num_spot)
		vms.append(num_vm)
		serv_cost.append(min_vm_cost + min_spot_cost + min_serv_cost)
		spot_cost.append(min_vm_cost + min_spot_cost)
		vm_cost.append(min_vm_cost)
		if num_serverless > 0:
			serv.append(1)
		else:
			serv.append(0)
		# results_price.append(cost)
		
		results.append(cost)
	filename = '../graphs/mg1_spot/OO_failure_rate_impact_lam_'  + str(val_lambda) + '.png'
	legends = ['OD cost', 'Spot cost', 'SC cost']
	ylabel = 'Cost'
	location = "upper left"
	xlabel = 'Spot failure rate %'
	title = ""
	plot_graph(filename, legends, failure_rates, vm_cost, spot_cost, serv_cost, xlabel, ylabel, title, location)


	# plt.plot(failure_rates[::5], results[::5], 'c*', markersize=7)
	# plt.plot(failure_rates[::5], spots[::5], 'ro', markersize=7)
	# plt.plot(failure_rates[::5], vms[::5], 'g^', markersize=7)
	# plt.plot(failure_rates[::5], serv[::5], 'bs', markersize=7)
	# # plt.plot(failure_rates, results, 'c', linewidth='2')
	# plt.plot(failure_rates, spots, 'r', linewidth='2')
	# plt.plot(failure_rates, vms, 'g', linewidth='2')
	# plt.plot(failure_rates, serv, 'b', linewidth='2')

	# plt.legend(legends, loc='upper right', fontsize=21)
	# plt.ylabel('Number of VMs', fontsize=25)
	# plt.autoscale(enable=False, axis='x')

	# # fig.suptitle(title)
	# plt.xlabel('Spot failure rate %', fontsize=25)
	# plt.savefig(filename)

def main():
	if len(sys.argv) < 2:
		print "USAGE: python generate_plots_mg1_spot.py <exp_type>"
		print "<exp_type> : AO_vary_price_ratio_no_fail/AO_vary_failure_rate"
		print "<exp_type> : OO_vary_price_ratio_no_fail/OO_vary_failure_rate/OO_vary_failure_rate_servers"
		print "<exp_type> : OO_price_ratio_spot_od/OO_failure_rate_impact"
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
	elif exp_type == 'OO_vary_failure_rate_servers':
		OO_vary_failure_rate_servers(float(sys.argv[2])) 
	elif exp_type == 'OO_price_ratio_spot_od':
		OO_price_ratio_spot_od()
	elif exp_type == 'OO_failure_rate_impact':
		OO_failure_rate_impact(float(sys.argv[2]))
	else:
		print "Wrong <exp_type>"
		print "<exp_type> : AO_vary_price_ratio_no_fail/AO_vary_failure_rate"
		print "<exp_type> : OO_vary_price_ratio_no_fail/OO_vary_failure_rate/OO_vary_failure_rate_servers"
		print "<exp_type> : OO_price_ratio_spot_od/OO_failure_rate_impact"

if __name__ == '__main__':
	main()