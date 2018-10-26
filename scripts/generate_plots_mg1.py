"""
Pricing Analysis: create plots for different serverless vs VMs scenario

VM : M/G/1 model

Usage: python generate_plots_mg1.py <exp_type>
<exp_type> : vary_num_VMs/vary_startup_delay/vary_service_rate_VM

Author: Kunal Mahajan
mkunal@cs.columbia.edu
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy
import math

markers = ['s', 'h', '^', '*', 'o', 'p', '+', 'x', '<', 'D', '>', 'v', 'd', 0, 5, 2, 7, 1, 4, 3, 6, '1', '2', '3', '4', '8']

def zipfdistribution():
	max_lambda = 10000
	delta = 0.9
	mu_v = 1.0
	mu_s = 2
	beta_v = 0.2
	beta_s = (mu_v/mu_s)*beta_v
	alpha_v = 1.0
	lambda_v = []
	lambda_s = []
	sc_load = []
	optimal_alpha_s = []
	exponent = [0.1*x for x in range(10,90)]
	print "lam\tp_sc\tp_scvm\tp_vm\ta_sc\ta_scvm\tmp\topt_as"
	for val_exponent in exponent:
		expected_lam = computeZipfConstant(max_lambda, val_exponent-1)/computeZipfConstant(max_lambda, val_exponent)
		opt_alpha_s = -1
		alpha_s_sc = mu_s/delta
		# p_sc = math.ceil(val_lambda/mu_s)*(alpha_s_sc - (beta_v/mu_s))
		p_sc = (expected_lam/mu_s)*(alpha_s_sc - (beta_v/mu_s))

		alpha_s_scvm = p_scvm = -1
		if expected_lam - (math.floor(expected_lam/(delta*mu_v))*(delta*mu_v)) != 0:
			alpha_s_scvm = mu_s/(expected_lam - ((math.floor(expected_lam/(delta*mu_v)))*(delta*mu_v)))
			p_scvm = (math.floor(expected_lam/(delta*mu_v))*(1-beta_v)) +   ((alpha_s_scvm - (beta_v/mu_s))*((expected_lam - ((math.floor(expected_lam/(delta*mu_v)))*(delta*mu_v)))/mu_s))
		p_vm = (1 + math.floor(expected_lam/(delta*mu_v)))*(1-beta_v)
		max_profit = max(p_sc, p_vm, p_scvm)
		if max_profit == p_sc:
			lambda_v.append(0)
			opt_alpha_s = alpha_s_sc
			lambda_s.append(expected_lam)
		elif max_profit == p_scvm:
			opt_alpha_s = alpha_s_scvm
			lambda_v.append(math.floor(expected_lam/(delta*mu_v))*delta*mu_v)
			lambda_s.append(expected_lam - ((math.floor(expected_lam/(delta*mu_v)))*(delta*mu_v)))
		else:
			opt_alpha_s = alpha_s_scvm
			lambda_v.append((1+math.floor(expected_lam/(delta*mu_v)))*delta*mu_v)
			lambda_s.append(0)		
		optimal_alpha_s.append(opt_alpha_s)
		# optimal_alpha_s.append(expected_lam)
		expected_lam = float("{0:.2f}".format(expected_lam))
		alpha_s_sc = float("{0:.2f}".format(alpha_s_sc))
		alpha_s_scvm = float("{0:.2f}".format(alpha_s_scvm))
		p_sc = float("{0:.2f}".format(p_sc))
		p_scvm = float("{0:.2f}".format(p_scvm))
		p_vm = float("{0:.2f}".format(p_vm))
		max_profit = float("{0:.2f}".format(max_profit))
		opt_alpha_s = float("{0:.2f}".format(opt_alpha_s))
		print str(expected_lam) + "\t" + str(p_sc) + "\t" + str(p_scvm) + "\t" + str(p_vm) + "\t" + str(alpha_s_sc) + "\t" + str(alpha_s_scvm) + "\t" +  str(max_profit) + "\t" +  str(opt_alpha_s)
			
	filename = '../graphs/mg1/zipf.png'
	fig = plt.figure()
	legends = []
	# plt.subplot(2,1,1)
	# for val_lambda in lambdas:
	# 	key = r'$\lambda$=' + str(val_lambda)
	# 	legends.append(key)
	plt.plot(exponent[::2], optimal_alpha_s[::2], 'c*', markersize=7)
	plt.plot(exponent, optimal_alpha_s, 'c', linewidth='2')
	plt.ylabel('Optimal ' + r'$\alpha_s$', fontsize=25)
	plt.xlabel('zipf exponent', fontsize=20)
	plt.savefig(filename)

def get_optimal_alpha_s_vary_beta_v():
	max_lambda = 15
	# lambdas = [x for x in range(1,max_lambda+1)]
	# lambdas = [x for x in range(1,max_lambda+1)]
	lambdas = [1,2,3,6]
	delta = 0.9
	mu_v = 1.0
	# mu_s_ratios = [1.3, 2, 3
	mu_s = 2.0
	beta_v_values = [0.01*x for x in range(10, 90)]
	
	alpha_v = 1.0
	lambda_v = []
	sc_load = []
	optimal_alpha_s = []
	print "lam\tp_sc\tp_scvm\tp_vm\ta_sc\ta_scvm\tmp\topt_as"
	for val_lambda in lambdas:
		lambda_s = []	
		results = []
		for beta_v in beta_v_values:
			beta_s = (mu_v/mu_s)*beta_v
			opt_alpha_s = -1
			alpha_s_sc = mu_s/delta
			# p_sc = math.ceil(val_lambda/mu_s)*(alpha_s_sc - (beta_v/mu_s))
			p_sc = (val_lambda/mu_s)*(alpha_s_sc - (beta_v/mu_s))

			alpha_s_scvm = p_scvm = -1
			if val_lambda - (math.floor(val_lambda/(delta*mu_v))*(delta*mu_v)) != 0:
				alpha_s_scvm = mu_s/(val_lambda - ((math.floor(val_lambda/(delta*mu_v)))*(delta*mu_v)))
				p_scvm = (math.floor(val_lambda/(delta*mu_v))*(1-beta_v)) +   ((alpha_s_scvm - (beta_v/mu_s))*((val_lambda - ((math.floor(val_lambda/(delta*mu_v)))*(delta*mu_v)))/mu_s))
			p_vm = (1 + math.floor(val_lambda/(delta*mu_v)))*(1-beta_v)
			max_profit = max(p_sc, p_vm, p_scvm)
			if max_profit == p_sc:
				lambda_v.append(0)
				opt_alpha_s = alpha_s_sc
				lambda_s.append(val_lambda)
			elif max_profit == p_scvm:
				opt_alpha_s = alpha_s_scvm
				lambda_v.append(math.floor(val_lambda/(delta*mu_v))*delta*mu_v)
				lambda_s.append(val_lambda - ((math.floor(val_lambda/(delta*mu_v)))*(delta*mu_v)))
			else:
				opt_alpha_s = alpha_s_scvm
				lambda_v.append((1+math.floor(val_lambda/(delta*mu_v)))*delta*mu_v)
				lambda_s.append(0)

			# alpha_s_sc = float("{0:.2f}".format(alpha_s_sc))
			# alpha_s_scvm = float("{0:.2f}".format(alpha_s_scvm))
			# p_sc = float("{0:.2f}".format(p_sc))
			# p_scvm = float("{0:.2f}".format(p_scvm))
			# p_vm = float("{0:.2f}".format(p_vm))
			# max_profit = float("{0:.2f}".format(max_profit))
			# opt_alpha_s = float("{0:.2f}".format(opt_alpha_s))
			# print str(val_lambda) + "\t" + str(p_sc) + "\t" + str(p_scvm) + "\t" + str(p_vm) + "\t" + str(alpha_s_sc) + "\t" + str(alpha_s_scvm) + "\t" +  str(max_profit) + "\t" +  str(opt_alpha_s)
			results.append(opt_alpha_s)
		optimal_alpha_s.append(results)
		sc_load.append(lambda_s)
	filename = '../graphs/mg1/optimal_alpha_vary_beta_v'  + '.png'
	fig = plt.figure()
	legends = []
	plt.subplot(2,1,1)
	for val_lambda in lambdas:
		key = r'$\lambda$=' + str(val_lambda)
		legends.append(key)
	plt.plot(beta_v_values[::5], optimal_alpha_s[0][::5], 'c*', markersize=7)
	plt.plot(beta_v_values[::5], optimal_alpha_s[1][::5], 'ro', markersize=7)
	plt.plot(beta_v_values[::5], optimal_alpha_s[2][::5], 'g^', markersize=7)
	plt.plot(beta_v_values[::5], optimal_alpha_s[3][::5], 'bs', markersize=7)
	plt.plot(beta_v_values, optimal_alpha_s[0], 'c', linewidth='2')
	plt.plot(beta_v_values, optimal_alpha_s[1], 'r', linewidth='2')
	plt.plot(beta_v_values, optimal_alpha_s[2], 'g', linewidth='2')
	plt.plot(beta_v_values, optimal_alpha_s[3], 'b', linewidth='2')

	plt.legend(legends, loc='upper right', fontsize=15)
	plt.ylabel('Optimal ' + r'$\alpha_s$', fontsize=25)


	# plt.subplot(2,1,2)
	# plt.plot(lambdas[::5], lambda_v[::5], 'c*', markersize=7)
	# plt.plot(lambdas, lambda_v, 'c', linewidth='2')
	# plt.ylabel(r'$\lambda_v$', fontsize=25)
	# plt.xlabel(r'$\lambda$', fontsize=25)
	# plt.savefig(filename)

	plt.subplot(2,1,2)
	plt.plot(beta_v_values[::5], sc_load[0][::5], 'c*', markersize=7)
	plt.plot(beta_v_values[::5], sc_load[1][::5], 'ro', markersize=7)
	plt.plot(beta_v_values[::5], sc_load[2][::5], 'g^', markersize=7)
	plt.plot(beta_v_values[::5], sc_load[3][::5], 'bs', markersize=7)
	plt.plot(beta_v_values, sc_load[0], 'c', linewidth='2')
	plt.plot(beta_v_values, sc_load[1], 'r', linewidth='2')
	plt.plot(beta_v_values, sc_load[2], 'g', linewidth='2')
	plt.plot(beta_v_values, sc_load[3], 'b', linewidth='2')
	plt.ylabel(r'$\lambda_s$', fontsize=25)
	plt.xlabel(r'$\beta_v$', fontsize=20)
	plt.savefig(filename)

def get_optimal_alpha_s():
	max_lambda = 15
	# lambdas = [x for x in range(1,max_lambda+1)]
	lambdas = [x for x in range(1,max_lambda+1)]
	delta = 0.9
	mu_v = 1.0
	mu_s_ratios = [1.3, 2, 3]
	# mu_s = 2.0
	beta_v = 0.2
	
	alpha_v = 1.0
	lambda_v = []
	sc_load = []
	optimal_alpha_s = []
	print "lam\tp_sc\tp_scvm\tp_vm\ta_sc\ta_scvm\tmp\topt_as"
	for ratio in mu_s_ratios:
		mu_s = ratio * mu_v
		beta_s = (mu_v/mu_s)*beta_v
		lambda_s = []	
		results = []
		for val_lambda in lambdas:
			opt_alpha_s = -1
			alpha_s_sc = mu_s/delta
			# p_sc = math.ceil(val_lambda/mu_s)*(alpha_s_sc - (beta_v/mu_s))
			p_sc = (val_lambda/mu_s)*(alpha_s_sc - (beta_v/mu_s))

			alpha_s_scvm = p_scvm = -1
			if val_lambda - (math.floor(val_lambda/(delta*mu_v))*(delta*mu_v)) != 0:
				alpha_s_scvm = mu_s/(val_lambda - ((math.floor(val_lambda/(delta*mu_v)))*(delta*mu_v)))
				p_scvm = (math.floor(val_lambda/(delta*mu_v))*(1-beta_v)) +   ((alpha_s_scvm - (beta_v/mu_s))*((val_lambda - ((math.floor(val_lambda/(delta*mu_v)))*(delta*mu_v)))/mu_s))
			p_vm = (1 + math.floor(val_lambda/(delta*mu_v)))*(1-beta_v)
			max_profit = max(p_sc, p_vm, p_scvm)
			if max_profit == p_sc:
				lambda_v.append(0)
				opt_alpha_s = alpha_s_sc
				lambda_s.append(val_lambda)
			elif max_profit == p_scvm:
				opt_alpha_s = alpha_s_scvm
				lambda_v.append(math.floor(val_lambda/(delta*mu_v))*delta*mu_v)
				lambda_s.append(val_lambda - ((math.floor(val_lambda/(delta*mu_v)))*(delta*mu_v)))
			else:
				opt_alpha_s = alpha_s_scvm
				lambda_v.append((1+math.floor(val_lambda/(delta*mu_v)))*delta*mu_v)
				lambda_s.append(0)

			alpha_s_sc = float("{0:.2f}".format(alpha_s_sc))
			alpha_s_scvm = float("{0:.2f}".format(alpha_s_scvm))
			p_sc = float("{0:.2f}".format(p_sc))
			p_scvm = float("{0:.2f}".format(p_scvm))
			p_vm = float("{0:.2f}".format(p_vm))
			max_profit = float("{0:.2f}".format(max_profit))
			opt_alpha_s = float("{0:.2f}".format(opt_alpha_s))
			print str(val_lambda) + "\t" + str(p_sc) + "\t" + str(p_scvm) + "\t" + str(p_vm) + "\t" + str(alpha_s_sc) + "\t" + str(alpha_s_scvm) + "\t" +  str(max_profit) + "\t" +  str(opt_alpha_s)
			results.append(opt_alpha_s)
		optimal_alpha_s.append(results)
		sc_load.append(lambda_s)
	filename = '../graphs/mg1/optimal_alpha'  + '.png'
	fig = plt.figure()
	legends = []
	plt.subplot(2,1,1)
	for ratio in mu_s_ratios:
		key = r'$\mu_s$=' + str(ratio) + r'$\mu_v$'
		legends.append(key)
	plt.plot(lambdas[::1], optimal_alpha_s[0][::1], 'c*', markersize=7)
	plt.plot(lambdas[::1], optimal_alpha_s[1][::1], 'ro', markersize=7)
	plt.plot(lambdas[::1], optimal_alpha_s[2][::1], 'g^', markersize=7)
	plt.plot(lambdas, optimal_alpha_s[0], 'c', linewidth='2')
	plt.plot(lambdas, optimal_alpha_s[1], 'r', linewidth='2')
	plt.plot(lambdas, optimal_alpha_s[2], 'g', linewidth='2')

	plt.legend(legends, loc='upper right', fontsize=15)
	plt.ylabel('Optimal ' + r'$\alpha_s$', fontsize=25)


	# plt.subplot(2,1,2)
	# plt.plot(lambdas[::5], lambda_v[::5], 'c*', markersize=7)
	# plt.plot(lambdas, lambda_v, 'c', linewidth='2')
	# plt.ylabel(r'$\lambda_v$', fontsize=25)
	# plt.xlabel(r'$\lambda$', fontsize=25)
	# plt.savefig(filename)

	plt.subplot(2,1,2)
	plt.plot(lambdas[::1], sc_load[0][::1], 'c*', markersize=7)
	plt.plot(lambdas[::1], sc_load[1][::1], 'ro', markersize=7)
	plt.plot(lambdas[::1], sc_load[2][::1], 'g^', markersize=7)
	plt.plot(lambdas, sc_load[0], 'c', linewidth='2')
	plt.plot(lambdas, sc_load[1], 'r', linewidth='2')
	plt.plot(lambdas, sc_load[2], 'g', linewidth='2')
	plt.ylabel(r'$\lambda_s$', fontsize=25)
	plt.xlabel(r'$\lambda$', fontsize=25)
	plt.savefig(filename)

def get_config(beta, mu_s, mu_v, alpha_v, alpha_s, gamma, val_lambda):
	Lv = beta * mu_v
	Ls = ((alpha_v*mu_s)/(alpha_s*mu_v))*(mu_v + gamma) - gamma	
	cost = 0
	num_vm = 0
	load_serv = 0
	# print Ls
	# if Ls < 0 :
	# 	print "Negative"
	if Lv <= Ls: 														# Serverless Always
		cost = alpha_s*(val_lambda/mu_s)
		load_serv = val_lambda
	else:																		# Serverless until VM on-off
		k = math.floor(val_lambda/Lv)
		extra_lambda = val_lambda-(math.floor(val_lambda/Lv)*Lv)
		if Ls > 0 and extra_lambda < Ls:	
			# print "VM(%d On-Off) + SC" % k
			Cv = alpha_v*((Lv*(mu_v + gamma))/((mu_v*(Lv + gamma))))*k
			cost = Cv + (alpha_s*extra_lambda/mu_s)
			load_serv = extra_lambda
			num_vm = k
		else:
			# print "VM(%d On-Off)" % (k+1)
			num_vm = k+1
			Cv = alpha_v*((Lv*(mu_v + gamma))/((mu_v*(Lv + gamma))))*k
			cost = Cv + (alpha_v*((extra_lambda*(mu_v + gamma))/((mu_v*(extra_lambda + gamma)))))
	return num_vm, load_serv, cost



def get_configAO(beta, mu_s, mu_v, alpha_v, alpha_s, gamma, val_lambda):
	Ls = float((alpha_v/float(alpha_s))*mu_s)
	Lv = float(beta*mu_v)
	# print "Ls = %d\tLv = %d" % (Ls, Lv)
	cost = 0
	num_vm = 0
	load_serv = 0
	if Lv <= Ls: 
		load_serv = val_lambda
		cost = alpha_s*(val_lambda/mu_s)
	else:
		k = math.floor(val_lambda/Lv)
		# print k
		if val_lambda-(math.floor(val_lambda/Lv)*Lv) < Ls:	
			num_vm = k
			load_serv = val_lambda-(math.floor(val_lambda/Lv)*Lv)
			# print "V+S"
		else:
			num_vm = k+1
			load_serv = 0
			# print "V"
		Cv = alpha_v*num_vm
		Cs = alpha_s*(load_serv/mu_s)
		cost = Cv + Cs
	return num_vm, load_serv, cost


def getCPCost(num_vm, load_serv, cp_alpha_v, cp_alpha_s, mu_s, mu_v, val_lambda, beta):
	# serv_cost = cp_alpha_s * (math.ceil((val_lambda / (beta*mu_s))) - num_vm)
	# serv_cost = cp_alpha_s * math.ceil(load_serv / mu_s) 
	serv_cost = cp_alpha_s * (load_serv / mu_s) 
	vm_cost = cp_alpha_v * num_vm
	return serv_cost + vm_cost

def plotSingleUserOptimalSCvaryCPcost():
	alpha_v = 1.0
	price_ratios = [0.01*x for x in range(100,300)]
	# price_ratios = [0.1*x for x in range(10,50)]
	mu_v = 10.0
	mu_s = 10.0
	cp_alpha_v = 0.25
	cp_ratios = [0.25, 0.5, 1, 1.5]
	cp_alpha_s = 0.375
	beta = 0.9
	gamma = 1
	total_num_vm = 0
	total_num_serv = 0
	results = []
	val_lambda = 6
	for cp_ratio in cp_ratios:
		cp_alpha_s = cp_alpha_v * cp_ratio
		results_num_vm = []
		results_num_serv = []
		results_total_servers = []
		results_cost = []
		for ratio in price_ratios:
			alpha_s = alpha_v * ratio		
			num_vm, num_serv, user_cost = get_config(beta, mu_s, mu_v, alpha_v, alpha_s, gamma, val_lambda)
			cp_cost = getCPCost(num_vm, num_serv, cp_alpha_v, cp_alpha_s, mu_s, mu_v, val_lambda, beta)		
			results_cost.append(user_cost - cp_cost)
			# print num_vm, num_serv
			# results_cost.append(user_cost)
		results.append(results_cost)
	filename = '../graphs/mg1/singleUserDelaysOptimalSCvaryCPcost'  + '.png'
	# filename = '../graphs/mg1/singleUserDelaysOptimalSCUser'  + '.png'
	fig = plt.figure()
	legends = []
	for cp_ratio in cp_ratios:
		key = r'$\alpha_{s\_cp}$=' + str(cp_ratio) + r'$\alpha_{v\_cp}$'
		legends.append(key)
	plt.plot(price_ratios[::20], results[0][::20], 'c*', markersize=7)
	plt.plot(price_ratios[::20], results[1][::20], 'ro', markersize=7)
	plt.plot(price_ratios[::20], results[2][::20], 'g^', markersize=7)
	plt.plot(price_ratios[::200], results[3][::200], 'bs', markersize=7)
	plt.plot(price_ratios, results[0], 'c', linewidth='2')
	plt.plot(price_ratios, results[1], 'r', linewidth='2')
	plt.plot(price_ratios, results[2], 'g', linewidth='2')
	plt.plot(price_ratios, results[3], 'b', linewidth='2')

	plt.legend(legends, loc='upper right', fontsize=21)
	plt.ylabel('Cloud Provider Profit', fontsize=25)
	# plt.ylabel('Revenue from User', fontsize=25)
	plt.xlabel(r'$\alpha_s$', fontsize=25)
	plt.savefig(filename)


def plotSingleUserOptimalSCvaryDelay():
	alpha_v = 1.0
	price_ratios = [0.01*x for x in range(100,300)]
	# price_ratios = [0.1*x for x in range(10,50)]
	mu_v = 10.0
	mu_s = 10.0
	cp_alpha_v = 0.25
	cp_alpha_s = 0.375
	beta = 0.9
	gamma = 1
	total_num_vm = 0
	total_num_serv = 0
	results = []
	val_lambda = 6
	gamma_ratios = [1, 0.5, 0.1]
	for gamma in gamma_ratios:
		results_num_vm = []
		results_num_serv = []
		results_total_servers = []
		results_cost = []
		for ratio in price_ratios:
			alpha_s = alpha_v * ratio		
			num_vm, num_serv, user_cost = get_config(beta, mu_s, mu_v, alpha_v, alpha_s, gamma, val_lambda)
			cp_cost = getCPCost(num_vm, num_serv, cp_alpha_v, cp_alpha_s, mu_s, mu_v, val_lambda, beta)		
			results_cost.append(user_cost - cp_cost)
			print num_vm, num_serv
			# results_cost.append(user_cost)
		results.append(results_cost)
	filename = '../graphs/mg1/singleUserDelaysOptimalSCvaryDelay'  + '.png'
	# filename = '../graphs/mg1/singleUserDelaysOptimalSCUser'  + '.png'
	fig = plt.figure()
	legends = []
	for gamma in gamma_ratios:
		key = '1/' + r'$\gamma$=' + str(1.0/gamma)
		legends.append(key)
	plt.plot(price_ratios[::20], results[0][::20], 'c*', markersize=7)
	plt.plot(price_ratios[::20], results[1][::20], 'ro', markersize=7)
	plt.plot(price_ratios[::20], results[2][::20], 'g^', markersize=7)
	# plt.plot(lambdas[::200], results[3][::200], 'bs', markersize=7)
	plt.plot(price_ratios, results[0], 'c', linewidth='2')
	plt.plot(price_ratios, results[1], 'r', linewidth='2')
	plt.plot(price_ratios, results[2], 'g', linewidth='2')
	# plt.plot(lambdas, results[3], 'b', linewidth='2')

	plt.legend(legends, loc='upper right', fontsize=21)
	plt.ylabel('Cloud Provider Profit', fontsize=25)
	# plt.ylabel('Revenue from User', fontsize=25)
	plt.xlabel(r'$\alpha_s$', fontsize=25)
	plt.savefig(filename)

def plotSingleUserOptimalSC():
	alpha_v = 1.0
	price_ratios = [0.01*x for x in range(100,1000)]
	# price_ratios = [0.1*x for x in range(10,50)]
	# mu_server = 30.0
	mu_v = 10.0
	mu_s = 10.0
	# eff_s = 5.0
	# eff_10 = 10.0

	cp_alpha_v = 0.2
	cp_alpha_s = 0.3
	beta = 0.9
	gamma = 1
	total_num_vm = 0
	total_num_serv = 0
	results = []
	lambdas = [4, 6, 8]
	# lambdas = [100]
	for val_lambda in lambdas:
		results_num_vm = []
		results_num_serv = []
		results_total_servers = []
		results_cost = []
		for ratio in price_ratios:
			alpha_s = alpha_v * ratio		
			num_vm, num_serv, user_cost = get_config(beta, mu_s, mu_v, alpha_v, alpha_s, gamma, val_lambda)
			cp_cost = getCPCost(num_vm, num_serv, cp_alpha_v, cp_alpha_s, mu_s, mu_v, val_lambda, beta)		
			results_cost.append(user_cost - cp_cost)
			print num_vm, num_serv
			# results_cost.append(user_cost)
		results.append(results_cost)
	filename = '../graphs/mg1/singleUserDelaysOptimalSC'  + '.png'
	# filename = '../graphs/mg1/singleUserDelaysOptimalSCUser'  + '.png'
	fig = plt.figure()
	legends = []
	for val_lambda in lambdas:
		key = r'$\lambda$=' + str(val_lambda)
		legends.append(key)
	plt.plot(price_ratios[::100], results[0][::100], 'c*', markersize=7)
	plt.plot(price_ratios[::100], results[1][::100], 'ro', markersize=7)
	plt.plot(price_ratios[::100], results[2][::100], 'g^', markersize=7)
	# plt.plot(lambdas[::200], results[3][::200], 'bs', markersize=7)
	plt.plot(price_ratios, results[0], 'c', linewidth='2')
	plt.plot(price_ratios, results[1], 'r', linewidth='2')
	plt.plot(price_ratios, results[2], 'g', linewidth='2')
	# plt.plot(lambdas, results[3], 'b', linewidth='2')

	plt.legend(legends, loc='upper left', fontsize=21)
	plt.ylabel('Cloud Provider Profit', fontsize=25)
	# plt.ylabel('Revenue from User', fontsize=25)
	plt.xlabel(r'$\alpha_s$', fontsize=25)
	plt.savefig(filename)


def getNumServers(num_vm, load_serv, mu_v, mu_s, mu_server, eff_v, eff_s):
	vm_per_server = mu_server/eff_v
	num_servers_for_vm_load = num_vm/vm_per_server
	sc_per_server = mu_server/eff_s
	num_servers_for_serv_load = math.ceil(load_serv/mu_s)/sc_per_server
	return math.ceil(num_servers_for_serv_load + num_servers_for_vm_load)

# 1: SC only, 2 : Hybrid, 3: VM only
def plotSingleUserEfficiencyVarySCcost():
	alpha_v = 1.0
	price_ratios = [0.01*x for x in range(100,400)]
	mu_server = 30.0
	mu_v = 5.0
	mu_s_ratios = [0.01*x for x in range(100,300)]
	# mu_s = 10.0
	eff_s = 5.0
	eff_v = 10.0
	cp_alpha_v = 0.2
	# cp_alpha_s = 0.3
	cp_cost_ratios = [0.5, 1.2, 1.5, 2.1]
	beta = 0.9
	gamma = 1
	total_num_vm = 0
	total_num_serv = 0
	results = []
	configs = []
	# lambdas = [4, 20, 55]
	val_lambda = 55
	num_servers = []
	# lambdas = [20, 60, 100]
	# lambdas = [100]

	for cp_ratio in cp_cost_ratios:
		cp_alpha_s = cp_ratio * cp_alpha_v
		results_config = []
		results_total_servers = []
		results_cost = []
		results_opt_alpha_s = []
		for mu_s_ratio in mu_s_ratios:
			mu_s = mu_s_ratio * mu_v
			max_profit = -10000000000000000
			optimal_alpha_s = -1
			config = 0
			for ratio in price_ratios:
				alpha_s = alpha_v * ratio
				num_vm, load_serv, revenue = 0,0,0
				num_vm, load_serv, revenue = get_config(beta, mu_s, mu_v, alpha_v, alpha_s, gamma, val_lambda)
				cp_cost = getCPCost(num_vm, load_serv, cp_alpha_v, cp_alpha_s, mu_s, mu_v, val_lambda, beta)		
				if revenue - cp_cost > max_profit:
					max_profit = revenue - cp_cost
					optimal_alpha_s = ratio
					if num_vm == 0:
						config = 1
					elif load_serv > 0:
						config = 2
					else:
						config = 3
			results_cost.append(max_profit)
			results_config.append(config)
			results_opt_alpha_s.append(optimal_alpha_s)
				# results_total_servers.append(getNumServers(num_vm, load_serv, mu_v, mu_s, mu_server, eff_v, eff_s))
				# print num_vm, load_serv
				# results_cost.append(cp_cost)
		results.append(results_opt_alpha_s)
		configs.append(results_config)
		# num_servers.append(results_total_servers)
	filename = '../graphs/mg1/SingleUserEfficiencyVarySCcost'+ '.png'
	# filename = '../graphs/mg1/singleUserDelaysOptimalSCUser'  + '.png'
	fig = plt.figure()
	legends = []
	# for val_lambda in lambdas:
	# 	key = r'$\lambda$=' + str(val_lambda)
	# 	legends.append(key)
	for ratio in cp_cost_ratios:
		key = r'$\alpha_{s\_cp}$=' + str(ratio) + r'$\alpha_{v\_cp}$'
		legends.append(key)
	plt.subplot(2, 1, 1)
	plt.plot(mu_s_ratios[::10], results[0][::10], 'c*', markersize=7)
	plt.plot(mu_s_ratios[::10], results[1][::10], 'ro', markersize=7)
	plt.plot(mu_s_ratios[::10], results[2][::10], 'g^', markersize=7)
	plt.plot(mu_s_ratios[::10], results[3][::10], 'bs', markersize=7)
	plt.plot(mu_s_ratios, results[0], 'c', linewidth='2')
	plt.plot(mu_s_ratios, results[1], 'r', linewidth='2')
	plt.plot(mu_s_ratios, results[2], 'g', linewidth='2')
	plt.plot(mu_s_ratios, results[3], 'b', linewidth='2')

	# plt.legend(legends, loc='lower right', fontsize=21)
	labelstr = 'Optimal ' + r'$\alpha_s$'
	# plt.ylabel('Cloud Provider Profit', fontsize=25)
	plt.ylabel(labelstr, fontsize=25)
	# plt.ylabel('Revenue from User', fontsize=25)
	# plt.xlabel(r'$\mu_s/\mu_v$', fontsize=25)

	plt.subplot(2, 1, 2)
	plt.plot(mu_s_ratios[::10], configs[0][::10], 'c*', markersize=7)
	plt.plot(mu_s_ratios[::10], configs[1][::10], 'ro', markersize=7)
	plt.plot(mu_s_ratios[::10], configs[2][::10], 'g^', markersize=7)
	plt.plot(mu_s_ratios[::10], configs[3][::10], 'bs', markersize=7)
	plt.plot(mu_s_ratios, configs[0], 'c', linewidth='2')
	plt.plot(mu_s_ratios, configs[1], 'r', linewidth='2')
	plt.plot(mu_s_ratios, configs[2], 'g', linewidth='2')
	plt.plot(mu_s_ratios, configs[3], 'b', markersize=7)

	# plt.legend(legends, loc='upper right', fontsize=21)
	plt.yticks([0,1,2,3,4], ('', 'SC', 'SC + VM', 'VM', ''))
	# plt.ylabel('Configuration', fontsize=25)
	plt.xlabel(r'$\mu_s/\mu_v$', fontsize=20)
	plt.legend(legends, loc='lower right', fontsize=15)

	plt.savefig(filename)


# 1: SC only, 2 : Hybrid, 3: VM only
def plotSingleUserEfficiencyVaryMu():
	alpha_v = 1.0
	price_ratios = [0.01*x for x in range(100,1000)]
	mu_server = 30.0
	mu_v = 5.0
	mu_s_ratios = [0.01*x for x in range(100,300)]
	# mu_s = 10.0
	eff_s = 5.0
	eff_v = 10.0
	cp_alpha_v = 0.2
	cp_alpha_s = 0.3
	# cp_cost_ratios = [0.5, 1, 1.5, 2.1]
	beta = 0.9
	gamma = 1
	total_num_vm = 0
	total_num_serv = 0
	results = []
	configs = []
	lambdas = [4, 20, 55]
	num_servers = []
	# lambdas = [20, 60, 100]
	# lambdas = [100]

	for val_lambda in lambdas:
		results_config = []
		results_total_servers = []
		results_cost = []
		results_opt_alpha_s = []
		for mu_s_ratio in mu_s_ratios:
			mu_s = mu_s_ratio * mu_v
			max_profit = -10000000000000000
			optimal_alpha_s = -1
			config = 0
			for ratio in price_ratios:
				alpha_s = alpha_v * ratio
				num_vm, load_serv, revenue = 0,0,0
				num_vm, load_serv, revenue = get_config(beta, mu_s, mu_v, alpha_v, alpha_s, gamma, val_lambda)
				cp_cost = getCPCost(num_vm, load_serv, cp_alpha_v, cp_alpha_s, mu_s, mu_v, val_lambda, beta)		
				if revenue - cp_cost > max_profit:
					max_profit = revenue - cp_cost
					optimal_alpha_s = ratio
					if num_vm == 0:
						config = 1
					elif load_serv > 0:
						config = 2
					else:
						config = 3
			results_cost.append(max_profit)
			results_config.append(config)
			results_opt_alpha_s.append(optimal_alpha_s)
				# results_total_servers.append(getNumServers(num_vm, load_serv, mu_v, mu_s, mu_server, eff_v, eff_s))
				# print num_vm, load_serv
				# results_cost.append(cp_cost)
		results.append(results_opt_alpha_s)
		configs.append(results_config)
		# num_servers.append(results_total_servers)
	filename = '../graphs/mg1/SingleUserEfficiencyVaryMu'+ '.png'
	# filename = '../graphs/mg1/singleUserDelaysOptimalSCUser'  + '.png'
	fig = plt.figure()
	legends = []
	for val_lambda in lambdas:
		key = r'$\lambda$=' + str(val_lambda)
		legends.append(key)
	plt.subplot(2, 1, 1)
	plt.plot(mu_s_ratios[::10], results[0][::10], 'c*', markersize=7)
	plt.plot(mu_s_ratios[::10], results[1][::10], 'ro', markersize=7)
	plt.plot(mu_s_ratios[::10], results[2][::10], 'g^', markersize=7)
	# plt.plot(mu_s_ratio[::100], results[3][::100], 'bs', markersize=7)
	plt.plot(mu_s_ratios, results[0], 'c', linewidth='2')
	plt.plot(mu_s_ratios, results[1], 'r', linewidth='2')
	plt.plot(mu_s_ratios, results[2], 'g', linewidth='2')
	# plt.plot(mu_s_ratio, results[3], 'b', linewidth='2')

	plt.legend(legends, loc='upper right', fontsize=21)
	labelstr = 'Optimal ' + r'$\alpha_s$'
	# plt.ylabel('Cloud Provider Profit', fontsize=25)
	plt.ylabel(labelstr, fontsize=25)
	# plt.ylabel('Revenue from User', fontsize=25)
	# plt.xlabel(r'$\mu_s/\mu_v$', fontsize=25)

	plt.subplot(2, 1, 2)
	plt.plot(mu_s_ratios[::10], configs[0][::10], 'c*', markersize=7)
	plt.plot(mu_s_ratios[::10], configs[1][::10], 'ro', markersize=7)
	plt.plot(mu_s_ratios[::10], configs[2][::10], 'g^', markersize=7)
	# plt.plot(lambdas[::200], results[3][::200], 'bs', markersize=7)
	plt.plot(mu_s_ratios, configs[0], 'c', linewidth='2')
	plt.plot(mu_s_ratios, configs[1], 'r', linewidth='2')
	plt.plot(mu_s_ratios, configs[2], 'g', linewidth='2')
	# plt.plot(lambdas, results[3], 'b', linewidth='2')

	# plt.legend(legends, loc='upper right', fontsize=21)
	plt.yticks([0,1,2,3,4], ('', 'SC', 'SC + VM', 'VM', ''))
	# plt.ylabel('Configuration', fontsize=25)
	plt.xlabel(r'$\mu_s/\mu_v$', fontsize=20)


	plt.savefig(filename)


def plotSingleUserEfficiency(mode):
	# val_lambda = 1
	alpha_v = 1.0
	price_ratios = [0.01*x for x in range(100,800)]
	mu_server = 30.0
	mu_v = 1.0
	mu_s = 2.0
	eff_s = 5.0
	eff_v = 10.0
	cp_alpha_v = 0.2
	cp_alpha_s = mu_v*cp_alpha_v/mu_s
	# cp_alpha_s = 0.3
	# cp_cost_ratios = [0.5, 1, 1.5, 2.1]
	cp_cost_ratios = [0.5]
	beta = 0.9
	gamma = 1
	total_num_vm = 0
	total_num_serv = 0
	results = []
	lambdas = [3, 10, 15]
	# lambdas = [3]
	
	num_servers = []
	# lambdas = [20, 60, 100]
	# lambdas = [100]
	for val_lambda in lambdas:
		# cp_alpha_s = cp_ratio * cp_alpha_v
		results_num_vm = []
		results_num_serv = []
		results_total_servers = []
		results_cost = []
		for ratio in price_ratios:
			alpha_s = alpha_v * ratio
			num_vm, load_serv, revenue = 0,0,0
			if mode == 'OO':		
				num_vm, load_serv, revenue = get_config(beta, mu_s, mu_v, alpha_v, alpha_s, gamma, val_lambda)
			elif mode == 'AO':
				num_vm, load_serv, revenue = get_configAO(beta, mu_s, mu_v, alpha_v, alpha_s, gamma, val_lambda)
				# print "here"
			cp_cost = getCPCost(num_vm, load_serv, cp_alpha_v, cp_alpha_s, mu_s, mu_v, val_lambda, beta)		
			results_cost.append(revenue - cp_cost)
			results_total_servers.append(getNumServers(num_vm, load_serv, mu_v, mu_s, mu_server, eff_v, eff_s))
			# print num_vm, load_serv
			# results_cost.append(cp_cost)
		results.append(results_cost)
		num_servers.append(results_total_servers)
	filename = '../graphs/mg1/singleUserEfficiency' + mode + '.png'
	# filename = '../graphs/mg1/singleUserDelaysOptimalSCUser'  + '.png'
	fig = plt.figure()
	plt.subplot(1,1,1)
	fig.text(0.06, 0.5, 'Cloud Provider Profit', ha='center', va='center', rotation='vertical', fontsize=25)
	# plt.ylabel('Cloud Provider Profit', fontsize=25)

	# for val_lambda in lambdas:
	# 	key = r'$\lambda$=' + str(val_lambda)
	# 	legends.append(key)
	# for ratio in cp_cost_ratios:
	# 	key = r'$\alpha_{s\_cp}$=' + str(ratio) + r'$\alpha_{v\_cp}$'
	# 	legends.append(key)
	plt.subplot(2,1,1)
	plt.plot(price_ratios[::90], results[0][::90], 'c*', markersize=7)
	# plt.plot(price_ratios[::100], results[1][::100], 'ro', markersize=7)
	# plt.plot(price_ratios[::100], results[2][::100], 'g^', markersize=7)
	# plt.plot(price_ratios[::100], results[3][::100], 'bs', markersize=7)
	plt.plot(price_ratios, results[0], 'c', linewidth='2')
	legends = []
	key = r'$\lambda$=' + str(lambdas[0])
	legends.append(key)
	plt.legend(legends, loc='lower right', fontsize=21)
	# plt.plot(price_ratios, results[1], 'r', linewidth='2')
	# plt.plot(price_ratios, results[2], 'g', linewidth='2')
	# plt.plot(price_ratios, results[3], 'b', linewidth='2')

	# plt.subplot(3,1,2)
	# plt.plot(price_ratios[::100], results[1][::100], 'ro', markersize=7)
	# plt.plot(price_ratios, results[1], 'r', linewidth='2')
	# legends = []
	# key = r'$\lambda$=' + str(lambdas[1])
	# legends.append(key)
	# plt.legend(legends, loc='lower right', fontsize=21)

	plt.subplot(2,1,2)
	plt.plot(price_ratios[::90], results[2][::90], 'g^', markersize=7)
	plt.plot(price_ratios, results[2], 'g', linewidth='2')
	legends = []
	key = r'$\lambda$=' + str(lambdas[2])
	legends.append(key)
	plt.legend(legends, loc='lower right', fontsize=21)

	# plt.legend(legends, loc='upper right', fontsize=21)
	# labelstr = 'Cloud Provider Profit for ' + r'$\lambda$' + ' = ' + str(val_lambda)
	# plt.ylabel(labelstr, fontsize=25)
	# plt.ylabel('Cloud Provider Profit', fontsize=25)
	# plt.ylabel('Revenue from User', fontsize=25)
	plt.xlabel(r'$\alpha_s$', fontsize=25)
	plt.savefig(filename)

	# filename = '../graphs/mg1/singleUserEfficiencyServers'  + '.png'
	# fig = plt.figure()
	# legends = []
	# for val_lambda in lambdas:
	# 	key = r'$\lambda$=' + str(val_lambda)
	# 	legends.append(key)
	# plt.subplot(2, 1, 2)
	# plt.plot(price_ratios[::100], num_servers[0][::100], 'c*', markersize=7)
	# plt.plot(price_ratios[::100], num_servers[1][::100], 'ro', markersize=7)
	# plt.plot(price_ratios[::100], num_servers[2][::100], 'g^', markersize=7)
	# # plt.plot(lambdas[::200], results[3][::200], 'bs', markersize=7)
	# plt.plot(price_ratios, num_servers[0], 'c', linewidth='2')
	# plt.plot(price_ratios, num_servers[1], 'r', linewidth='2')
	# plt.plot(price_ratios, num_servers[2], 'g', linewidth='2')
	# # plt.plot(lambdas, results[3], 'b', linewidth='2')

	# plt.legend(legends, loc='upper right', fontsize=21)
	# plt.ylabel('Servers ', fontsize=25)
	# # plt.ylabel('Revenue from User', fontsize=25)
	# plt.xlabel(r'$\alpha_s$', fontsize=25)

	# plt.subplot(2,1,1)
	# legends = []
	# for val_lambda in lambdas:
	# 	key = r'$\lambda$=' + str(val_lambda)
	# 	legends.append(key)
	# plt.plot(price_ratios[::100], results[0][::100], 'c*', markersize=7)
	# plt.plot(price_ratios[::100], results[1][::100], 'ro', markersize=7)
	# plt.plot(price_ratios[::100], results[2][::100], 'g^', markersize=7)
	# # plt.plot(lambdas[::200], results[3][::200], 'bs', markersize=7)
	# plt.plot(price_ratios, results[0], 'c', linewidth='2')
	# plt.plot(price_ratios, results[1], 'r', linewidth='2')
	# plt.plot(price_ratios, results[2], 'g', linewidth='2')
	# # plt.plot(lambdas, results[3], 'b', linewidth='2')

	# plt.legend(legends, loc='upper right', fontsize=21)
	# plt.ylabel('Cloud Profit', fontsize=25)
	# # plt.ylabel('Revenue from User', fontsize=25)
	# # plt.xlabel(r'$\alpha_s$', fontsize=25)


	# plt.savefig(filename)

def computeZipfConstant(max_lambda, zipf_alpha):
	const = 0
	for val_lambda in range(1, max_lambda+1):
		const += math.pow((1.0/val_lambda), zipf_alpha)
	# print const 
	return const


def plotSingleUserDistVaryMu(mode, dist):
	max_lambda = 10000
	lambdas = [x for x in range(1, max_lambda)]
	alpha_v = 1.0
	price_ratios = [.1*x for x in range(10,300)]
	cp_cost_ratios = [0.5, 1, 1.5]
	mu_server = 30.0
	mu_v = 5.0
	# mu_s = 10.0
	mu_s_ratios = [0.01*x for x in range(100,600)]
	eff_s = 5.0
	eff_v = 10.0
	cp_alpha_v = 0.2
	# cp_alpha_s = 0.3
	beta = 0.9
	gamma = 1
	zipf_alpha = 100
	zipf_contant = computeZipfConstant(max_lambda, zipf_alpha)
	total_num_vm = 0
	total_num_serv = 0
	results = []
	total_profit = []
	profits = []
	optimal_a_s = []
	configs = []
	for cp_ratio in cp_cost_ratios:
		print cp_ratio
		cp_alpha_s = cp_alpha_v * cp_ratio
		total_profit = [] 
		optimal_a_s = []
		optimal_config = []
		for mu_ratio in mu_s_ratios:
			mu_s = mu_ratio * mu_v
			max_profit = -10000000000000000
			optimal_alpha_s = -1
			opt_config = -1
			for ratio in price_ratios:
				alpha_s = alpha_v * ratio
				results_num_vm = []
				results_num_serv = []
				results_total_servers = []
				results_cost = []
				profit = 0
				vms = 0
				serv = 0
				for val_lambda in lambdas:
					total_num_vm = 0
					total_num_serv = 0
					num_vm, load_serv, revenue = 0,0,0
					if mode == 'OO':		
						num_vm, load_serv, revenue = get_config(beta, mu_s, mu_v, alpha_v, alpha_s, gamma, val_lambda)
					elif mode == 'AO':
						num_vm, load_serv, revenue = get_configAO(beta, mu_s, mu_v, alpha_v, alpha_s, gamma, val_lambda)
					# num_vm, num_serv, user_cost = get_config(beta, mu_s, mu_v, alpha_v, alpha_s, gamma, val_lambda)
					cp_cost = getCPCost(num_vm, load_serv, cp_alpha_v, cp_alpha_s, mu_s, mu_v, val_lambda, beta)
					# print i, num_vm, num_serv
					vms += num_vm
					serv += load_serv
					results_num_vm.append(total_num_vm)
					results_num_serv.append(total_num_serv)
					results_total_servers.append(total_num_vm + total_num_serv)		
					results_cost.append(revenue - cp_cost)
					if dist == 'Uniform':
						profit += ((1.0/max_lambda)*(revenue - cp_cost))
					elif dist == 'Zipf':
						if math.pow((1.0/val_lambda),zipf_alpha) < math.pow(10, -10):
							break
						profit += ((math.pow((1.0/val_lambda),zipf_alpha)*(revenue - cp_cost))/zipf_contant)
				if profit > max_profit:
					max_profit = profit
					optimal_alpha_s = alpha_s
					if vms == 0:
						opt_config = 1
					elif serv == 0:
						opt_config = 3
					else:
						opt_config = 2
			total_profit.append(max_profit)
			optimal_a_s.append(optimal_alpha_s)
			optimal_config.append(opt_config)
		results.append(optimal_a_s)
		configs.append(optimal_config)
	filename = '../graphs/mg1/' + dist + 'Dist' + mode + 'VaryMu.png'
	fig = plt.figure()
	legends = []
	for ratio in cp_cost_ratios:
		key = r'$\alpha_{s\_cp}$=' + str(ratio) + r'$\alpha_{v\_cp}$'
		legends.append(key)
	plt.subplot(2,1,1)
	plt.plot(mu_s_ratios[::10], results[0][::10], 'c*', markersize=7)
	plt.plot(mu_s_ratios[::10], results[1][::10], 'ro', markersize=7)
	plt.plot(mu_s_ratios[::10], results[2][::10], 'g^', markersize=7)
	# plt.plot(price_ratios[::10], profits[3][::10], 'bs', markersize=7)
	plt.plot(mu_s_ratios, results[0], 'c', linewidth='2')
	plt.plot(mu_s_ratios, results[1], 'r', linewidth='2')
	plt.plot(mu_s_ratios, results[2], 'g', linewidth='2')
	# plt.plot(price_ratios, profits[3], 'b', linewidth='2')

	# plt.legend(legends, loc='upper right', fontsize=21)
	labelstr = 'Optimal ' + r'$\alpha_s$'
	plt.ylabel(labelstr, fontsize=25)
	

	plt.subplot(2, 1, 2)
	plt.plot(mu_s_ratios[::10], configs[0][::10], 'c*', markersize=7)
	plt.plot(mu_s_ratios[::10], configs[1][::10], 'ro', markersize=7)
	plt.plot(mu_s_ratios[::10], configs[2][::10], 'g^', markersize=7)
	plt.plot(mu_s_ratios, configs[0], 'c', linewidth='2')
	plt.plot(mu_s_ratios, configs[1], 'r', linewidth='2')
	plt.plot(mu_s_ratios, configs[2], 'g', linewidth='2')
	plt.yticks([0,1,2,3,4], ('', 'SC', 'SC + VM', 'VM', ''))
	plt.legend(legends, loc='lower right', fontsize=21)
	plt.xlabel(r'$\mu_s/\mu_v$', fontsize=20)
	plt.savefig(filename)


def plotSingleUserDist(mode, dist):
	max_lambda = 10000
	lambdas = [x for x in range(1, max_lambda)]
	alpha_v = 1.0
	price_ratios = [.01*x for x in range(100,1000)]
	cp_cost_ratios = [0.5]
	mu_server = 30.0
	mu_v = 5.0
	mu_s = 10.0
	eff_s = 5.0
	eff_v = 10.0
	cp_alpha_v = 0.2
	# cp_alpha_s = 0.3
	beta = 0.9
	gamma = 1
	zipf_alpha = 100
	zipf_contant = computeZipfConstant(max_lambda, zipf_alpha)
	total_num_vm = 0.0
	total_num_serv = 0.0
	results = []
	total_profit = []
	profits = []
	configs = []
	vms = []
	containers = []
	percent_load_containers = []
	total_load = 0
	for cp_ratio in cp_cost_ratios:
		print cp_ratio
		cp_alpha_s = cp_alpha_v * cp_ratio
		total_profit = [] 
		opt_config = []
		opt_vms = []
		opt_containers = []
		total_percent_SC = []
		for ratio in price_ratios:
			alpha_s = alpha_v * ratio
			results_num_vm = []
			results_num_serv = []
			results_total_servers = []
			results_cost = []
			profit = 0
			total_num_vm = 0.0
			total_num_serv = 0.0
			config = -1
			total_load = 0.0
			for val_lambda in lambdas:
				num_vm, load_serv, revenue = 0,0,0
				if mode == 'OO':		
					num_vm, load_serv, revenue = get_config(beta, mu_s, mu_v, alpha_v, alpha_s, gamma, val_lambda)
				elif mode == 'AO':
					num_vm, load_serv, revenue = get_configAO(beta, mu_s, mu_v, alpha_v, alpha_s, gamma, val_lambda)
				# num_vm, num_serv, user_cost = get_config(beta, mu_s, mu_v, alpha_v, alpha_s, gamma, val_lambda)
				cp_cost = getCPCost(num_vm, load_serv, cp_alpha_v, cp_alpha_s, mu_s, mu_v, val_lambda, beta)
				# print i, num_vm, num_serv
				total_num_serv += load_serv
				total_num_vm += num_vm
				results_cost.append(revenue - cp_cost)
				if dist == 'Uniform':
					profit += ((1.0/max_lambda)*(revenue - cp_cost))
				elif dist == 'Zipf':
					profit += ((math.pow((1.0/val_lambda),zipf_alpha)*(revenue - cp_cost))/zipf_contant)
			results.append(results_cost)
			total_profit.append(profit)
			if total_num_vm == 0:
				config = 1
			elif total_num_serv == 0:
				config = 3
			else:
				config = 2
			opt_config.append(config)
			opt_vms.append(total_num_vm)
			opt_containers.append(math.ceil(total_num_serv/mu_s))
			total_percent_SC.append(100.0*float(total_num_serv)/float(sum(range(max_lambda+1))))
			# total_percent_SC.append(total_num_serv)
		profits.append(total_profit)
		configs.append(opt_config)
		vms.append(opt_vms)
		containers.append(opt_containers)
		percent_load_containers.append(total_percent_SC)
	filename = '../graphs/mg1/' + dist + 'Dist' + mode + '.png'
	fig = plt.figure()
	legends = []
	for ratio in cp_cost_ratios:
		key = r'$\alpha_{s\_cp}$=' + str(ratio) + r'$\alpha_{v\_cp}$'
		legends.append(key)
	plt.subplot(2,1,1)
	plt.plot(price_ratios[::100], profits[0][::100], 'c*', markersize=7)
	# plt.plot(price_ratios[::10], profits[1][::10], 'ro', markersize=7)
	# plt.plot(price_ratios[::10], profits[2][::10], 'g^', markersize=7)
	# plt.plot(price_ratios[::10], profits[3][::10], 'bs', markersize=7)
	plt.plot(price_ratios, profits[0], 'c', linewidth='2')
	# plt.plot(price_ratios, profits[1], 'r', linewidth='2')
	# plt.plot(price_ratios, profits[2], 'g', linewidth='2')
	# plt.plot(price_ratios, profits[3], 'b', linewidth='2')

	plt.legend(legends, loc='lower right', fontsize=21)
	plt.ylabel('Cloud Provider Profit', fontsize=15)

	# plt.subplot(3, 1, 2)
	# plt.plot(price_ratios[::100], configs[0][::100], 'c*', markersize=7)

	# plt.plot(price_ratios, configs[0], 'c', linewidth='2')
	# plt.yticks([0,1,2,3,4], ('', 'SC', 'SC + VM', 'VM', ''))

	# plt.subplot(2,1,2)
	# plt.plot(price_ratios[::100], vms[0][::100], 'ro', markersize=7)
	# # plt.plot(price_ratios[::100], containers[0][::100], 'g^', markersize=7)
	# plt.plot(price_ratios, vms[0], 'r', linewidth='2')
	# # plt.plot(price_ratios, containers[0], 'g', linewidth='2')	
	# plt.ylabel('Number of resources', fontsize=15)
	# plt.legend(['VM', 'SC'])
	# # plt.yscale('log')
	# plt.xlabel(r'$\alpha_s$', fontsize=25)

	plt.subplot(2,1,2)
	plt.plot(price_ratios[::100], percent_load_containers[0][::100], 'c*', markersize=7)
	plt.plot(price_ratios, percent_load_containers[0], 'c', linewidth='2')
	plt.ylabel('% SC load', fontsize=15)
	plt.yscale('log')
	plt.xlabel(r'$\alpha_s$', fontsize=25)

	plt.savefig(filename)

def plotSingleUser():
	max_lambda = 40
	lambdas = [.01*x for x in range(0,((max_lambda+1)*100))]
	alpha_v = 1.0
	price_ratios = [.1*x for x in range(10,100)]
	mu_server = 30.0
	mu_v = 5.0
	mu_s = 10.0
	eff_s = 5.0
	eff_v = 10.0
	cp_alpha_v = 0.2
	cp_alpha_s = 0.3
	beta = 0.9
	gamma = 1
	total_num_vm = 0
	total_num_serv = 0
	results = []
	for ratio in price_ratios:
		alpha_s = alpha_v * ratio
		results_num_vm = []
		results_num_serv = []
		results_total_servers = []
		results_cost = []
		profit = 0
		for val_lambda in lambdas:
			total_num_vm = 0
			total_num_serv = 0
			num_vm, num_serv, user_cost = get_config(beta, mu_s, mu_v, alpha_v, alpha_s, gamma, val_lambda)
			cp_cost = getCPCost(num_vm, num_serv, cp_alpha_v, cp_alpha_s, mu_s, mu_v, val_lambda, beta)
			# print i, num_vm, num_serv
			results_num_vm.append(total_num_vm)
			results_num_serv.append(total_num_serv)
			results_total_servers.append(total_num_vm + total_num_serv)		
			results_cost.append(user_cost - cp_cost)
			profit += user_cost - cp_cost
		results.append(results_cost)
	filename = '../graphs/mg1/singleUserDelays'  + '.png'
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
	plt.ylabel('Cloud Provider Profit', fontsize=25)
	plt.xlabel(r'$\lambda$', fontsize=25)
	plt.savefig(filename)

def plotMultUsers():
	lambdas = [10,100]
	delays = [0.2,0.8]
	alpha_v = 1.0
	# price_ratios = [0.01*x for x in range(100,1000)]
	price_ratios = [x for x in range(1,10)]
	mu_v = 10.0
	mu_s = 12.0
	beta = 0.9
	gamma = 1.0
	total_num_vm = 0
	total_num_serv = 0
	results_num_vm = []
	results_num_serv = []
	results_total_servers = []
	for ratio in price_ratios:
		alpha_s = alpha_v * ratio
		total_num_vm = 0
		total_num_serv = 0
		for i in range(len(lambdas)):
			val_lambda = lambdas[i]
			new_mu_v, new_mu_s = mu_v/delays[i], mu_s/delays[i]
			num_vm, num_serv, cost = get_config(beta, new_mu_s, new_mu_v, alpha_v, alpha_s, gamma, val_lambda)
			print i, num_vm, num_serv
			total_num_vm += num_vm
			total_num_serv += num_serv
		results_num_vm.append(total_num_vm)
		results_num_serv.append(total_num_serv)
		results_total_servers.append(total_num_vm + total_num_serv)
	filename = '../graphs/mg1/delaysTwoUsers'  + '.png'
	fig = plt.figure()
	legends = ['OD', 'SC']
	# plt.plot(price_ratios[::200], results_num_vm[0][::200], 'c*', markersize=7)
	# plt.plot(price_ratios[::200], results_num_serv[1][::200], 'ro', markersize=7)
	# plt.plot(price_ratios[::200], results_total_servers[::200], 'g^', markersize=7)
	plt.plot(price_ratios, results_num_vm, 'c', linewidth='2')
	plt.plot(price_ratios, results_num_serv, 'r', linewidth='2')
	# plt.plot(price_ratios, results_total_servers, 'g', linewidth='2')

	plt.legend(legends, loc='upper right', fontsize=21)
	plt.ylabel('Number of servers', fontsize=25)
	# title = "Multiple VMs, different price ratios"
	# fig.suptitle(title)
	plt.xlabel(r'$\alpha_s/alpha_v$', fontsize=25)
	plt.savefig(filename)

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

def plotmultipleVMonoff_vary_price_ratio():
	max_lambda = 24
	lambdas = [.01*x for x in range(0,((max_lambda+1)*100))]
	alpha_v = 1.0
	price_ratios = [1,1.5,3,10]
	mu_v = 10.0
	mu_s = 12.0
	beta = 0.9
	gamma = 1.0

	results = []
	for ratio in price_ratios:
		print "price ratio = %f" % ratio
		results_price = []
		alpha_s = ratio * alpha_v
		Lv = beta * mu_v
		Ls = ((alpha_v*mu_s)/(alpha_s*mu_v))*(mu_v + gamma) - gamma
		print "Ls = %f\tLv = %f" % (Ls, Lv)
		for val_lambda in lambdas:
 			cost = 0
			if Lv <= Ls or Ls < 0: 														# Serverless Always
				cost = alpha_s*(val_lambda/mu_s)
			else:																		# Serverless until VM on-off
				k = math.floor(val_lambda/Lv)
				extra_lambda = val_lambda-(math.floor(val_lambda/Lv)*Lv)
				if extra_lambda < Ls:	
					# print "VM(%d On-Off) + SC" % k
					Cv = alpha_v*((Lv*(mu_v + gamma))/((mu_v*(Lv + gamma))))*k
					cost = Cv + (alpha_s*extra_lambda/mu_s)
				else:
					# print "VM(%d On-Off)" % (k+1)
					Cv = alpha_v*((Lv*(mu_v + gamma))/((mu_v*(Lv + gamma))))*k
					if extra_lambda == 0 and k == 1:
						print Cv
					cost = Cv + (alpha_v*((extra_lambda*(mu_v + gamma))/((mu_v*(extra_lambda + gamma)))))
			results_price.append(cost)
		results.append(results_price)
	filename = '../graphs/mg1/multVMsonoff_vary_price_ratios'  + '.png'
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

def plotmultipleVMonoff_vary_mu():
	max_lambda = 24
	lambdas = [.01*x for x in range(0,((max_lambda+1)*100))]
	alpha_v_multiple = 1.0
	alpha_s = 3*alpha_v_multiple
	mu_ratios = [1,2,3]
	mu_v_multiple = 7.0
	mu_s = 7
	beta = 0.9
	gamma = 1
	results = []
	for ratio in mu_ratios:
		print "mu ratio = %d" % ratio
		results_price = []
		mu_v = ratio * mu_v_multiple
		alpha_v = ratio * alpha_v_multiple
		Lv = beta * mu_v
		Ls = ((alpha_v*mu_s)/(alpha_s*mu_v))*(mu_v + gamma) - gamma
		print "Ls = %f\tLv = %f" % (Ls, Lv)
		for val_lambda in lambdas:
 			cost = 0
			if Lv <= Ls or Ls < 0: 														# Serverless Always
				cost = alpha_s*(val_lambda/mu_s)
			else:																		# Serverless until VM on-off
				k = math.floor(val_lambda/Lv)
				extra_lambda = val_lambda-(math.floor(val_lambda/Lv)*Lv)
				if extra_lambda < Ls:	
					# print "VM(%d On-Off) + SC" % k
					Cv = alpha_v*((Lv*(mu_v + gamma))/((mu_v*(Lv + gamma))))*k
					cost = Cv + (alpha_s*extra_lambda/mu_s)
				else:
					# print "VM(%d On-Off)" % (k+1)
					Cv = alpha_v*((Lv*(mu_v + gamma))/((mu_v*(Lv + gamma))))*k
					if extra_lambda == 0 and k == 1:
						print Cv
					cost = Cv + (alpha_v*((extra_lambda*(mu_v + gamma))/((mu_v*(extra_lambda + gamma)))))
			results_price.append(cost)
		results.append(results_price)
	filename = '../graphs/mg1/multVMsonoff_vary_mu'  + '.png'
	fig = plt.figure()
	legends = []
	for ratio in mu_ratios:
		key = r'$\mu_v$=' + str(ratio) + r'$\mu_s$' + ', ' + r'$\alpha_v$=' + str(ratio)
		legends.append(key)
	# plt.plot(lambdas, results[0], 'r', marker='o', linestyle='-')
	# plt.plot(lambdas, results[1], 'g', marker='s', linestyle='-')
	# plt.plot(lambdas, results[2], 'b', marker='^', linestyle='-')
	plt.plot(lambdas[::200], results[0][::200], 'ro', markersize=7)
	plt.plot(lambdas[::200], results[1][::200], 'g^', markersize=7)
	plt.plot(lambdas[::200], results[2][::200], 'bs', markersize=7)
	plt.plot(lambdas, results[0], 'r', linewidth='2')
	plt.plot(lambdas, results[1], 'g', linewidth='2')
	plt.plot(lambdas, results[2], 'b', linewidth='2')
	
	# plt.plot(lambdas, results[1], 'g', linewidth='3')
	# plt.plot(lambdas, results[2], 'b', linewidth='3')

	# plt.plot(lambdas, results[3], 'cD')
	plt.legend(legends, loc='upper left', fontsize=22)
	plt.ylabel('Cost', fontsize=25)
	# title = "Multiple VMs, different price ratios"
	# fig.suptitle(title)
	plt.xlabel(r'$\lambda$', fontsize=25)
	plt.savefig(filename)

def plotmultipleVMonoff_vary_gamma():
	max_lambda = 13
	lambdas = [.01*x for x in range(0,((max_lambda+1)*100))]
	alpha_v = 1.0
	alpha_s = 4*alpha_v
	gamma_ratios = [0.25,0.5,1]
	mu_v = 7.0
	mu_s = 10.0
	beta = 0.9
	gamma_multiple = 1

	results = []
	for ratio in gamma_ratios:
		print "gamma = %f" % ratio
		results_price = []
		gamma = ratio * gamma_multiple
		Lv = beta * mu_v
		Ls = ((alpha_v*mu_s)/(alpha_s*mu_v))*(mu_v + gamma) - gamma
		print "Ls = %f\tLv = %f" % (Ls, Lv)
		for val_lambda in lambdas:
 			cost = 0
			if Lv <= Ls or Ls < 0: 														# Serverless Always
				cost = alpha_s*(val_lambda/mu_s)
			else:																		# Serverless until VM on-off
				k = math.floor(val_lambda/Lv)
				extra_lambda = val_lambda-(math.floor(val_lambda/Lv)*Lv)
				if extra_lambda < Ls:	
					# print "VM(%d On-Off) + SC" % k
					Cv = alpha_v*((Lv*(mu_v + gamma))/((mu_v*(Lv + gamma))))*k
					cost = Cv + (alpha_s*extra_lambda/mu_s)
				else:
					# print "VM(%d On-Off)" % (k+1)
					Cv = alpha_v*((Lv*(mu_v + gamma))/((mu_v*(Lv + gamma))))*k
					if extra_lambda == 0 and k == 1:
						print Cv
					cost = Cv + (alpha_v*((extra_lambda*(mu_v + gamma))/((mu_v*(extra_lambda + gamma)))))
			results_price.append(cost)
		results.append(results_price)
	filename = '../graphs/mg1/multVMsonoff_vary_gamma'  + '.png'
	fig = plt.figure()
	legends = []
	for ratio in gamma_ratios:
		# key = "hello" + str(ratio) 
		key = '1/' + r'$\gamma$' + ' = ' + str(int(1/ratio))
		legends.append(key)
	plt.plot(lambdas[::100], results[0][::100], 'ro', markersize=7)
	plt.plot(lambdas[::100], results[1][::100], 'g^', markersize=7)
	plt.plot(lambdas[::100], results[2][::100], 'bs', markersize=7)
	plt.plot(lambdas, results[0], 'r', linewidth='2')
	plt.plot(lambdas, results[1], 'g', linewidth='2')
	plt.plot(lambdas, results[2], 'b', linewidth='2')
	# plt.plot(lambdas, results[3], 'cD')
	plt.legend(legends, loc='upper left', fontsize=22)
	plt.ylabel('Cost', fontsize=25)
	# title = "Multiple VMs, different price ratios"
	# fig.suptitle(title)
	plt.xlabel(r'$\lambda$', fontsize=25)
	plt.savefig(filename)

def plotmus_to_muv_ON_OFF():
	max_lambda = 3
	lambdas = [x for x in range(1, max_lambda+1)]
	gamma = 1.0
	beta = 0.9
	alpha_s = 3.0
	min_alpha_v = 1.0
	results = []
	mu_v_values_final = []
	for val_lambda in lambdas:
		min_mu_v = val_lambda/beta
		mu_v_values = []
		results_mu_v = []
		for i in range(0,600):
			mu_v = min_mu_v + (0.01*i)
			mu_v_values.append(mu_v)
			alpha_v = (min_alpha_v/min_mu_v) * mu_v
			mu_s = (alpha_s/alpha_v) * mu_v * ((val_lambda + gamma) /(mu_v + gamma) )
			results_mu_v.append(mu_s)
		results.append(results_mu_v)
		mu_v_values_final.append(mu_v_values)
	filename = '../graphs/mg1/mus_vs_muv_ON_OFF'  + '.png'
	fig = plt.figure()
	plt.plot(mu_v_values_final[0][::50], results[0][::50], 'ro', markersize=7)
	plt.plot(mu_v_values_final[1][::50], results[1][::50], 'g^', markersize=7)
	plt.plot(mu_v_values_final[2][::50], results[2][::50], 'bs', markersize=7)
	plt.plot(mu_v_values_final[0], results[0], 'r', linewidth='2')
	plt.plot(mu_v_values_final[1], results[1], 'g', linewidth='2')
	plt.plot(mu_v_values_final[2], results[2], 'b', linewidth='2')
	legends = []
	for val_lambda in lambdas: 
		key = r'$\lambda$' + ' = ' + str(val_lambda)
		legends.append(key)
	plt.legend(legends, loc='upper left', fontsize=22)
	plt.ylabel(r'$\mu_s$', fontsize=25)
	plt.xlabel(r'$\mu_v$', fontsize=25)
	plt.savefig(filename)


def main():
	if len(sys.argv) >  4:
		print "USAGE: python generate_plots_mg1.py <exp_type>"
		print "<exp_type> : vary_num_VMs/vary_startup_delay/vary_service_rate_VM/plotVMcost/plotTotalcost"
		print "<exp_type> : multipleVMs_vary_price_ratio/multipleVMs_vary_mu/multipleVMonoff_vary_price_ratio/multipleVMonoff_vary_mu/multipleVMonoff_vary_gamma"
		print "<exp_type> : costserv_to_vm_ON/costserv_to_vm_ON_OFF"
		print "<exp_type> : mus_to_muv_ON_OFF"
		print "<exp_type> : plotMultUsers/plotSingleUser/plotSingleUserOptimalSC/plotSingleUserOptimalSCvaryDelay/plotSingleUserEfficiency/plotSingleUserDist/plotSingleUserEfficiencyVaryMu/plotSingleUserDistVaryMu/plotSingleUserEfficiencyVarySCcost"
		return
	exp_type = sys.argv[1]
	if exp_type == 'vary_startup_delay':
		vary_startup_delay()
	elif exp_type == 'vary_num_VMs':
		vary_num_VMs()
	elif exp_type == 'vary_service_rate_VM':
		vary_service_rate_VM()
	elif exp_type == 'plotVMcost':
		plotVMcost()
	elif exp_type == 'plotTotalcost':
		plotTotalcost()
	elif exp_type == 'singleVM':
		plotsingleVM()
	elif exp_type == 'multipleVMs_vary_price_ratio':
		plotmultipleVMs_vary_price_ratio()
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
	elif exp_type == 'plotMultUsers':
		plotMultUsers()
	elif exp_type == 'plotSingleUser':
		plotSingleUser()
	elif exp_type == 'plotSingleUserEfficiencyVaryMu':
		plotSingleUserEfficiencyVaryMu()
	elif exp_type == 'plotSingleUserDist':
		plotSingleUserDist(sys.argv[2], sys.argv[3])
	elif exp_type == 'plotSingleUserDistVaryMu':
		plotSingleUserDistVaryMu(sys.argv[2], sys.argv[3])
	elif exp_type == 'plotSingleUserEfficiency':
		plotSingleUserEfficiency(sys.argv[2])
	elif exp_type == 'plotSingleUserOptimalSC':
		plotSingleUserOptimalSC()
	elif exp_type == 'plotSingleUserOptimalSCvaryDelay':
		plotSingleUserOptimalSCvaryDelay()
	elif exp_type == 'plotSingleUserOptimalSCvaryCPcost':
		plotSingleUserOptimalSCvaryCPcost()
	elif exp_type == 'plotSingleUserEfficiencyVarySCcost':
		plotSingleUserEfficiencyVarySCcost()
	elif exp_type == 'get_optimal_alpha_s':
		get_optimal_alpha_s()
	elif exp_type == 'get_optimal_alpha_s_vary_beta_v':
		get_optimal_alpha_s_vary_beta_v()
	elif exp_type == 'zipfdistribution':
		zipfdistribution()
	else:
		print "Wrong <exp_type>"
		print "<exp_type> : vary_num_VMs/vary_startup_delay/vary_service_rate_VM/plotVMcost/plotTotalcost"
		print "<exp_type> : multipleVMs_vary_price_ratio/multipleVMs_vary_mu/multipleVMonoff_vary_price_ratio/multipleVMonoff_vary_mu/multipleVMonoff_vary_gamma"
		print "<exp_type> : costserv_to_vm_ON/costserv_to_vm_ON_OFF"
		print "<exp_type> : mus_to_muv_ON_OFF"
		print "<exp_type> : plotMultUsers/plotSingleUser/plotSingleUserOptimalSC/plotSingleUserOptimalSCvaryDelay/plotSingleUserEfficiency/plotSingleUserDist/plotSingleUserEfficiencyVaryMu/plotSingleUserDistVaryMu/plotSingleUserEfficiencyVarySCcost"

if __name__ == '__main__':
	main()