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
	Ls = (alpha_v/alpha_s)*mu_s
	Lv = beta*mu_v
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
	serv_cost = cp_alpha_s * math.ceil(load_serv / mu_s) 
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





def plotSingleUserEfficiency(mode):
	alpha_v = 1.0
	price_ratios = [0.01*x for x in range(100,600)]
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
	lambdas = [4, 20, 100]
	num_servers = []
	# lambdas = [20, 60, 100]
	# lambdas = [100]
	for val_lambda in lambdas:
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
				print "here"
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

	plt.legend(legends, loc='upper right', fontsize=21)
	plt.ylabel('Cloud Provider Profit', fontsize=25)
	# plt.ylabel('Revenue from User', fontsize=25)
	plt.xlabel(r'$\alpha_s$', fontsize=25)
	plt.savefig(filename)

	filename = '../graphs/mg1/singleUserEfficiencyServers'  + '.png'
	fig = plt.figure()
	legends = []
	for val_lambda in lambdas:
		key = r'$\lambda$=' + str(val_lambda)
		legends.append(key)
	plt.subplot(2, 1, 2)
	plt.plot(price_ratios[::100], num_servers[0][::100], 'c*', markersize=7)
	plt.plot(price_ratios[::100], num_servers[1][::100], 'ro', markersize=7)
	plt.plot(price_ratios[::100], num_servers[2][::100], 'g^', markersize=7)
	# plt.plot(lambdas[::200], results[3][::200], 'bs', markersize=7)
	plt.plot(price_ratios, num_servers[0], 'c', linewidth='2')
	plt.plot(price_ratios, num_servers[1], 'r', linewidth='2')
	plt.plot(price_ratios, num_servers[2], 'g', linewidth='2')
	# plt.plot(lambdas, results[3], 'b', linewidth='2')

	plt.legend(legends, loc='upper right', fontsize=21)
	plt.ylabel('Servers ', fontsize=25)
	# plt.ylabel('Revenue from User', fontsize=25)
	plt.xlabel(r'$\alpha_s$', fontsize=25)

	plt.subplot(2,1,1)
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

	plt.legend(legends, loc='upper right', fontsize=21)
	plt.ylabel('Cloud Profit', fontsize=25)
	# plt.ylabel('Revenue from User', fontsize=25)
	# plt.xlabel(r'$\alpha_s$', fontsize=25)


	plt.savefig(filename)

def plotSingleUserUniformDist(mode):
	max_lambda = 10000
	lambdas = [.01*x for x in range(0,((max_lambda+1)*100))]
	alpha_v = 1.0
	price_ratios = [.1*x for x in range(10,60)]
	cp_cost_ratios = [0.5, 1, 1.5]
	mu_server = 30.0
	mu_v = 5.0
	mu_s = 10.0
	eff_s = 5.0
	eff_v = 10.0
	cp_alpha_v = 0.2
	# cp_alpha_s = 0.3
	beta = 0.9
	gamma = 1
	total_num_vm = 0
	total_num_serv = 0
	results = []
	total_profit = []
	profits = []
	for cp_ratio in cp_cost_ratios:
		cp_alpha_s = cp_alpha_v * cp_ratio
		total_profit = [] 
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
				num_vm, load_serv, revenue = 0,0,0
				if mode == 'OO':		
					num_vm, load_serv, revenue = get_config(beta, mu_s, mu_v, alpha_v, alpha_s, gamma, val_lambda)
				elif mode == 'AO':
					num_vm, load_serv, revenue = get_configAO(beta, mu_s, mu_v, alpha_v, alpha_s, gamma, val_lambda)
				# num_vm, num_serv, user_cost = get_config(beta, mu_s, mu_v, alpha_v, alpha_s, gamma, val_lambda)
				cp_cost = getCPCost(num_vm, load_serv, cp_alpha_v, cp_alpha_s, mu_s, mu_v, val_lambda, beta)
				# print i, num_vm, num_serv
				results_num_vm.append(total_num_vm)
				results_num_serv.append(total_num_serv)
				results_total_servers.append(total_num_vm + total_num_serv)		
				results_cost.append(revenue - cp_cost)
				profit += revenue - cp_cost
			results.append(results_cost)
			total_profit.append(profit)
		profits.append(total_profit)
	filename = '../graphs/mg1/UniformDist' + mode + '.png'
	fig = plt.figure()
	legends = []
	for ratio in cp_cost_ratios:
		key = r'$\alpha_{s\_cp}$=' + str(ratio) + r'$\alpha_{v\_cp}$'
		legends.append(key)
	plt.plot(price_ratios[::10], profits[0][::10], 'c*', markersize=7)
	plt.plot(price_ratios[::10], profits[1][::10], 'ro', markersize=7)
	plt.plot(price_ratios[::10], profits[2][::10], 'g^', markersize=7)
	# plt.plot(price_ratios[::10], profits[3][::10], 'bs', markersize=7)
	plt.plot(price_ratios, profits[0], 'c', linewidth='2')
	plt.plot(price_ratios, profits[1], 'r', linewidth='2')
	plt.plot(price_ratios, profits[2], 'g', linewidth='2')
	# plt.plot(price_ratios, profits[3], 'b', linewidth='2')

	plt.legend(legends, loc='upper right', fontsize=21)
	plt.ylabel('Cloud Provider Profit', fontsize=25)
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
	if len(sys.argv) >  3:
		print "USAGE: python generate_plots_mg1.py <exp_type>"
		print "<exp_type> : vary_num_VMs/vary_startup_delay/vary_service_rate_VM/plotVMcost/plotTotalcost"
		print "<exp_type> : multipleVMs_vary_price_ratio/multipleVMs_vary_mu/multipleVMonoff_vary_price_ratio/multipleVMonoff_vary_mu/multipleVMonoff_vary_gamma"
		print "<exp_type> : costserv_to_vm_ON/costserv_to_vm_ON_OFF"
		print "<exp_type> : mus_to_muv_ON_OFF"
		print "<exp_type> : plotMultUsers/plotSingleUser/plotSingleUserOptimalSC/plotSingleUserOptimalSCvaryDelay/plotSingleUserEfficiency/plotSingleUserUniformDist"
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
	elif exp_type == 'plotSingleUserUniformDist':
		plotSingleUserUniformDist(sys.argv[2])
	elif exp_type == 'plotSingleUserEfficiency':
		plotSingleUserEfficiency(sys.argv[2])
	elif exp_type == 'plotSingleUserOptimalSC':
		plotSingleUserOptimalSC()
	elif exp_type == 'plotSingleUserOptimalSCvaryDelay':
		plotSingleUserOptimalSCvaryDelay()
	elif exp_type == 'plotSingleUserOptimalSCvaryCPcost':
		plotSingleUserOptimalSCvaryCPcost()
	else:
		print "Wrong <exp_type>"
		print "<exp_type> : vary_num_VMs/vary_startup_delay/vary_service_rate_VM/plotVMcost/plotTotalcost"
		print "<exp_type> : multipleVMs_vary_price_ratio/multipleVMs_vary_mu/multipleVMonoff_vary_price_ratio/multipleVMonoff_vary_mu/multipleVMonoff_vary_gamma"
		print "<exp_type> : costserv_to_vm_ON/costserv_to_vm_ON_OFF"
		print "<exp_type> : mus_to_muv_ON_OFF"
		print "<exp_type> : plotMultUsers/plotSingleUser/plotSingleUserOptimalSC/plotSingleUserOptimalSCvaryDelay/plotSingleUserEfficiency/plotSingleUserUniformDist"

if __name__ == '__main__':
	main()