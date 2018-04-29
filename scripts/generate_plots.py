"""
Pricing Analysis: create plots for different serverless vs VMs scenario

Usage: python generate_plots.py <exp_type>
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
# # Parameters
# PRICE_CONSTANT = 5
# beta = 0.8
# lambdas = [10*x for x in range(1,11)]
# alpha_m = 1
# alpha_s = PRICE_CONSTANT * alpha_m
# mu_s = 5
# mu_m = 5
# n = 5  # NUM VMs
# gamma = 5

def solve_quadratic_eq(a,b,c):
	d = (b**2)-(4*a*c) # discriminant
	string = "a: " + str(a) + "\tb: " + str(b) + "\tc: " + str(c) + "\td: " + str(d)
	print string
	if d < 0:
		print "This equation has no real solution"
		return [] 
	elif d == 0:
		x = (-b+math.sqrt(d))/(2*a)
		print "This equation has one solutions: ", x
		if x >= 0:
			return [x]
		else:
			return []
	else:
		x1 = (-b+math.sqrt(d))/(2*a)
		x2 = (-b-math.sqrt(d))/(2*a)
		print "This equation has two solutions: ", x1, " and", x2
		# if min(x1, x2) < 0:
		# 	return [max(x1,x2)]
		# else:
		# 	return [min(x1,x2)]
		return [min(x1,x2)]

def solve_quadratic_eq2(a,b,c):
	d = (b**2)-(4*a*c) # discriminant
	# string = "a: " + str(a) + "\tb: " + str(b) + "\tc: " + str(c) + "\td: " + str(d)
	# print string
	if d < 0:
		print "This equation has no real solution"
		return [] 
	elif d == 0:
		x = (-b+math.sqrt(d))/(2*a)
		print "This equation has one solutions: ", x
		if x >= 0:
			return [x]
		else:
			return []
	else:
		x1 = (-b+math.sqrt(d))/(2.0*a)
		x2 = (-b-math.sqrt(d))/(2.0*a)
		# print "This equation has two solutions: ", x1, " and", x2
		# print [min(x1,x2)]
		return [min(x1,x2)]

# mode : upper/lower
def calc_lambda_m(alpha_m, alpha_s, mu_m, mu_s, gamma, i, mode):
	# lambda_m = -((gamma * mu_m * alpha_s * n)/(2 * mu_s * alpha_m)) + ((gamma + mu_m)*n/2) 
	a_ub = (alpha_m * mu_s) - (alpha_s * mu_m)
	b_ub = (2 * alpha_s * mu_m * gamma * i) + (2 * alpha_s * (mu_m**2) * i) - (2 * alpha_m * mu_s * gamma * i) - (2 * alpha_m * mu_m * mu_s * i)
	c_ub = (alpha_m * mu_m * mu_s * gamma * (i**2)) + (alpha_m * mu_s * (gamma**2) * (i**2)) + (alpha_m * (mu_m**2) * mu_s * (i**2)) - (2 * alpha_s * (mu_m**2) * gamma * (i**2)) - (alpha_s * mu_m * (gamma**2) * (i**2)) - (alpha_s * (mu_m**3) * (i**2))
	a_lb = a_ub
	b_lb = (2 * alpha_s * (mu_m**2) * i) + (4 * alpha_s * mu_m * gamma * i) - (2 * alpha_m * mu_m * mu_s * i) - (4 * alpha_m * mu_s * gamma * i)
	c_lb = (4 * alpha_m * mu_s * (gamma**2) * (i**2)) + (alpha_m * (mu_m**2) * mu_s * (i**2)) - (4 * alpha_s * mu_m * (gamma**2) * (i**2)) - (alpha_s * (mu_m**3) * (i**2)) - (4 * alpha_s * (mu_m**2) * gamma * (i**2)) + (2 * alpha_m * mu_m * mu_s * gamma * (i**2))
	if mode == 'upper':
		roots = solve_quadratic_eq(a_ub, b_ub, c_ub)
		if len(roots) != 0:
			return roots[0]
	elif mode == 'lower':
		roots = solve_quadratic_eq(a_lb, b_lb, c_lb)
		if len(roots) != 0:
			return roots[0]
	else:
		print 'wrong mode'
		return 'wrong mode'
	# print 'error'
	return 'error'

def create_plot(filename, title, legends, xaxis, yaxis):
	fig = plt.figure()
	for j in range(len(yaxis)):
		plt.plot(xaxis, yaxis[j], marker=markers[j])
	if len(legends) != 0:
		plt.legend(legends, loc='upper left')
	fig.suptitle(title)
	plt.xlabel(r'$\lambda$')
	plt.ylabel(r'$\lambda_m$')
	plt.savefig(filename)

def create_plot_lambdas(filename, title, legends, xaxis, yaxis, lambda_s):
	fig = plt.figure()
	print yaxis
	for j in range(len(yaxis)):
		plt.plot(xaxis, yaxis[j], marker=markers[j])
		plt.plot(xaxis, lambda_s[j], marker=markers[j+1])
	plt.legend(legends, loc='upper right')
	fig.suptitle(title)
	plt.xlabel(r'$\lambda$')
	plt.ylabel(r'$\lambda_values$')
	plt.savefig(filename)

def vary_service_rate_VM():
	PRICE_CONSTANT = 5
	beta = 0.8
	lambdas = [10*x for x in range(1,9)]
	alpha_m = 1
	alpha_s = PRICE_CONSTANT * alpha_m
	mu_s = 5
	gamma = 0.5
	i = 5
	mu_m_multiples = [0.1, 0.5, 1, 1.5, 2]
	# UPPER
	results_ub = []
	legends_ub = []
	for m in range(len(mu_m_multiples)):
		multiple = mu_m_multiples[m]
		val_mu_m = multiple * mu_s
		results_lambda_m = []
		for val_lambda in lambdas:
			lambda_m = calc_lambda_m(alpha_m, alpha_s, val_mu_m, mu_s, gamma, i, 'upper')
			if lambda_m == 'error':
				lambda_m = 0
			elif lambda_m > val_lambda:
				lambda_m = val_lambda
			# if lambda_m > beta*val_lambda:
			# 	lambda_m = beta*val_lambda
			# elif lambda_m < 0:
			# 	lambda_m = 0
			results_lambda_m.append(lambda_m)
		key_ub = 'UB$\mu_m=$' + str(multiple) + '$\mu_s$'
		legends_ub.append(key_ub)
		results_ub.append(results_lambda_m)
	# LOWER
	results_lb = []
	legends_lb = []
	for m in range(len(mu_m_multiples)):
		multiple = mu_m_multiples[m]
		val_mu_m = multiple * mu_s
		results_lambda_m = []
		for val_lambda in lambdas:
			lambda_m = calc_lambda_m(alpha_m, alpha_s, val_mu_m, mu_s, gamma, i, 'lower')
			if lambda_m == 'error':
				lambda_m = 0
			elif lambda_m > val_lambda:
				lambda_m = val_lambda
			# if lambda_m > beta*val_lambda:
			# 	lambda_m = beta*val_lambda
			# elif lambda_m < 0:
			# 	lambda_m = 0
			results_lambda_m.append(lambda_m)
		key_lb = 'LB$\mu_m=$' + str(multiple) + '$\mu_s$'
		legends_lb.append(key_lb)
		results_lb.append(results_lambda_m)
	# plot
	variables = ",gamma=" + str(gamma) + ",alpha_m=" + str(PRICE_CONSTANT) + "*alpha_s,i=" + str(i)
	variables_filename = "__i=" + str(i) + "__alpha_m=" + str(PRICE_CONSTANT) + "*alpha_s__i=" + str(i)
	title = "Varying service rate of VM" + variables
	filename = '../graphs/vary_service_rate_VM' + variables_filename + '.png'
	results = results_lb
	# results.extend(results_ub)
	legends = legends_lb
	# legends.extend(legends_ub)
	create_plot(filename, title, legends, lambdas, results)

def vary_num_VMs():
	PRICE_CONSTANT = 5
	beta = 0.8
	lambdas = [10*x for x in range(1,9)]
	alpha_m = 1
	alpha_s = PRICE_CONSTANT * alpha_m
	mu_s = 5
	mu_m = 5
	i = [5*i for i in range(1,6)]
	i.insert(0,1)
	gamma = 0.1
	# UPPER
	results_ub = []
	legends_ub = []
	# results_lambda_s = []
	for val_i in i:
		results_lambda_m = []
		# result_s = []
		for val_lambda in lambdas:
			lambda_m = calc_lambda_m(alpha_m, alpha_s, mu_m, mu_s, gamma, val_i, 'upper')
			if lambda_m == 'error':
				lambda_m = 0
			elif lambda_m > val_lambda:
				lambda_m = val_lambda
			results_lambda_m.append(lambda_m)
			# result_s.append(val_lambda - lambda_m)
		key_ub = 'UB: i=' + str(val_i)
		legends_ub.append(key_ub)
		results_ub.append(results_lambda_m)
		# results_lambda_s.append(result_s)
	# LOWER
	results_lb = []
	legends_lb = []
	# results_lambda_s = []
	for val_i in i:
		results_lambda_m = []
		# result_s = []
		for val_lambda in lambdas:
			lambda_m = calc_lambda_m(alpha_m, alpha_s, mu_m, mu_s, gamma, val_i, 'lower')
			if lambda_m == 'error':
				lambda_m = 0
			elif lambda_m > val_lambda:
				lambda_m = val_lambda
			results_lambda_m.append(lambda_m)
			# result_s.append(val_lambda - lambda_m)
		key_lb = 'LB: i=' + str(val_i)
		legends_lb.append(key_lb)
		results_lb.append(results_lambda_m)
		# results_lambda_s.append(result_s)

	# plot
	variables = ",gamma=" + str(gamma) + ",alpha_m=" + str(PRICE_CONSTANT) + "*alpha_s,mu_m=" + str(mu_m) + ",mu_s=" + str(mu_s)
	variables_filename = "__gamma=" + str(gamma) + "__alpha_m=" + str(PRICE_CONSTANT) + "*alpha_s__mu_m=" + str(mu_m) + "__mu_s=" + str(mu_s) 
	title = "Varying number of VMs" + variables
	filename = '../graphs/vary_num_VMs' + variables_filename + '.png'
	results = results_lb
	# results.extend(results_ub)
	legends = legends_lb
	# legends.extend(legends_ub)
	create_plot(filename, title, legends, lambdas, results)
	# title = "Lambdas for serverless and VMs"
	# filename = '../graphs/lambdas_split.png'
	# legends = ['lambda_m', 'lambda_s']
	# create_plot_lambdas(filename, title, legends, lambdas, results, results_lambda_s)

def vary_startup_delay():
	PRICE_CONSTANT = 10
	beta = 0.8
	lambdas = [10*x for x in range(1,9)]
	alpha_m = 1
	alpha_s = PRICE_CONSTANT * alpha_m
	mu_s = 5
	mu_m = 5
	i = 5
	gamma_multiples = [0.1, 0.5, 1, 1.5, 2]
	# Upper bound:
	results_ub = []
	legends_ub = []
	for multiple in gamma_multiples:
		val_gamma = multiple * mu_m
		results_lambda_m = []
		for val_lambda in lambdas:
			lambda_m = calc_lambda_m(alpha_m, alpha_s, mu_m, mu_s, val_gamma, i, 'upper')
			if lambda_m == 'error':
				lambda_m = 0
			elif lambda_m > val_lambda:
				lambda_m = val_lambda
			# elif lambda_m > i*mu_m:
			# 	lambda_m = i*mu_m
			# if lambda_m > beta*val_lambda:s
			# 	lambda_m = beta*val_lambda
			# elif lambda_m < 0:
			# 	lambda_m = 0
			results_lambda_m.append(lambda_m)
		key_ub = 'UB: gamma=' + str(multiple) + '$\mu_m$'
		legends_ub.append(key_ub)
		results_ub.append(results_lambda_m)
	# Lower bound
	results_lb = []
	legends_lb = []
	for multiple in gamma_multiples:
		val_gamma = multiple * mu_m
		results_lambda_m = []
		for val_lambda in lambdas:
			lambda_m = calc_lambda_m(alpha_m, alpha_s, mu_m, mu_s, val_gamma, i, 'lower')
			if lambda_m == 'error':
				lambda_m = 0
			elif lambda_m > val_lambda:
				lambda_m = val_lambda
			# elif lambda_m > i*mu_m:
			# 	lambda_m = i*mu_m
			# if lambda_m > beta*val_lambda:
			# 	lambda_m = beta*val_lambda
			# elif lambda_m < 0:
			# 	lambda_m = 0
			results_lambda_m.append(lambda_m)
		key_lb = 'LB: gamma=' + str(multiple) + '$\mu_m$'
		legends_lb.append(key_lb)
		results_lb.append(results_lambda_m)

	# plot
	variables = ",i=" + str(i) + ",alpha_m=" + str(PRICE_CONSTANT) + "*alpha_s,mu_m=" + str(mu_m) + ",mu_s=" + str(mu_s)
	variables_filename = "__i=" + str(i) + "__alpha_m=" + str(PRICE_CONSTANT) + "*alpha_s__mu_m=" + str(mu_m) + "__mu_s=" + str(mu_s) 
	title = "Varying startup delay" + variables
	filename = '../graphs/vary_startup_delay' + variables_filename + '.png'
	results = results_lb
	# results.extend(results_ub)
	legends = legends_lb
	# legends.extend(legends_ub)
	create_plot(filename, title, legends, lambdas, results)

def plotTotalcost():
	val_lambda = 24
	lambdas_m = [x for x in range(1,val_lambda+1)]
	alpha_m = 1.0
	PRICE_CONSTANT = 5
	alpha_s = PRICE_CONSTANT * alpha_m
	gamma = 2.0
	i = 5.0
	mu_m = 5.0
	mu_s = 5.0
	results = []
	results_ub = []
	results_lb = []
	results_orig = []
	results_setup_orig = []
	legends_ub = []
	xaxis = []
	xaxis_ub = []
	xaxis_lb = []
	xaxis_orig = []
	xaxis_setup_orig = []
	count_ub, count_lb = 0,0

	for val_lambda_m in lambdas_m:
		val_ub = calcTotalcost(alpha_s, alpha_m, mu_s, mu_m, gamma, i, val_lambda, val_lambda_m,'upper')
		val_lb = calcTotalcost(alpha_s, alpha_m, mu_s, mu_m, gamma, i, val_lambda, val_lambda_m,'lower')
		val_orig = calcTotalcost(alpha_s, alpha_m, mu_s, mu_m, gamma, i, val_lambda, val_lambda_m,'original')
		val_setup_orig = calcTotalcost(alpha_s, alpha_m, mu_s, mu_m, gamma, i, val_lambda, val_lambda_m,'setup_original')
		# val_lb = (alpha_m/gamma)*(1/((1/mu_m) + (i/val_lambda) + (1/gamma)))
		# print val_lb
		if val_ub != 'div0':
			results_ub.append(val_ub)	
			xaxis_ub.append(val_lambda_m)
		if val_lb != 'div0':
			results_lb.append(val_lb)
			xaxis_lb.append(val_lambda_m)
		if val_orig != 'div0':
			results_orig.append(val_orig)
			xaxis_orig.append(val_lambda_m)
		if val_setup_orig != 'div0':
			results_setup_orig.append(val_setup_orig)
			xaxis_setup_orig.append(val_lambda_m)
	results.append(results_ub)
	results.append(results_lb)
	results.append(results_orig)
	results.append(results_setup_orig)
	xaxis.append(xaxis_ub)
	xaxis.append(xaxis_lb)
	xaxis.append(xaxis_orig)
	xaxis.append(xaxis_setup_orig)
	variables = ",new_LB,i=" + str(i) + ",alpha_m=" + str(alpha_m) + ",mu_m=" + str(mu_m) + ",gamma=" + str(gamma) + ",lambda=" + str(val_lambda)
	title = "Total cost" + variables
	filename = '../graphs/Totalcost' + variables + '.png'
	fig = plt.figure()
	plt.plot(xaxis[0], results[0], 'ro')
	plt.plot(xaxis[1], results[1], 'bo')
	plt.plot(xaxis[2], results[2], 'g*')
	plt.plot(xaxis[3], results[3], 'ms')
	# legends = ['upper bound', 'lower bound']
	legends = ['upper bound', 'lower bound', 'actual', 'setup cost actual']
	# legends = ['serverless', 'lower bound', 'actual', 'setup cost actual']
	plt.legend(legends, loc='upper right')

	plt.ylabel('Total Cost')
	fig.suptitle(title)
	plt.xlabel(r'$\lambda_m$')
	plt.savefig(filename)

# mode : 'upper'/'lower'
def calcTotalcost(alpha_s, alpha_m, mu_s, mu_m, gamma, i, val_lambda, val_lambda_m, mode):
	serverless_cost = (alpha_s * (val_lambda - val_lambda_m))/mu_s
	serve_jobs_cost = (alpha_m * val_lambda_m)/mu_m
	ub_num = (alpha_m * mu_m * (i**2)) - (val_lambda_m * alpha_m * i)
	ub_den = (gamma * i) + (mu_m * i) - (val_lambda_m)
	lb_num = ub_num
	lb_den = (2 * gamma * i) + (mu_m * i) - (val_lambda_m)
	# val_lb = (alpha_m/gamma)*(1/((1/mu_m) + (i/val_lambda) + (1/gamma)))*i
	val_orig = (alpha_m/gamma)*(1/((i/(i*mu_m - val_lambda_m)) + (i/val_lambda_m) + (1/gamma)))*i
	if mode == 'upper':
		if ub_den == 0:
			return 'div0'
		return serverless_cost + serve_jobs_cost + (ub_num/ub_den)
	elif mode == 'lower':
		if lb_den == 0:
			return 'div0'
		return serverless_cost + serve_jobs_cost + (lb_num/lb_den) # OLD Lower Bound
		# return serverless_cost + serve_jobs_cost + val_lb # NEW Lower Bound
	elif mode == 'original':
		if mu_m*i == val_lambda_m:
			return 'div0'
		return serverless_cost + serve_jobs_cost + val_orig # Actual cost
		# return serve_jobs_cost # Serving jobs cost only
	elif mode == 'setup_original':
		if mu_m*i == val_lambda_m:
			return 'div0'
		return val_orig
	else:
		print 'wrong mode to calcVMcost'
	return 'wrong mode to calcVMcost'

def plotVMcost():
	
	alpha_m = 1.0
	gamma = 2.0
	i = 5.0
	mu_m = 5.0
	lambdas = [x for x in range(1,int(mu_m*i))]
	results = []
	results_ub = []
	results_lb = []
	results_orig = []
	results_setup_orig = []
	xaxis = []
	xaxis_ub = []
	xaxis_lb = []
	xaxis_orig = []
	xaxis_setup_orig = []
	for val_lambda in lambdas:
		val_ub = calcVMcost(alpha_m, mu_m, gamma, i, val_lambda, 'upper')
		val_lb = calcVMcost(alpha_m, mu_m, gamma, i, val_lambda, 'lower')
		val_orig = calcVMcost(alpha_m, mu_m, gamma, i, val_lambda, 'original')
		val_setup_orig = calcVMcost(alpha_m, mu_m, gamma, i, val_lambda, 'setup_original')
		# print val_lb
		if val_ub != 'div0':
			results_ub.append(val_ub)	
			xaxis_ub.append(val_lambda)
		if val_lb != 'div0':
			results_lb.append(val_lb)
			xaxis_lb.append(val_lambda)
		if val_orig != 'div0':
			results_orig.append(val_orig)
			xaxis_orig.append(val_lambda)
		if val_setup_orig != 'div0':
			results_setup_orig.append(val_setup_orig)
			xaxis_setup_orig.append(val_lambda)
	results.append(results_ub)
	results.append(results_lb)
	results.append(results_orig)
	results.append(results_setup_orig)
	xaxis.append(xaxis_ub)
	xaxis.append(xaxis_lb)
	xaxis.append(xaxis_orig)
	xaxis.append(xaxis_setup_orig)
	legends = ['upper bound', 'lower bound', 'actual', 'setup cost actual']
	variables = ",new_LB,i=" + str(i) + ",alpha_m=" + str(alpha_m) + ",mu_m=" + str(mu_m) + ",gamma=" + str(gamma)
	title = "VM cost" + variables
	filename = '../graphs/VMcost' + variables + '.png'
	fig = plt.figure()
	plt.plot(xaxis[0], results[0], 'ro')
	plt.plot(xaxis[1], results[1], 'bo')
	plt.plot(xaxis[2], results[2], 'g*')
	plt.plot(xaxis[3], results[3], 'ms')
	plt.ylabel('Total VM Cost')
	if len(legends) != 0:
		plt.legend(legends, loc='upper left')
	fig.suptitle(title)
	plt.xlabel(r'$\lambda$')
	plt.savefig(filename)

# mode : 'upper'/'lower'
def calcVMcost(alpha_m, mu_m, gamma, i, lambda_m, mode):
	serve_jobs_cost = (alpha_m * lambda_m)/mu_m
	ub_num = (alpha_m * mu_m * (i**2)) - (lambda_m * alpha_m * i)
	ub_den = (gamma * i) + (mu_m * i) - (lambda_m)
	lb_num = ub_num
	lb_den = (2 * gamma * i) + (mu_m * i) - (lambda_m)
	# val_lb = (alpha_m/gamma)*(1/((1/mu_m) + (i/lambda_m) + (1/gamma)))*i
	val_orig = (alpha_m/gamma)*(1/((i/(i*mu_m - lambda_m)) + (i/lambda_m) + (1/gamma)))*i
	if mode == 'upper':
		if ub_den == 0:
			return 'div0'
		return serve_jobs_cost + (ub_num/ub_den) # Upper Bound
		# return (ub_num/ub_den)  # startup cost only
	elif mode == 'lower':
		if lb_den == 0:
			return 'div0'
		return serve_jobs_cost + (lb_num/lb_den) # Lower Bound
		# return (lb_num/lb_den) # startup cost only
	elif mode == 'original':
		if mu_m*i == lambda_m:
			return 'div0'
		return serve_jobs_cost + val_orig # Actual cost
		# return serve_jobs_cost # Serving jobs cost only
	elif mode == 'setup_original':
		if mu_m*i == lambda_m:
			return 'div0'
		return val_orig
	else:
		print 'wrong mode to calcVMcost'
	return 'wrong mode to calcVMcost'


def plotmultipleVMs_vary_mu():
	max_lambda = 24
	lambdas = [.01*x for x in range(0,((max_lambda+1)*100))]
	alpha_v_multiple = 1.0
	alpha_s = 3*alpha_v_multiple
	mu_ratios = [1,2,3]
	mu_v_multiple = 7.0
	mu_s = 7
	beta = 0.9
	results = []
	for ratio in mu_ratios:
		print "mu ratio = %d" % ratio
		results_price = []
		mu_v = ratio * mu_v_multiple
		alpha_v = ratio * alpha_v_multiple
		Ls = (alpha_v/alpha_s)*mu_s
		Lv = beta*mu_v
		print "Ls = %d\tLv = %d" % (Ls, Lv)
 		for val_lambda in lambdas:
 			cost = 0
			if Lv <= Ls: 
				cost = alpha_s*(val_lambda/mu_s)
			else:
				k = math.floor(val_lambda/Lv)
				print k
				V = 0			# number of VMs
				S = 0			# workload to serverless
				if val_lambda-(math.floor(val_lambda/Lv)*Lv) < Ls:	
					V = k
					S = val_lambda-(math.floor(val_lambda/Lv)*Lv)
					print "V+S"
				else:
					V = k+1
					S = 0
					print "V"
				Cv = alpha_v*V
				Cs = alpha_s*(S/mu_s)
				cost = Cv + Cs
			results_price.append(cost)
		results.append(results_price)

	filename = '../graphs/multVMs_vary_mu'  + '.png'
	fig = plt.figure()
	legends = []
	for ratio in mu_ratios:
		key = r'$\mu_v$=' + str(ratio) + r'$\mu_s$' + ', ' + r'$\alpha_v$=' + str(ratio)
		legends.append(key)
	plt.plot(lambdas[::200], results[0][::200], 'ro', markersize=7)
	plt.plot(lambdas[::200], results[1][::200], 'g^', markersize=7)
	plt.plot(lambdas[::200], results[2][::200], 'bs', markersize=7)
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

def plotmultipleVMs_vary_price_ratio():
	max_lambda = 24
	lambdas = [.01*x for x in range(0,((max_lambda+1)*100))]
	alpha_v = 1.0
	price_ratios = [1.3,3,10]
	mu_v = 7.0
	mu_s = 10.0
	beta = 0.9

	results = []
	for ratio in price_ratios:
		print "price ratio = %d" % ratio
		results_price = []
		alpha_s = ratio * alpha_v
		Ls = (alpha_v/alpha_s)*mu_s
		Lv = beta*mu_v
		print "Ls = %d\tLv = %d" % (Ls, Lv)
 		for val_lambda in lambdas:
 			cost = 0
			if Lv <= Ls: 
				cost = alpha_s*(val_lambda/mu_s)
			else:
				k = math.floor(val_lambda/Lv)
				print k
				V = 0			# number of VMs
				S = 0			# workload to serverless
				if val_lambda-(math.floor(val_lambda/Lv)*Lv) < Ls:	
					V = k
					S = val_lambda-(math.floor(val_lambda/Lv)*Lv)
					print "V+S"
				else:
					V = k+1
					S = 0
					print "V"
				Cv = alpha_v*V
				Cs = alpha_s*(S/mu_s)
				cost = Cv + Cs
			results_price.append(cost)
		results.append(results_price)
	# print results
	
	filename = '../graphs/multVMs_vary_price_ratios'  + '.png'
	fig = plt.figure()
	legends = []
	for ratio in price_ratios:
		# key = "hello" + str(ratio) 
		key = r'$\alpha_s$=' + str(ratio) + r'$\alpha_v$'
		legends.append(key)
	plt.plot(lambdas[::200], results[0][::200], 'ro', markersize=7)
	plt.plot(lambdas[::200], results[1][::200], 'g^', markersize=7)
	plt.plot(lambdas[::200], results[2][::200], 'bs', markersize=7)
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

def plotmultipleVMonoff_vary_price_ratio():
	max_lambda = 24
	lambdas = [.01*x for x in range(0,((max_lambda+1)*100))]
	alpha_v = 1.0
	price_ratios = [1,1.4,3,10]
	mu_v = 10.0
	mu_s = 12.0
	beta = 0.9
	gamma = 1

	results = []
	for ratio in price_ratios:
		print "price ratio = %f" % ratio
		results_price = []
		alpha_s = ratio * alpha_v
		Lv = beta * mu_v
		# compute Lv'
		Lv_a = beta - 1.0
		Lv_b = mu_v + gamma - (beta * mu_v)
		Lv_c = -beta * gamma * mu_v
		Lv_prime = solve_quadratic_eq2(Lv_a, Lv_b, Lv_c)[0]
		print "Lv_prime = %f\tLv = %f" % (Lv_prime, Lv)

		if(alpha_s/mu_s) >= (alpha_v * ((1/gamma) + (1/mu_v))):					# Serverless never, VM on-off until VM On
			for val_lambda in lambdas:
	 			cost = 0
				k = math.floor(val_lambda/Lv)
				extra_lambda = val_lambda-(math.floor(val_lambda/Lv)*Lv)
				if Lv_prime < Lv:				# VM on-OFF till VM on
					if extra_lambda < Lv_prime:
						V = k
						print "V(%d ON + 1 On-Off)" % k
						Cv = alpha_v*(k + (((1/(mu_v-extra_lambda))+(1/gamma))/((1/(mu_v-extra_lambda))+(1/gamma)+(1/extra_lambda))))
						cost = Cv
					else:						# VM on always
						V = k+1
						print "V (%d ON)" % V
						Cv = alpha_v*V
						cost = Cv
				else:								# VM on always
					V = k+1
					print "V (%d ON)" % V
					Cv = alpha_v*V
					cost = Cv
				results_price.append(cost)
		else:							# Serverless + On-off + On			
			# compute Ls'
			Ls_a = alpha_s
			Ls_b = (-alpha_s*mu_v) - (alpha_v*mu_s)
			Ls_c = (alpha_v*mu_s)*(gamma+mu_v) - (alpha_s*mu_v*gamma)
			Ls_roots = solve_quadratic_eq(Ls_a, Ls_b, Ls_c)
			if len(Ls_roots) == 0:
				for val_lambda in lambdas:
					cost = alpha_s*(val_lambda/mu_s)
					results_price.append(cost)
			else:
				Ls_prime = Ls_roots[0]
				print "Ls_prime = %f\tLv_prime = %f\tLv = %f" % (Ls_prime, Lv_prime, Lv)
				for val_lambda in lambdas:
					cost = 0
					k = math.floor(val_lambda/Lv)
					extra_lambda = val_lambda-(math.floor(val_lambda/Lv)*Lv)
					if Ls_prime <= Lv_prime:			# serverless till Ls_prime, then go to on-off
						if extra_lambda <= Ls_prime:				# VM on-OFF till VM on
							V = k
							S = extra_lambda
							print "V(%d ON)+S" % k
							Cv = alpha_v*V
							Cs = alpha_s*(S/mu_s)
							cost = Cv + Cs
						elif Lv > Lv_prime and extra_lambda < Lv_prime:
							V = k
							print "V(%d ON + 1 On-Off)" % k
							Cv = alpha_v*(k + (((1/(mu_v-extra_lambda))+(1/gamma))/((1/(mu_v-extra_lambda))+(1/gamma)+(1/extra_lambda))))
							cost = Cv
						else:								# VM on always
							V = k+1
							print "V (%d ON)" % V
							Cv = alpha_v*V
							cost = Cv
					else:							# serverless till Ls_prime, then go to ON
						if extra_lambda <= Ls_prime:
							V = k
							S = extra_lambda
							print "V(%d ON)+S" % k
							Cv = alpha_v*V
							Cs = alpha_s*(S/mu_s)
							cost = Cv + Cs
						else:
							V = k+1
							print "V (%d ON)" % V
							Cv = alpha_v*V
							cost = Cv
					results_price.append(cost)
		results.append(results_price)
	filename = '../graphs/multVMsonoff_vary_price_ratios'  + '.png'
	fig = plt.figure()
	legends = []
	for ratio in price_ratios:
		# key = "hello" + str(ratio) 
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
		# compute Lv'
		Lv_a = beta - 1.0
		Lv_b = mu_v + gamma - (beta * mu_v)
		Lv_c = -beta * gamma * mu_v
		Lv_prime = solve_quadratic_eq2(Lv_a, Lv_b, Lv_c)[0]
		print "Lv_prime = %f\tLv = %f" % (Lv_prime, Lv)

		if(alpha_s/mu_s) >= (alpha_v * ((1/gamma) + (1/mu_v))):					# Serverless never, VM on-off until VM On
			for val_lambda in lambdas:
	 			cost = 0
				k = math.floor(val_lambda/Lv)
				extra_lambda = val_lambda-(math.floor(val_lambda/Lv)*Lv)
				if Lv_prime < Lv:				# VM on-OFF till VM on
					if extra_lambda < Lv_prime:
						V = k
						print "V(%d ON + 1 On-Off)" % k
						Cv = alpha_v*(k + (((1/(mu_v-extra_lambda))+(1/gamma))/((1/(mu_v-extra_lambda))+(1/gamma)+(1/extra_lambda))))
						cost = Cv
					else:						# VM on always
						V = k+1
						print "V (%d ON)" % V
						Cv = alpha_v*V
						cost = Cv
				else:								# VM on always
					V = k+1
					print "V (%d ON)" % V
					Cv = alpha_v*V
					cost = Cv
				results_price.append(cost)
		else:							# Serverless + On-off + On			
			# compute Ls'
			Ls_a = alpha_s
			Ls_b = (-alpha_s*mu_v) - (alpha_v*mu_s)
			Ls_c = (alpha_v*mu_s)*(gamma+mu_v) - (alpha_s*mu_v*gamma)
			Ls_roots = solve_quadratic_eq(Ls_a, Ls_b, Ls_c)
			if len(Ls_roots) == 0:
				for val_lambda in lambdas:
					cost = alpha_s*(val_lambda/mu_s)
					results_price.append(cost)
			else:
				Ls_prime = Ls_roots[0]
				print "Ls_prime = %f\tLv_prime = %f\tLv = %f" % (Ls_prime, Lv_prime, Lv)
				for val_lambda in lambdas:
					cost = 0
					k = math.floor(val_lambda/Lv)
					extra_lambda = val_lambda-(math.floor(val_lambda/Lv)*Lv)
					if Ls_prime <= Lv_prime:			# serverless till Ls_prime, then go to on-off
						if extra_lambda <= Ls_prime:				# VM on-OFF till VM on
							V = k
							S = extra_lambda
							print "V(%d ON)+S" % k
							Cv = alpha_v*V
							Cs = alpha_s*(S/mu_s)
							cost = Cv + Cs
						elif Lv > Lv_prime and extra_lambda < Lv_prime:
							V = k
							print "V(%d ON + 1 On-Off)" % k
							Cv = alpha_v*(k + (((1/(mu_v-extra_lambda))+(1/gamma))/((1/(mu_v-extra_lambda))+(1/gamma)+(1/extra_lambda))))
							cost = Cv
						else:								# VM on always
							V = k+1
							print "V (%d ON)" % V
							Cv = alpha_v*V
							cost = Cv
					else:							# serverless till Ls_prime, then go to ON
						if extra_lambda <= Ls_prime:
							V = k
							S = extra_lambda
							print "V(%d ON)+S" % k
							Cv = alpha_v*V
							Cs = alpha_s*(S/mu_s)
							cost = Cv + Cs
						else:
							V = k+1
							print "V (%d ON)" % V
							Cv = alpha_v*V
							cost = Cv
					results_price.append(cost)
		results.append(results_price)
	filename = '../graphs/multVMsonoff_vary_mu'  + '.png'
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
		print "price ratio = %f" % ratio
		results_price = []
		gamma = ratio * gamma_multiple
		Lv = beta * mu_v
		# compute Lv'
		Lv_a = beta - 1.0
		Lv_b = mu_v + gamma - (beta * mu_v)
		Lv_c = -beta * gamma * mu_v
		Lv_prime = solve_quadratic_eq2(Lv_a, Lv_b, Lv_c)[0]
		print "Lv_prime = %f\tLv = %f" % (Lv_prime, Lv)

		if(alpha_s/mu_s) >= (alpha_v * ((1/gamma) + (1/mu_v))):					# Serverless never, VM on-off until VM On
			for val_lambda in lambdas:
	 			cost = 0
				k = math.floor(val_lambda/Lv)
				extra_lambda = val_lambda-(math.floor(val_lambda/Lv)*Lv)
				if Lv_prime < Lv:				# VM on-OFF till VM on
					if extra_lambda < Lv_prime:
						V = k
						print "V(%d ON + 1 On-Off)" % k
						Cv = alpha_v*(k + (((1/(mu_v-extra_lambda))+(1/gamma))/((1/(mu_v-extra_lambda))+(1/gamma)+(1/extra_lambda))))
						cost = Cv
					else:						# VM on always
						V = k+1
						print "V (%d ON)" % V
						Cv = alpha_v*V
						cost = Cv
				else:								# VM on always
					V = k+1
					print "V (%d ON)" % V
					Cv = alpha_v*V
					cost = Cv
				results_price.append(cost)
		else:							# Serverless + On-off + On			
			# compute Ls'
			Ls_a = alpha_s
			Ls_b = (-alpha_s*mu_v) - (alpha_v*mu_s)
			Ls_c = (alpha_v*mu_s)*(gamma+mu_v) - (alpha_s*mu_v*gamma)
			Ls_roots = solve_quadratic_eq(Ls_a, Ls_b, Ls_c)
			if len(Ls_roots) == 0:
				for val_lambda in lambdas:
					cost = alpha_s*(val_lambda/mu_s)
					results_price.append(cost)
			else:
				Ls_prime = Ls_roots[0]
				print "Ls_prime = %f\tLv_prime = %f\tLv = %f" % (Ls_prime, Lv_prime, Lv)
				for val_lambda in lambdas:
					cost = 0
					k = math.floor(val_lambda/Lv)
					extra_lambda = val_lambda-(math.floor(val_lambda/Lv)*Lv)
					if Ls_prime <= Lv_prime:			# serverless till Ls_prime, then go to on-off
						if extra_lambda <= Ls_prime:				# VM on-OFF till VM on
							V = k
							S = extra_lambda
							print "V(%d ON)+S" % k
							Cv = alpha_v*V
							Cs = alpha_s*(S/mu_s)
							cost = Cv + Cs
						elif Lv > Lv_prime and extra_lambda < Lv_prime:
							V = k
							print "V(%d ON + 1 On-Off)" % k
							Cv = alpha_v*(k + (((1/(mu_v-extra_lambda))+(1/gamma))/((1/(mu_v-extra_lambda))+(1/gamma)+(1/extra_lambda))))
							cost = Cv
						else:								# VM on always
							V = k+1
							print "V (%d ON)" % V
							Cv = alpha_v*V
							cost = Cv
					else:							# serverless till Ls_prime, then go to ON
						if extra_lambda <= Ls_prime:
							V = k
							S = extra_lambda
							print "V(%d ON)+S" % k
							Cv = alpha_v*V
							Cs = alpha_s*(S/mu_s)
							cost = Cv + Cs
						else:
							V = k+1
							print "V (%d ON)" % V
							Cv = alpha_v*V
							cost = Cv
					results_price.append(cost)
		results.append(results_price)
	filename = '../graphs/multVMsonoff_vary_gamma'  + '.png'
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

def plotcostserv_to_vm_ON():
	max_lambda = 3
	lambdas = [x for x in range(1, max_lambda+1)]
	mu_s = 10.0
	max_alpha_v = 7 
	alpha_v_range = [0.01*x for x in range(0,((max_alpha_v+1)*100))]

	results = []
	for val_lambda in lambdas:
		results_alpha_s = []
		for alpha_v in alpha_v_range:
			alpha_s = alpha_v * (mu_s/val_lambda)
			results_alpha_s.append(alpha_s)
		results.append(results_alpha_s)
	filename = '../graphs/as_vs_av_ON'  + '.png'
	fig = plt.figure()
	legends = []
	for val_lambda in lambdas: 
		key = r'$\lambda$' + ' = ' + str(val_lambda)
		legends.append(key)
	plt.plot(alpha_v_range[::100], results[0][::100], 'ro', markersize=7)
	plt.plot(alpha_v_range[::100], results[1][::100], 'g^', markersize=7)
	plt.plot(alpha_v_range[::100], results[2][::100], 'bs', markersize=7)
	plt.plot(alpha_v_range, results[0], 'r', linewidth='2')
	plt.plot(alpha_v_range, results[1], 'g', linewidth='2')
	plt.plot(alpha_v_range, results[2], 'b', linewidth='2')
	# plt.plot(lambdas, results[3], 'cD')
	plt.legend(legends, loc='upper left', fontsize=22)
	plt.ylabel(r'$\alpha_s$', fontsize=25)
	plt.ylim([0,90])
	# title = "Multiple VMs, different price ratios"
	# fig.suptitle(title)
	plt.xlabel(r'$\alpha_v$', fontsize=25)
	plt.savefig(filename)

def plotcostserv_to_vm_ON_OFF():
	max_lambda = 3
	lambdas = [x for x in range(1, max_lambda+1)]
	mu_v = 7.0
	mu_s = 10.0
	max_alpha_v = 7 
	alpha_v_range = [0.01*x for x in range(0,((max_alpha_v+1)*100))]
	gamma = 1.0

	results = []
	for val_lambda in lambdas:
		results_alpha_s = []
		for alpha_v in alpha_v_range:
			alpha_s = alpha_v * (mu_s/val_lambda) * (  (1/gamma) + (1/(mu_v - val_lambda) )/(  (1/gamma) + (1/(mu_v - val_lambda) ) + (1/val_lambda)  )    )
			results_alpha_s.append(alpha_s)
		results.append(results_alpha_s)
	filename = '../graphs/as_vs_av_ON_OFF'  + '.png'
	fig = plt.figure()
	legends = []
	for val_lambda in lambdas: 
		key = r'$\lambda$' + ' = ' + str(val_lambda)
		legends.append(key)
	plt.plot(alpha_v_range[::100], results[0][::100], 'ro', markersize=7)
	plt.plot(alpha_v_range[::100], results[1][::100], 'g^', markersize=7)
	plt.plot(alpha_v_range[::100], results[2][::100], 'bs', markersize=7)
	plt.plot(alpha_v_range, results[0], 'r', linewidth='2')
	plt.plot(alpha_v_range, results[1], 'g', linewidth='2')
	plt.plot(alpha_v_range, results[2], 'b', linewidth='2')
	plt.legend(legends, loc='upper left', fontsize=22)
	plt.ylabel(r'$\alpha_s$', fontsize=25)
	# title = "Multiple VMs, different price ratios"
	# fig.suptitle(title)
	plt.ylim([0,90])
	plt.xlabel(r'$\alpha_v$', fontsize=25)
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
			mu_s = (alpha_s/alpha_v) * val_lambda * (1 / (  (1/gamma) + (1/(mu_v - val_lambda) )/(  (1/gamma) + (1/(mu_v - val_lambda) ) + (1/val_lambda)  )    ))
			results_mu_v.append(mu_s)
		results.append(results_mu_v)
		mu_v_values_final.append(mu_v_values)
	filename = '../graphs/mus_vs_muv_ON_OFF'  + '.png'
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
	if len(sys.argv) != 2:
		print "USAGE: python generate_plots.py <exp_type>"
		print "<exp_type> : vary_num_VMs/vary_startup_delay/vary_service_rate_VM/plotVMcost/plotTotalcost"
		print "<exp_type> : multipleVMs_vary_price_ratio/multipleVMs_vary_mu/multipleVMonoff_vary_price_ratio/multipleVMonoff_vary_mu/multipleVMonoff_vary_gamma"
		print "<exp_type> : costserv_to_vm_ON/costserv_to_vm_ON_OFF"
		print "<exp_type> : mus_to_muv_ON_OFF"
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
	else:
		print "Wrong <exp_type>"
		print "<exp_type> : vary_num_VMs/vary_startup_delay/vary_service_rate_VM/plotVMcost/plotTotalcost"
		print "<exp_type> : multipleVMs_vary_price_ratio/multipleVMs_vary_mu/multipleVMonoff_vary_price_ratio/multipleVMonoff_vary_mu/multipleVMonoff_vary_gamma"
		print "<exp_type> : costserv_to_vm_ON/costserv_to_vm_ON_OFF"
		print "<exp_type> : mus_to_muv_ON_OFF"

if __name__ == '__main__':
	main()
