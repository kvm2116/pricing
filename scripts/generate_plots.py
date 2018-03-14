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
		return [x]
	else:
		x1 = (-b+math.sqrt(d))/(2*a)
		x2 = (-b-math.sqrt(d))/(2*a)
		print "This equation has two solutions: ", x1, " and", x2
		return [x1,x2]

# mode : upper/lower
def calc_lambda_m(alpha_m, alpha_s, mu_m, mu_s, gamma, i, mode):
	# lambda_m = -((gamma * mu_m * alpha_s * n)/(2 * mu_s * alpha_m)) + ((gamma + mu_m)*n/2) 
	a_ub = (alpha_m * mu_s) - (alpha_s * mu_m)
	b_ub = (2 * alpha_s * mu_m * gamma * i) + (2 * alpha_s * (mu_m**2) * i) - (2 * alpha_m * mu_s * gamma * i) - (2 * alpha_m * mu_m * mu_s * i)
	c_ub = (alpha_m * mu_m * mu_s * gamma * (i**2)) + (alpha_m * mu_s * (gamma**2) * (i**2)) + (alpha_m * (mu_m**2) * mu_s * (i**2)) - (2 * alpha_s * (mu_m**2) * gamma * (i**2)) - (alpha_s * mu_m * (gamma**2) * (i**2)) - (alpha_s * (mu_m**3) * (i**2))
	a_lb = a_ub
	b_lb = (2 * alpha_s * (mu_m**2) * i) + (4 * alpha_s * mu_m * gamma * i) - (2 * alpha_m * mu_m * mu_s * i) - (4 * alpha_m * mu_s * gamma * i)
	c_lb = (4 * alpha_m * mu_s * (gamma**2) * (i**2)) + (alpha_m * (mu_m**2) * mu_s * (i**2)) - (4 * alpha_s * mu_m * (gamma**2) * (i**2)) - (alpha_s * (mu_m**3) * (i**2)) - (2 * alpha_s * (mu_m**2) * gamma * (i**2))
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

def plotVMcost():
	lambdas = [x for x in range(1,101)]
	alpha_m = 1
	gamma = 1
	i = 5
	mu_m = 5
	results = []
	results_ub = []
	results_lb = []
	for val_lambda in lambdas:
		results_ub.append(calcVMcost(alpha_m, mu_m, gamma, i, val_lambda, 'upper'))
		results_lb.append(calcVMcost(alpha_m, mu_m, gamma, i, val_lambda, 'lower'))
	results.append(results_ub)
	results.append(results_lb)
	legends = ['upper bound', 'lower bound']
	title = "VM cost"
	filename = '../graphs/VMcost' + '.png'
	create_plot(filename, title, legends, lambdas, results)

# mode : 'upper'/'lower'
def calcVMcost(alpha_m, mu_m, gamma, i, lambda_m, mode):
	serve_jobs_cost = (alpha_m * lambda_m)/mu_m
	ub_num = (alpha_m * mu_m * (i**2)) - (lambda_m * alpha_m * i)
	ub_den = (gamma * i) + (mu_m * i) - (lambda_m)
	lb_num = ub_num
	lb_den = (2 * gamma * i) + (mu_m * i) - (lambda_m)
	if mode == 'upper':
		return ub_num/ub_den
	elif mode == 'lower':
		return lb_num/lb_den
	else:
		print 'wrong mode to calcVMcost'
	return 'wrong mode to calcVMcost'

def main():
	if len(sys.argv) != 2:
		print "USAGE: python generate_plots.py <exp_type>"
		print "<exp_type> : vary_num_VMs/vary_startup_delay/vary_service_rate_VM/plotVMcost"
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
	else:
		print "Wrong <exp_type>"
		print "<exp_type> : vary_num_VMs/vary_startup_delay/vary_service_rate_VM/plotVMcost"

if __name__ == '__main__':
	main()
