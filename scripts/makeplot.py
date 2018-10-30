

import os
import sys
import matplotlib.pyplot as plt
import numpy
import math


opt_1 = [2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0]
opt_2 = [2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0]
opt_3 = [2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 20.0, 20.0, 20.0, 20.0]
beta_v_values = [0.2, 0.4, 0.5] 
zipf_alpha_values = [0.1*x for x in range(1, 30)]


filename = '../graphs/mg1/ZipfDistAOVaryExp.png'
fig = plt.figure()
legends = []
for cp_alpha_v in beta_v_values:
	key = r'$\beta_v$=' + str(cp_alpha_v)
	legends.append(key)

plt.plot(zipf_alpha_values[::2], opt_1[::2], 'c*', markersize=7)
plt.plot(zipf_alpha_values[::2], opt_2[::2], 'ro', markersize=7)
plt.plot(zipf_alpha_values[::2], opt_3[::2], 'g^', markersize=7)
plt.plot(zipf_alpha_values, opt_1, 'c', linewidth='2')
plt.plot(zipf_alpha_values, opt_2, 'r', linewidth='2')
plt.plot(zipf_alpha_values, opt_3, 'g', linewidth='2')
plt.ylabel('Optimal ' + r'$\alpha_s$', fontsize=21)
plt.ylim(0, 25)
plt.legend(legends, loc='upper left', fontsize=21)
plt.xlabel('zipf exponent (' + r'$\gamma$' + ')', fontsize=20)

plt.savefig(filename)