import os
import numpy as np
import matplotlib.pyplot as plt

directory = os.getcwd()+'/reconstructions/mnist/csdip/loss_plot/save2/'
mn_ratios = np.load(directory + 'mn_ratios.npy')
avg_losses = np.load(directory + 'avg_losses.npy')
measurement_list = mn_ratios*(28*28)

plt.plot(measurement_list, avg_losses)
plt.xlabel('m/n (sensing ratio)')
plt.ylabel('mse (per pixel)')
plt.title('mse loss vs sensing ratio')
plt.xlim(50,500)
plt.ylim(0,0.1)
plt.savefig(directory + 'loss_vs_measure2.png')

print(mn_ratios)
print(measurement_list)
print(avg_losses)