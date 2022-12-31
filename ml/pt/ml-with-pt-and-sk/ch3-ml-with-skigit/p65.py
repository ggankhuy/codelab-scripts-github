import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def loss_1(z):
    return - np.log(sigmoid(z))

def loss_0(z):
    return - np.log(1-sigmoid(z))

z=np.arange(-10, 10, 0.1)
sigma_z = sigmoid(z)
print(f'z shape/min/max: {z.shape}/{min(z)}/{max(z)}')
print(f'sigma z shape/min/max: {sigma_z.shape}/{min(sigma_z)}/{max(sigma_z)}')
c1=[loss_1(x) for x in z]
plt.plot(sigma_z, c1, label='L(w,b) if y=1')
c0=[loss_0(x) for x in z]
print(f'c1 len/min/max: {len(c1)}/{min(c1)}/{max(c1)}')
print(f'c0 len/min/max: {len(c0)}/{min(c0)}/{max(c0)}')
plt.plot(sigma_z, c0, linestyle='--', label='L(w,b) if y=0')
plt.ylim(0.0,5.1)
plt.xlim([0,1])
plt.xlabel('$\sigma(z)$')
plt.ylabel('L(w.b)')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
