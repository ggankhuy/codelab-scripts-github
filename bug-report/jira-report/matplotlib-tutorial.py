
import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0., 5., 0.2)

#plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')

#plt.plot([1, 2, 3, 4], [10, 20, -4, 100], 'ro')
#plt.ylabel('some numbers')

'''
names = ['group_a', 'group_b', 'group_c']
values = [1, 10, 100]
plt.figure(figsize=(9, 3))
plt.subplot(131)
plt.bar(names, values)
plt.subplot(132)
plt.scatter(names, values)
plt.subplot(133)
plt.plot(names, values)
plt.suptitle('Categorical Plotting')
'''

data = {'a': np.arange(50),
        'c': np.random.randint(0, 50, 50),
        'd': np.random.randn(50)}
data['b'] = data['a'] + 10 * np.random.randn(50)
data['d'] = np.abs(data['d']) * 100

plt.scatter('a', 'b', c='c', s='d', data=data)
plt.xlabel('entry a')
plt.ylabel('entry b')
plt.show()
