from matplotlib import pyplot as plt
import numpy as np

x = np.linspace(0, 30, 100)
y = x
error = np.random.normal(0, 5, size=y.shape)

plt.plot(x, y, 'k-', color='blue')
plt.fill_between(x, y-error, y+error, color='blue', alpha=0.3)
plt.show()
