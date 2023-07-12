import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10)
y = x

fig1 = plt.figure(1)
fig2 = plt.figure(2)

plt.plot(x,y)
print(plt.get_fignums())
plt.show()
