import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scienceplots
mpl.rcParams.update(mpl.rcParamsDefault)

# plt.rcParams['font.sans-serif'] = ['Times New Roman']
# plt.rcParams['axes.unicode_minus'] = False
plt.style.use(['science','ieee'])
# plt.style.use(['bmh'])

n = 210
nClusters = 3

mu = np.array([
    [0, 0],
    [0, 10],
    [0, 20]
])
sigma = np.array([
    [5, 5],
    [5, 5],
    [5, 5]
])
sigma = [np.diag(sigma[i]) for i in range(3)]

X_class1 = np.random.multivariate_normal(mu[0], sigma[0], size=int(n/nClusters))
X_class2 = np.random.multivariate_normal(mu[1], sigma[1], size=int(n/nClusters))
X_class3 = np.random.multivariate_normal(mu[2], sigma[2], size=int(n/nClusters))

plt.figure(figsize=(10, 6))

plt.scatter(X_class1[:, 0], X_class1[:, 1], color='r', label='Classe 1')
plt.scatter(X_class2[:, 0], X_class2[:, 1], color='g', label='Classe 2')
plt.scatter(X_class3[:, 0], X_class3[:, 1], color='b', label='Classe 3')
plt.xlabel('Eixo X')
plt.ylabel('Eixo Y')
plt.title('Gráfico 2D das Classes')
plt.legend()
plt.grid(True)


plt.xlim(-20, 20)
plt.ylim(-20, 50)

plt.show()