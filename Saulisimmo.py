import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**4 - 2.5*x**2 + 1

# Задаем начальные условия
N = 100
T = 0.5
h = 1/N
tau = h**2/2
M = int(T/tau)

u = np.zeros((M+1,N+1))

for j in range(N+1):
    u[0][j] = f(j*h)

for k in range(1, M+1):
    # Аппроксимация граничных условий
    u[k][0] = 1
    u[k][-1] = u[k-1][-1] - h

    # Метод последовательной прогонки
    alpha = np.zeros(N-1)
    beta = np.zeros(N-1)
    for j in range(1, N):
        A = tau/h**2
        B = tau/h**2
        C = 2*tau/h**2 + 1/tau
        F = u[k-1][j]/tau + (u[k-1][j+1] - 2*u[k-1][j] + u[k-1][j-1])/h**2
        alpha[j-1] = B/(C - A*alpha[j-2])
        beta[j-1] = (A*beta[j-2] + F)/(C - A*alpha[j-2])

    for j in range(N-2, -1, -1):
        u[k][j+1] = alpha[j]*u[k][j+2] + beta[j]

# График численного решения
x = np.linspace(0, 1, N+1)
t = np.linspace(0, T, M+1)
X, T = np.meshgrid(x, t)
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, u, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u(x,t)')
plt.show()

tn = 0
for k in range(1, M+1):
    if np.allclose(u[k], u[k-1], rtol=1e-5, atol=1e-8):
        tn = k*tau
        break
print('Time tn needed for solution to be established: ', tn)