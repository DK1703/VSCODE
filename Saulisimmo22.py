import numpy as np
import matplotlib.pyplot as plt

x_min = 0
x_max = 1
t_min = 0
t_max = 0.1
h = 0.1 
tau = 0.01 
N = int((x_max - x_min) / h)

def f(x):
    return x ** 4 - 2.5 * x ** 2 + 1

def exact(x, t):
    return np.exp(-t) * (f(x) + (np.exp(t) - 1) * (1 - x))

#u = np.zeros((int(t_max / tau) + 1, N + 1))

# initial condition
# for j in range(N + 1):
#     u[0, j] = f(j * h)

# boundary condition
# for n in range(int(t_max / tau) + 1):
#     u[n, 0] = 1
    #u[n, N] = u[n, N - 1] - h
    #u[n, N] = (-2.0 * tau * h + 2.0 * tau * u[n, N - 1] + (h**2) * u[n - 1, N]) / (2.0 * tau + h**2)

def exact_u(x ,t):
    sum = 0
    for k in range (1, 2):
        sum += np.exp(-k * 2 * t) * (f(x) * np.cos(k*x) - np.sin(k*x)/k)
    return sum

# exact solution
# u_exact = np.zeros((int(t_max / tau) + 1, N + 1))
# for n in range(int(t_max / tau) + 1):
#     for j in range(N + 1):
#         u_exact[n, j] = exact_u(j * h, n * tau)

# Saulyev scheme
# for n in range(int(t_max / tau)):
#     for j in range(1, N):
#         u[n + 1, j] = u[n, j] + 0.5 * tau * ((u[n, j + 1] - 2 * u[n, j] + u[n, j - 1]) / h ** 2 + 
#                                                (u[n + 1, j + 1] - 2 * u[n + 1, j] + u[n + 1, j - 1]) / h ** 2)
def Saul(N, M, h):
    u = np.zeros((M + 1, N + 1))
    for j in range(N + 1):
        u[0, j] = f(j * h)
    for n in range(M + 1):
        u[n, 0] = 1
    #u[n, N] = u[n, N - 1] - h
    #u[n, N] = (-2.0 * tau * h + 2.0 * tau * u[n, N - 1] + (h**2) * u[n - 1, N]) / (2.0 * tau + h**2)

    eps = 10**(-6)
    max_pogr = 1.0
    k = 1
    while (max_pogr > eps):
        max_pogr_1 = 0
        for j in range (1, N):
              u[k, j] = (tau * (u[k - 1, j + 1] + u[k, j - 1]) + (h**2 - tau) * u[k - 1, j]) /(h**2 + tau)
        u[k, N] = u[k, N - 1] - h 
        k += 1
        u[k, N - 1] = -tau/h + u[k - 1, N - 1]*(1 - tau/h**2) + u[k - 1, N - 2] * tau/h**2
        for j in range(N - 2, 0, -1): 
          u[k, j] = (tau * (u[k, j + 1] + u[k - 1, j - 1]) + (h**2 - tau) * u[k - 1, j]) /(h**2 + tau)
        u[k, N] = u[k, N - 1] - h 
        for i in range (0, N+1):
            #if np.abs(u[k, i] - u[k - 1, i]) > max_pogr_1: max_pogr_1 = np.abs(u[k, i] - u[k - 1, i])
            if np.abs(u[k, i] - exact(i * h, k * tau)) > max_pogr_1: max_pogr_1 = np.abs(u[k, i] - exact(i * h, k * tau))
        if max_pogr_1 < max_pogr: max_pogr = max_pogr_1
        k+= 1
        if k >= M: break
    
    return u, k, max_pogr


# Вычисляем решение для разных размеров сетки
N_values = [10, 20, 40]
M_values = [100*N for N in N_values]
h_values = [1/N for N in N_values]
errors = []
iterations = []

for N, M, h in zip(N_values, M_values, h_values):
    u, it, max_pogr = Saul(N, M, h)
    print(max_pogr)
    print(it)
    # График численного решения
    x = np.linspace(0, 1, N+1)
    t = np.linspace(0, t_max, M+1)
    X, T = np.meshgrid(x, t)
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, T, u, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u(x,t)')
    plt.show()




# plot
# T, X = np.meshgrid(np.linspace(t_min, t_max, int(t_max / tau) + 1), np.linspace(x_min, x_max, N + 1))
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, T, u_exact)
# ax.set_xlabel('t')
# ax.set_ylabel('x')
# ax.set_zlabel('u')
# plt.show()

# T, X = np.meshgrid(np.linspace(t_min, t_max, int(t_max / tau) + 1), np.linspace(x_min, x_max, N + 1))
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, T, u)
# ax.set_xlabel('t')
# ax.set_ylabel('x')
# ax.set_zlabel('u')
# plt.show()

# max error
# space_error = np.amax(np.abs(u_exact[int(t_max / tau)] - u[int(t_max / tau)]))
# time_error = np.amax(np.abs(u_exact - u))
#print(f"Max error in space at tn*={t_max}: {space_error}")
#print(f"Max error in time: {time_error}")

