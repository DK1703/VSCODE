import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**4 - 2.5*x**2 + 1

n = 10
M = 100
t_star = 0.2

h = 1/n
tau = t_star/M

u = np.zeros((M+1, n+1))
u[0] = f(np.linspace(0, 1, n+1))
u[:, 0] = 1
u[:, n] = u[:, n-1] - h

for m in range(M):
    for j in range(1, n):
        u[m+1, j] = u[m, j] + tau*(u[m, j-1] - 2*u[m, j] + u[m, j+1])/h**2

#Exact solution
exact_u = np.zeros((M+1, n+1))
for m in range(M+1):
    for j in range(n+1):
        exact_u[m, j] = np.exp(-m*tau)*(f(j*h) - (np.exp(1) - 1)/(np.exp(1) + 1)*(1 - np.exp(-(np.exp(1) + 1)/2*m*tau)*np.cos(j*h)))

#Convergence analysis
h_values = []
err_values = []
for i in range(3):
    n_new = n*(2**i)
    h_new = 1/n_new
    tau_new = (t_star/M)*(n/n_new)**2
    u_new = np.zeros((M+1, n_new+1))
    u_new[0] = f(np.linspace(0, 1, n_new+1))
    u_new[:, 0] = 1
    u_new[:, n_new] = u_new[:, n_new-1] - h_new
    
    for m in range(M):
        for j in range(1, n_new):
            u_new[m+1, j] = u_new[m, j] + tau_new*(u_new[m, j-1] - 2*u_new[m, j] + u_new[m, j+1])/h_new**2
    
    #Exact solution for new grid
    exact_u_new = np.zeros((M+1, n_new+1))
    for m in range(M+1):
        for j in range(n_new+1):
            exact_u_new[m, j] = np.exp(-m*tau_new)*(f(j*h_new) - (np.exp(1) - 1)/(np.exp(1) + 1)*(1 - np.exp(-(np.exp(1) + 1)/2*m*tau_new)*np.cos(j*h_new)))
    
    #Error calculation
    err = np.max(np.abs(u_new - exact_u_new))
    h_values.append(h_new)
    err_values.append(err)
    
#Convergence rate
convergence_order_h = -(np.log(err_values[2])-np.log(err_values[1]))/np.log(2)
convergence_order_tau = -(np.log(err_values[1])-np.log(err_values[0]))/np.log(2)

#Plotting
plt.figure(figsize=(10,5))
plt.suptitle("Numerical Solution")
plt.subplot(121)
plt.imshow(u, aspect='auto')
plt.xlabel("x")
plt.ylabel("t")
plt.title("Numerical solution for heat conduction equation")
plt.colorbar()
plt.subplot(122)
plt.imshow(exact_u, aspect='auto')
plt.xlabel("x")
plt.ylabel("t")
plt.title("Exact solution for heat conduction equation")
plt.colorbar()

plt.figure(figsize=(8,6))
plt.loglog(h_values, err_values, marker='o')
plt.xlabel("h")
plt.ylabel("Error")
plt.title("Convergence of the numerical method")
plt.show()
