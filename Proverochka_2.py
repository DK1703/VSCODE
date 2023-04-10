import numpy as np
import matplotlib.pyplot as plt

# define exact solution
def u_exact(x, y):
    return np.exp(x*y)*(x**2 + y**2 - 4 * (np.pi**2) - 16 * (np.pi**2) * x**2)*np.cos(2*np.pi*(x**2 + y)) - 4 * np.pi * np.exp(x*y)*(1 + x + 2 * x * y) * np.sin(2*np.pi*(x**2 + y))

# define functions for boundary conditions and source term
def phi1(y):
    return u_exact(0, y)

def phi2(y):
    return u_exact(1, y)

def phi3(x):
    return u_exact(x, 0)

def phi4(x):
    return u_exact(x, 1)

def a(x, y):
    return (x**2 + y**2 - 4 * (np.pi**2) - 16 * (np.pi**2) * x**2)*np.cos(2*np.pi*(x**2 + y))

def b(x, y):
    return 1 + x + 2*x*y

c = 2 - 32*(np.pi**2)

def si(x,y):
    return np.sin(2*np.pi*(x**2 + y))

def co(x,y):
    return np.cos(2*np.pi*(x**2 + y))

def proiz(x,y):
    return co(x,y) * si(x,y)

def u_xx(x,y):
    return -4*np.pi*(y**2)*np.exp(x*y)*b(x,y)*si(x,y) - 8*np.pi*x*(y*np.exp(x*y)*a(x,y)+x*c*np.exp(x*y))*si(x,y) + (y**2 * np.exp(x*y)*a(x,y) + 2*x*y*c*np.exp(x*y) + c*np.exp(x*y))*co(x,y) + np.exp(x*y)*a(x,y)*(-4*np.pi*si(x,y)-16*(np.pi**2)*(x**2)*co(x,y)) - 8*np.pi*y*np.exp(x*y)*((2*y+1)*si(x,y) + 4*np.pi*x*b(x,y)*co(x,y)) + 4*np.pi*np.exp(x*y)*(8*np.pi*x*(2*y+1)*co(x,y) + b(x,y)*(4*np.pi*co(x,y) - 16*(np.pi**2)*(x**2)*si(x,y)))

def u_yy(x, y):
    return np.exp(x*y) * ((x**4 + (x**2)*(y**2) + 16*(np.pi**4)*(4*x**2 + 1) - 4 * np.pi**2 * (4*x**4 + (x**2) * (8*y + 6) + 12 * x + y**2) * np.cos(2*np.pi*(x**2 + y))) + 4*np.pi*(-2*x**3 * (y + 1) + 4 * np.pi**2 * (4*x**3 + 2*x*(y+1) + 1) - 5*x**2 - x*y**2 - 2*y)* np.sin(2*np.pi*(x**2 + y)))

def f(x, y):
    return u_xx(x ,y) + u_yy(x, y) 


print(u_exact(1, 1))

# define iterative method (Gauss-Seidel)
def gauss_seidel(N, h):
    # initialize grid of values
    u = np.zeros((N+1, N+1))
    
    # set boundary conditions
    for i in range(N+1):
        u[i, 0] = phi1(i * h)
        u[i, N] = phi2(i * h)
    for j in range(N+1):
        u[0, j] = phi3(j * h)
        u[N, j] = phi4(j * h)
        
    # iterate until convergence
    max_error = 1.0
    tolerance = 1e-6
    iteration = 0
    while max_error > tolerance:
        max_error = 0.0
        for i in range(1, N):
            for j in range(1, N):
                value = 0.25*(u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1] - h**2 * f(i*h, j*h))
                error = abs(value - u[i, j])
                if error > max_error:
                    max_error = error
                u[i, j] = value
        iteration += 1
    
    return u, iteration

# compute solution for different grid sizes
N_values = [4, 8, 16]
h_values = [1/N for N in N_values]
errors = []
iterations = []

for N, h in zip(N_values, h_values):
    u, it = gauss_seidel(N, h)
    print(np.max(np.abs(u - u_exact(np.arange(0, 1+h, h), np.arange(0, 1+h, h)))))
    errors.append(np.max(np.abs(u - u_exact(np.arange(0, 1+h, h), np.arange(0, 1+h, h)))))
    iterations.append(it)
    print(it)

# plot errors vs. grid size
plt.loglog(h_values, errors, 'o-')
plt.xlabel('Grid spacing')
plt.ylabel('Max error')
plt.title('Convergence of numerical method')
plt.show()

# plot iterations vs. grid size
plt.loglog(h_values, iterations, 'o-')
plt.xlabel('Grid spacing')
plt.ylabel('Iterations')
plt.title('Convergence of numerical method')
plt.show() 
