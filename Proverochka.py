import numpy as np
import matplotlib.pyplot as plt

# Определяем точное решение
def u_exact(x, y):
    return np.exp(x*y)*(x**2 + y**2 - 4 * (np.pi**2) - 16 * (np.pi**2) * x**2)*np.cos(2*np.pi*(x**2 + y)) - 4 * np.pi * np.exp(x*y)*(1 + x + 2 * x * y) * np.sin(2*np.pi*(x**2 + y))

# Определяем функции для граничных условий и исходного члена
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

def u_xx(x,y):
    return -4*np.pi*(y**2)*np.exp(x*y)*b(x,y)*si(x,y) - 8*np.pi*x*(y*np.exp(x*y)*a(x,y)+x*c*np.exp(x*y))*si(x,y) + (y**2 * np.exp(x*y)*a(x,y) + 2*x*y*c*np.exp(x*y) + c*np.exp(x*y))*co(x,y) + np.exp(x*y)*a(x,y)*(-4*np.pi*si(x,y)-16*(np.pi**2)*(x**2)*co(x,y)) - 8*np.pi*y*np.exp(x*y)*((2*y+1)*si(x,y) + 4*np.pi*x*b(x,y)*co(x,y)) + 4*np.pi*np.exp(x*y)*(8*np.pi*x*(2*y+1)*co(x,y) + b(x,y)*(4*np.pi*co(x,y) - 16*(np.pi**2)*(x**2)*si(x,y)))

def u_yy(x, y):
    return np.exp(x*y) * ((x**4 + (x**2)*(y**2) + 16*(np.pi**4)*(4*x**2 + 1) - 4 * np.pi**2 * (4*x**4 + (x**2) * (8*y + 6) + 12 * x + y**2) * np.cos(2*np.pi*(x**2 + y))) -8*np.pi*x**3*np.exp(x*y)*a(x,y)*co(x,y)+ 4*np.pi*(-2*x**3 * (y + 1) + 4 * np.pi**2 * (4*x**3 + 2*x*(y+1) + 1) - 5*x**2 - x*y**2 - 2*y)* np.sin(2*np.pi*(x**2 + y)))

def f(x, y):
    return u_xx(x ,y) + u_yy(x, y)

# Определяем итерационный метод (Гаусса-Зейделя)
def gauss_seidel(N, h):
    # Инициализируем сетку значений
    u = np.zeros((N+1, N+1))
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    # Создаем массивы координат для сетки
    xx, yy = np.meshgrid(x, y)

    # Вычисляем значения функции в каждой точке сетки
    zz = u_exact(xx, yy)

    # Вычисляем лапласиан функции
    laplacian = np.gradient(np.gradient(zz, axis=0), axis=1)

    # Задаём граничные условия
    for i in range(N+1):
        u[i, 0] = phi1(i * h)
        u[i, N] = phi2(i * h)
    for j in range(N+1):
        u[0, j] = phi3(j * h)
        u[N, j] = phi4(j * h)
        
    # Итерации до достижения сходимости
    max_error = 1.0
    tolerance = 1e-6
    iteration = 0
    while max_error > tolerance:
        max_error = 0.0
        for i in range(1, N):
            for j in range(1, N):
                value = 0.25*(u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1] - h**2 * f(i * h, j * h))
                #error = abs(value - u_exact(i * h, j * h))
                error = abs(value - u[i, j])
                if error > max_error:
                    max_error = error
                u[i, j] = value
        iteration += 1
    
    return u, iteration


# Вычисляем решение для разных размеров сетки
N_values = [10, 20]
h_values = [1/N for N in N_values]
errors = []
iterations = []

for N, h in zip(N_values, h_values):
    u, it = gauss_seidel(N, h)
    print(np.max(np.abs(u - u_exact(np.arange(0, 1+h, h)[:, np.newaxis], np.arange(0, 1+h, h)[np.newaxis, :]))))
    errors.append(np.max(np.abs(u - u_exact(np.arange(0, 1+h, h)[np.newaxis, :], np.arange(0, 1+h, h)[:, np.newaxis]))))
    iterations.append(it)
    print(it)

# График зависимости ошибок от размера сетки
plt.loglog(N_values, errors, 'o-')
plt.xlabel('Grid spacing')
plt.ylabel('Max error')
plt.title('Convergence of numerical method')
plt.show()

# График зависимости количества итераций от размера сетки
plt.loglog(N_values, iterations, 'o-')
plt.xlabel('Grid spacing')
plt.ylabel('Iterations')
plt.title('Convergence of numerical method')
plt.show()
