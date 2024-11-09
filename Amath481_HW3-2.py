#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eig
from scipy.sparse.linalg import eigs
from scipy.integrate import solve_ivp ,odeint
from scipy.special import eval_hermite
from scipy.integrate import simpson

#Part a from hw 2

L = 4
x = np.linspace(-L, L, 81)
n = len(x)
tol = 1e-4

E0 = 0.1

Esola = []
ysola = []

def hw3_rhs_a(x, y, E):
    return [y[1], (x**2 - E) * y[0]]

for jmodes in range(1, 6):
    E = E0
    dE = 0.1
    for _ in range(1000):
        
        y0 = [1,np.sqrt(L**2 - E)]
        sol = solve_ivp(hw3_rhs_a, [-L, L], y0, args=(E,), t_eval=x)
        ys = sol.y.T
        
        bc = ys[-1, 1] + np.sqrt(L**2 - E) * ys[-1, 0]
        if abs(bc) < tol:
            Esola.append(E)  
            break  
        if (-1) ** (jmodes + 1) * bc > 0:
            E += dE
        else:
            E -= dE /2
            dE /= 2
    E0 = E + 0.1       
    norm = np.trapz(ys[:, 0] * ys[:, 0], x)  
    ysola.append(abs(ys[:, 0] / np.sqrt(norm)))
    
A1 = np.array(ysola).T
A2 = Esola
A2


# In[ ]:





# In[2]:



# part b
L = 4
dx = 0.1
x = np.arange(-L, L+dx, dx)
n = len(x)
A = np.zeros((n - 2, n - 2))

for j in range(n - 2):
    A[j, j] = -2 - (dx**2) * x[j + 1]**2  
    if j < n - 3:
        A[j + 1, j] = 1                  
        A[j, j + 1] = 1 
        
A[0, 0] = A[0, 0] + 4 / 3
A[0, 1] = A[0, 1] - 1 / 3
A[-1, -1] = A[-1, -1] + 4 / 3
A[-1, -2] = A[-1, -2] - 1 / 3


eigenvals, eigenvec = eigs(-A, k=5, which = "SM")
left_bound = 4/3 * eigenvec[0, :] -1/3 * eigenvec[1, :]
right_bound = 4/3 * eigenvec[-1, :] - 1/3 * eigenvec[-2, :]

ysolb = np.zeros((n,5))
Esolb = np.zeros(5)
v2 = np.vstack([left_bound, eigenvec, right_bound])

for j in range(5):
    norm = np.sqrt(np.trapz(v2[:, j] ** 2, x)) 
    ysolb[:, j] = abs(v2[:, j] / norm)
    
Esolb = eigenvals[:5] / (dx ** 2) 

A3 = ysolb
A4 = Esolb


# In[3]:


# Part c
def hw3_rhs_c(x, y, E, gamma):
    return [y[1], (gamma * np.abs(y[0])**2 + x**2 - E) * y[0]]

L = 2
x = np.arange(-L, L + 0.1, 0.1)
n = len(x)
tol = 1e-6

Esol_pos = np.zeros(2)
Esol_neg = np.zeros(2)
ysol_pos = np.zeros((n, 2))
ysol_neg = np.zeros((n, 2))

for gamma in [0.05, -0.05]:
    E0 = 0.1 
    A = 1e-6  
    for j_modes in range(2):  
        dA = 0.01  
        for ij in range(100): 
            E, dE = E0, 0.2 
            for j in range(100):  
                y0 = [A, np.sqrt(L**2 - E) * A]
                sol = solve_ivp(lambda x, y: hw3_rhs_c(x, y, E, gamma), [x[0], x[-1]], y0, t_eval=x)
                ys = sol.y.T
                xs = sol.t

                #norm = np.trapz(ys[:, 0]**2, xs)
                bc = ys[-1, 1] + np.sqrt(L**2 - E) * ys[-1, 0]

                if abs(bc) < tol:
                    break
                if (-1)**j_modes * bc > tol:
                    E += dE
                else:
                    E -= dE
                    dE /= 2     
            area = simpson(ys[:, 0]**2, x = xs)
            if abs(area - 1) < tol:
                break
            if area < 1:
                A += dA
            else:
                A -= dA / 2
                dA /= 2
        E0 = E + 0.2  

        if gamma > 0:
            Esol_pos[j_modes] = E
            ysol_pos[:, j_modes] = np.abs(ys[:, 0])
        else:
            Esol_neg[j_modes] = E
            ysol_neg[:, j_modes] = np.abs(ys[:, 0])

A5 = ysol_pos  # Eigenfunctions for gamma = 0.05
A7 = ysol_neg  # Eigenfunctions for gamma = -0.05
A6 = Esol_pos  # Eigenvalues for gamma = 0.05
A8 = Esol_neg  # Eigenvalues for gamma = -0.05


# In[4]:


# Part d - Simplified and Combined Version
def hw3_rhs_a(x, y, E):
    return [y[1], (x**2 - E) * y[0]]

L = 2
x_span = (-L, L)
E = 1.0
y0 = [1, np.sqrt(L**2 - E)]  
tols = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]  
methods = ['RK45', 'RK23', 'Radau', 'BDF'] 

step_sizes = []

for tol in tols:
    options = {'rtol': tol, 'atol': tol} 
    
    current_step_sizes = []
    for method in methods:
        sol = solve_ivp(hw3_rhs_a, x_span, y0, method=method, args=(E,), **options)
        
        avg_step = np.mean(np.diff(sol.t))
        current_step_sizes.append(avg_step)
    
    step_sizes.append(current_step_sizes)

step_sizes = np.array(step_sizes)

slopes = []

for i, method in enumerate(methods):
    log_steps = np.log(step_sizes[:, i])
    log_tols = np.log(tols)
    
    slope, intercept = np.polyfit(log_steps, log_tols, 1)
    slopes.append(slope)
    plt.plot(step_sizes[:, i], tols, marker='o', label=f'{method} (Slope: {slope:.2f})')

plt.xscale('log')  
plt.yscale('log')  
plt.xlabel('Average Step Size (log scale)')
plt.ylabel('Tolerance (log scale)')
plt.title('Convergence Study of ODE Solvers')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()

A9 = slopes
print(A9)


# In[6]:



def factorial(n):
    x = 1
    for i in range(1, n + 1):
        x *= i
    return x
L = 4
x=  np.linspace(-L, L,81)

# Initialize h array
h = np.array([np.ones_like(x), 2*x, 4*x**2 - 2, 8*x**3 - 12*x, 16*x**4 - 48*x**2 + 12])
# Initialize phi
phi = np.zeros((len(x), 5))
for j in range(5):
    phi[:, j] = (np.exp(-x**2/2) * h[j, :] / np.sqrt(2**j * np.sqrt(np.pi) * factorial(j))).T

eigfun_a = np.zeros(5)
eigfun_b = np.zeros(5)
eigval_a = np.zeros(5)
eigval_b = np.zeros(5)

for j in range(5):
    eigfun_a[j] = np.trapz(((np.abs(A1[:, j]) - np.abs(phi[:, j]))**2),x)
    eigfun_b[j] = np.trapz(((np.abs(A3[:, j]) - np.abs(phi[:, j]))**2),x)
    eigval_a[j] = 100 * np.abs(A2[j] - (2 * j + 1)) / (2 * j + 1)
    eigval_b[j] = 100 * np.abs(A4[j] - (2 * j + 1)) / (2 * j + 1)

A10 = eigfun_a
A12 = eigfun_b
A11 = eigval_a
A13 = eigval_b


# In[ ]:




