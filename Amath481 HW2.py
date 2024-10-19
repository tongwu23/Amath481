#!/usr/bin/env python
# coding: utf-8

# In[10]:


# Load packages
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define constants
L = 4
K = 1
xspan = np.linspace(-L, L, 81)  

def bvp(y, x, epsilon):
    return [y[1], (K * x**2 - epsilon) * y[0]]

# Initial conditions and boundary condition check
tol = 1e-4  
col = ['r', 'b', 'g', 'c', 'm', 'k']  

A1 = [] #eigenfunctions 
A2 = [] #eigen

epsilon_start = 0.1  
for modes in range(1, 6):  # Loop over the first 5 modes
    epsilon = epsilon_start 
    depsilon = 0.2  

    for _ in range(1000):  # Iterative loop for adjusting epsilon
        y0 = [1, np.sqrt(L**2 - epsilon)]
        # initial condition -> equation = 0 y0 = 0 phi2prime and phi = 0 
        # 1 comes from y1
        # Solve the system using odeint
        sol = odeint(bvp, y0, xspan, args=(epsilon,))

        if abs(sol[-1, 1] + np.sqrt(L**2 - epsilon) * sol[-1, 0]) < tol:
            A2.append(epsilon)  # Store the converged eigenvalue
            break  

        if (-1) ** (modes + 1) * (sol[-1, 1] + np.sqrt(L**2 - epsilon) * sol[-1, 0])> 0:
            epsilon += depsilon
        else:
            epsilon -= depsilon / 2
            depsilon /= 2

    epsilon_start = epsilon + 0.1  
    
    norm = np.trapz(sol[:, 0] ** 2, xspan)  # Calculate the normalization factor
    normalized_phi = abs(sol[:, 0] / np.sqrt(norm))  # Normalize the eigenfunction
    A1.append(normalized_phi)  # Store the normalized eigenfunction

   
    plt.plot(xspan, normalized_phi, col[modes - 1], label=f"Mode {modes}")

A1 = np.array(A1).T

# Plot configuration
plt.xlabel('x')
plt.ylabel('Normalized $\phi_n(x)$')
plt.title('First 5 Eigenfunctions of the Quantum Harmonic Oscillator')
plt.legend()
plt.show()


# In[11]:


A1


# In[12]:


A2

