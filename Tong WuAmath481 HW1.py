#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Question 1 Part 1
import numpy as np
x = np.array([-1.6])
for j in range (1000):
    fx = x[j]*np.sin(3*x[j])-np.exp(x[j])
    f_prime = np.sin(3*x[j])+3*x[j]*np.cos(3*x[j])-np.exp(x[j])
    next_x = x[j] - (fx/f_prime)
    x = np.append(x, next_x)
    f_next_x = next_x*np.sin(3 * next_x) - np.exp(next_x) 
    if abs(fx) < 1e-6:
        break
print(x)  
A1 = x
iteration_1 = j+1


# In[4]:


#Question 1 Part 2

def f(x):
    return x* np.sin(3*x)-np.exp(x)

a = -0.7
b = -0.4

midpt = np.array([])

for n in range (1000):
    mid = (a+b)/2
    midpt = np.append(midpt,mid)
    f_mid = f(mid)

    if  f_mid  > 0 :
        a = mid
    else:
        b = mid
    if abs(f_mid) < 1e-6:
        break
print("Midpoints:", midpt)
A2 = midpt
iteration_2 = n+1
A3 = np.array([iteration_1,iteration_2 ])
print(A3)


# In[3]:


#Question 2 
A = np.array([[1,2],[-1,1]])
B = np.array([[2, 0], [0, 2]])
C = np.array([[2, 0, -3], [0, 0, -1]])
D = np.array([[1, 2], [2, 3], [-1, 0]])
x = np.array([1, 0])
y = np.array([0, 1])
z = np.array([1, 2, -1])

A4 = A + B #a
print(A4)
A5 = 3*x - 4*y #b
print(A5)
A6 = np.dot(A,x) #c
print(A6)
a7 = x-y 
A7 = np.dot(B,a7)#d
print(A7)
A8 = np.dot(D,x)
print(A8)
A9 = np.dot(D,y)+z #f
print(A9)
A10 = np.dot(A,B)
print(A10)
A11 = np.dot(B,C)
print(A11)
A12 = np.dot(C,D)
print(A12)


# In[ ]:




