#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
from sklearn.datasets import fetch_openml 
import matplotlib.pyplot as plt


# In[2]:


mnist = fetch_openml('mnist_784')
mnist.data = np.asarray(mnist['data'], dtype=np.float64)
os.mkdir('fig') 


# In[3]:


# Part 1: PCA
# Q1
# plt.imshow(np.mean(mnist.data, axis=0).reshape(28, 28), 'gray')


# In[4]:


def plot(picArr, l, title):
    plt.subplot(191+l)
    plt.imshow(picArr.reshape(28, 28), 'gray')
    plt.title(title)
    plt.axis('off')


# In[5]:


# Q2
idx5 = (mnist['target'] == '5')
subset5 = mnist.data[idx5]
mean5 = np.mean(subset5, axis=0)
Center5 = subset5 - mean5

XXT5 = Center5.T @ Center5 / (Center5.shape[0]-1)
eigenvalue5, eigenvector5 = np.linalg.eig(XXT5)
eigenvalue5 = np.real_if_close(eigenvalue5, tol=1)
eigenvector5 = np.real_if_close(eigenvector5, tol=1)

fig = plt.figure(figsize = (15, 3))
fig.patch.set_facecolor('white')
for l in range(3):
    plot(eigenvector5.T[l], l, "λ = %.2f" % eigenvalue5[l])
plt.savefig('./fig/Q2.jpg')
plt.close()

# In[6]:


#Q3
fig = plt.figure(figsize = (15, 3))
fig.patch.set_facecolor('white')

s = np.zeros(784)
plot(subset5[0], 0, 'Original')
# plt.imshow(subset5[0].reshape(28, 28), 'gray')

for i in range(100):
    c = eigenvector5.T[i] @ subset5[0]
    s += eigenvector5.T[i]*c
    if i == 2:
        plot(s, 1, '3d')
    elif i == 9:
        plot(s, 2, '10d')
    elif i == 29:
        plot(s, 3, '30d')
    elif i == 99:
        plot(s, 4, '100d')
plt.savefig('./fig/Q3.jpg')
plt.close()

# In[7]:


#Q4
tar = mnist['target'][0:10000]
idx136 = (tar == '1') | (tar == '3') | (tar == '6') 
tar = tar[idx136].astype(int)
subset136 = mnist.data[0:10000][idx136]
Center136 = subset136 - np.mean(subset136, axis = 0)

XXT136 = Center136.T @ Center136 / (Center136.shape[0]-1)
eigenvalue136, eigenvector136 = np.linalg.eig(XXT136)
eigenvalue136 = np.real_if_close(eigenvalue136, tol=1)
eigenvector136 = np.real_if_close(eigenvector136, tol=1)

c1 = np.zeros(subset136.shape[0])
c2 = np.zeros(subset136.shape[0])
for i in range(subset136.shape[0]):
    c1[i] = eigenvector136.T[0] @ Center136[i]
    c2[i] = eigenvector136.T[1] @ Center136[i]
    
plt.scatter(c1, c2, c = tar)
plt.savefig('./fig/Q4.jpg')
plt.close()

# In[8]:


# Part 2: OMP
# Q5
train = mnist.data[0:10000]
test = mnist.data[10000]

def findMax(j, i, Max, r):
    for k in range(j): # check 
        if(i == idx[k]):
            return Max # continue
    tmp = abs(A[i] @ r)
    if tmp > Max:
        Max = tmp;
        idx[j] = i
    return Max

A = train / np.linalg.norm(train, axis = 1)[:, np.newaxis] #normalize
idx = [-1, -1, -1, -1, -1] # basis index
B = np.zeros((5, 784)) # basis 
r = test.T # residual
for j in range(5): # find 5 basis
    Max =  -100000
    for i in range(10000): # find max inner product
        Max = findMax(j, i, Max, r)
    B[j] = A[idx[j]]
    c = np.linalg.pinv(B[0:j+1].T) @ test.T #np.linalg.pinv:pseudo inverse
    r = test.T - B[0:j+1].T @ c

fig = plt.figure(figsize = (15, 3))
fig.patch.set_facecolor('white')
for l in range(5):
    plot(A[idx[l]], l, '')
plt.savefig('./fig/Q5.jpg')
plt.close()

# In[9]:


#Q6
test = mnist.data[10001]
A = train / np.linalg.norm(train, axis = 1)[:, np.newaxis] #normalize
idx = [-1] * 200
B = np.zeros((200, 784))
r = test.T
fig = plt.figure(figsize = (15, 3))
fig.patch.set_facecolor('white')
plot(test, 0, 'L-2 = 0')
t = 1
for j in range(200):
    Max =  -100000
    for i in range(10000):
        Max = findMax(j, i, Max, r)
    B[j] = A[idx[j]]
    c = np.linalg.pinv(B[0:j+1].T) @ test.T
    reconstruct = B[0:j+1].T @ c
    r = test.T - reconstruct
    if j == 4 or j == 9 or j == 39 or j == 199:
        plot(reconstruct.T, t, 'L-2 = %.2f'%np.linalg.norm(r))
        t += 1
plt.savefig('./fig/Q6.jpg')
plt.close()