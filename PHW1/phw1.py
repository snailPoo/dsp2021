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


# X in run.py
print(mnist.keys())
print(mnist['data'].shape)
print(mnist['target'].shape)


# In[4]:


def plot(picArr, l, title):
    plt.subplot(191+l)
    plt.imshow(picArr.reshape(28, 28), 'gray')
    plt.title(title)
    plt.axis('off')
def findMax(j, i, Max, r):
    for k in range(j): # check 
        if(i == idx[k]):
            return Max # continue
    tmp = abs(A[i] @ r)
    if tmp > Max:
        Max = tmp;
        idx[j] = i
    return Max


# In[5]:


# X in run.py
fig = plt.figure(figsize = (15, 3))
fig.patch.set_facecolor('white')
for l in range(9):
    plot(mnist['data'][l], l, mnist['target'][l])
#     plt.subplot(191+l)
#     plt.imshow(mnist['data'][l].reshape(28, 28), 'gray')
#     plt.title(mnist['target'][l])
#     plt.axis('off')


# In[6]:


def plot(picArr, l, title):
    plt.subplot(191+l)
    plt.imshow(picArr.reshape(28, 28), 'gray')
    plt.title(title)
    plt.axis('off')


# ### Part 1: PCA

# In[7]:


# Q1
# mean = np.zeros(784)
# for i in range(70000):
#     mean += mnist.data[i]
# for i in range(784):
#     mean[i] /= 70000
# plt.imshow(mean.reshape(28, 28), 'gray')
plt.imshow(np.mean(mnist.data, axis=0).reshape(28, 28), 'gray')


# In[8]:


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


# In[9]:


# X in run.py
U5, sigma5, VT5 = np.linalg.svd(subset5)
fig = plt.figure(figsize = (15, 3))
fig.patch.set_facecolor('white')
for l in range(3):
    plot(VT5[l], l, "λ = %.2f" % np.power(sigma5[l], 2))


# In[10]:


# X in run.py
from sklearn.decomposition import PCA
pca=PCA(n_components=100)
pca.fit_transform(Center5)
fig = plt.figure(figsize = (15, 3))
fig.patch.set_facecolor('white')
for l in range(3):
    plot(pca.components_[l], l, "λ = %.2f" % pca.explained_variance_[l])


# In[11]:


#Q3
fig = plt.figure(figsize = (15, 3))
fig.patch.set_facecolor('white')

s = np.zeros(784)
plot(subset5[0], 0, 'Original')

for i in range(100):
    c = eigenvector5.T[i] @ subset5[0] #VT5[i], pca.components_[i]
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


# In[12]:


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
    c1[i] = eigenvector136.T[0] @ subset136[i]
    c2[i] = eigenvector136.T[1] @ subset136[i]
    
plt.scatter(c1, c2, c = tar)
plt.savefig('./fig/Q4.jpg')


# ### Part 2: OMP

# In[13]:


# Part 2: OMP
# Q5
train = mnist.data[0:10000]
test = mnist.data[10000]

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
# reconstruct = B[0:6].T @ c
# plot(reconstruct.T, 5, 'L-2 = %.2f'%np.linalg.norm(r))
# print(idx)


# In[14]:


# X in run.py
from sklearn.linear_model import OrthogonalMatchingPursuit
omp = OrthogonalMatchingPursuit(n_nonzero_coefs=5)
omp.fit(train, mnist['target'][0:10000])
coef = omp.coef_
idx_r, = coef.nonzero()
fig = plt.figure(figsize = (15, 3))
fig.patch.set_facecolor('white')
for l in range(5):
    plt.subplot(191+l)
    plt.imshow(train[idx_r[l]].reshape(28, 28), 'gray')
    plt.title(idx_r[l])
    plt.axis('off')


# In[15]:


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


# ### Part 3: lasso

# In[16]:


# Part 3: lasso
# Q7
idx8 = (mnist['target'] == '8')
subset8 = mnist.data[idx8]
test = subset8[6824]

fig = plt.figure(figsize = (15, 3))
fig.patch.set_facecolor('white')
plot(test, 0, 'Original')

#PCA
mean8 = np.mean(subset8, axis=0)
Center8 = subset8 - mean8
XXT8 = Center8.T @ Center8 / (Center8.shape[0]-1)
eigenvalue8, eigenvector8 = np.linalg.eig(XXT8)
eigenvalue8 = np.real_if_close(eigenvalue8, tol=1)
eigenvector8 = np.real_if_close(eigenvector8, tol=1)
s = np.zeros(784)
for i in range(5):
    c = eigenvector8.T[i] @ test
    s += eigenvector8.T[i]*c
plot(s, 1, 'PCA')

#OMP
A = subset8 / np.linalg.norm(subset8, axis = 1)[:, np.newaxis]
idx = [-1] * 5
B = np.zeros((5, 784))
r = test.T
for j in range(5):
    Max =  -100000
    for i in range(subset8.shape[0]):
        Max = findMax(j, i, Max, r)
    B[j] = A[idx[j]]
    c = np.linalg.pinv(B[0:j+1].T) @ test.T
    reconstruct = B[0:j+1].T @ c
    r = test.T - reconstruct
plot(reconstruct.T, 2, 'OMP')

#Lasso
from sklearn.linear_model import Lasso

def Max5():
    Max = [-1] * 5
    idx = [-1] * 5
    for i in range(6824):
        for j in range(5):
            if lasso.coef_[i] > Max[j]:
                for k in range(4,j,-1):
                    Max[k] = Max[k-1]
                    idx[k] = idx[k-1]
                Max[j] = lasso.coef_[i]
                idx[j] = i
                break;
    return idx
def reconstruct():
    s = np.zeros(784)
    for i in range(5):
        c = subset8[:6824][idx[i]] @ test
        s += subset8[:6824][idx[i]]*c
    return s

nor = Center8 / np.linalg.norm(Center8, axis = 1)[:, np.newaxis]
lasso = Lasso(alpha=1)
lasso.fit(nor[:6824].T, test)
idx = Max5()
s = reconstruct()
plot(s, 3, 'Lasso')
# plt.savefig('./fig/Q7.jpg')


# In[17]:


fig = plt.figure(figsize = (15, 3))
fig.patch.set_facecolor('white')
plot(test, 0, 'Original')

S = np.zeros((2,30))
sep = np.linspace(0.1, 3, num=30)
for i in range(30):
    lasso = Lasso(alpha=sep[i])
    lasso.fit(nor[:6824].T, test)
    idx = Max5()
    s = reconstruct()
    s = ((s - s.min()) * (1/(s.max() - s.min()) * 255)).astype('uint8')
    S[0][i] = np.linalg.norm(test - s)
    S[1][i] = sum(lasso.coef_!=0)
    if i % 4 == 0:
        plot(s, int(i/4+1), '%.1f:%.2f'%(sep[i],S[0][i]))
# plt.savefig('./fig/Q8.jpg')


# In[18]:


plt.plot(sep, S[0])
# plt.savefig('./fig/Q9.jpg')


# In[19]:


plt.plot(sep, S[1])
# plt.savefig('./fig/Q10.jpg')





