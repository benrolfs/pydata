'''
Created on Mar 28, 2013

@author: ben

Examples from chapter 4. 
'''

import numpy as np

data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1)

data2 = [[1,2,3,4], 5,6,7,8]
arr2 = np.array(data2)

data3 = [[1,2,3,4], [5,6,7,8]]
arr3 = np.array(data3)

arr = np.array([1,2,3,4,5])
float_arr = arr.astype(np.float64)
comp_arr = arr.astype(np.complex64)

#let's see how array multiplication works in numpy

arr = np.array([[1,2,3], [4,5,6]], dtype = float)
arr_mult = arr*arr # Hadamard/elementwise product

# Explore some indices/slices

arr = np.arange(10)
arr[5]

for i in range(10):
    print(arr[i])

for i in range(10):
    print(arr[-i])
    
for i in range(11):
    print(arr[0:i])
    
for i in range(11):
    print(arr[-0:i])
    
for i in range(11):
    print(arr[0:-i])
    
# Very important to remember that arrays are pointers

arr_slice = arr[5:8]
arr_slice
arr
arr_slice[1] = 10
arr[:]

# n-d arrays
arr2d = np.array([[1,2,3], [4,5,6], [7,8,9]])
arr3d = np.array([[[1,2,3], [4,5,6]], [[7,8,9], [10,11,12]]])

# demo copying on 3d array
old_values = arr3d[0].copy()
arr3d[0] = 42
arr3d[0] = old_values

# Boolean indices
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
from numpy.matlib import randn
data = randn(7,4)
bob_index = names == 'Bob'
names[bob_index]
data[bob_index]
data[bob_index, :2]
data[bob_index, :10]
data[:, bob_index]
not_bob = names != 'Bob'
data[not_bob]
data[bob_index]
data[~bob_index]
data[~not_bob]

data[data < 0] = 0
data[names != 'Joe'] = 7

# reshape and transpose operators
arr = np.arange(32).reshape((8,4))
arr
arr.T
arr.reshape(4,8)

arr = np.random.randn(6,3)
np.dot(arr, arr.T)
np.dot(arr.T, arr)

# unfuncs, bfuncs
arr = randn(7) * 5
np.modf(arr)

np.multiply(arr,arr)

# vectorization
points =  np.arange(-5,5, 0.01)
xs, ys, = np.meshgrid(points,points)
import matplotlib.pyplot as plt
z = np.sqrt(xs **2 + ys **2)
plt.imshow(z); plt.colorbar()
plt.show()

# mean and sd
arr = np.random.randn(5,4)
arr.mean()
np.mean(arr)
arr.sum()
np.sum(arr)
arr.mean(axis=1)
arr.mean(axis=0)

# cumsum cumprod over arrays
arr = np.array([[1,2,3], [4,5,6], [7,8,9]])
arr.cumsum(0)
arr.cumsum(1)
arr.cumprod(0)
arr.cumprod(1)
arr.reshape(arr.size,).cumprod()

# file input output
arr = np.arange(10)
np.save('some_array', arr)
some_array = np.load('some_array.npy')
np.savez('array_archive.npz', a=arr, b=data)
arch = np.load('array_archive.npz')
arch_arr = arch['a']
arch_data = arch['b']

# some simple linalg
x = np.array([[1,2,3], [4,5,6]], dtype=float)
y = np.array([[6,23], [-1,7], [8,9]], dtype=float)
x.dot(y) 
np.dot(x, np.ones(3))

from numpy.linalg import inv, qr
X = randn(5,5)
mat = X.T.dot(X) #X^T X
mat_inv = inv(mat)
np.dot(mat, mat_inv)
np.dot(mat, inv(mat))

q,r = qr(mat)

# example random walk
#from random import normalvariate
#N = 1000000
#%timeit samples = [normalvariate(0,1) for _ in xrange(N)]
#%timeit np.random.normal(size = N)

import random
position = 0
walk = [position]
steps = 1000
for i in xrange(steps):
    step = 1 if random.randint(0, 1) else -1
    position += step
    walk.append(position)
    
plt.plot(walk[:100])
plt.show()

nwalks = 5000
nsteps = 1000
draws = np.random.randint(0,2, size=(nwalks,nsteps))
steps = np.where(draws > 0, 1, -1)
walks = steps.cumsum(1)
walksum = walks.sum(0) 
