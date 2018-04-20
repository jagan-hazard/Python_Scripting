# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 11:59:13 2018

@author: Jagan
"""
'''Function Realizer'''
'''================'''
#%%
from matplotlib import pyplot  as plt
import numpy as np

poly=int(input("Please select the polinomal value:"))
cons=int(input("Please enter the constant value:"))
coeff=[]
for i in range(1,poly+1):
    x1=int(input("Please enter the co-eff of x to the power of {}:".format(i)))
    coeff.append(x1)
x_axis_set=int(input("Please select the x-axis max value:"))
x_pos=[]
x_neg=[]
x_neg_sort=[]
x_all=[]
for i in range(1,x_axis_set+1):
    x_pos.append(i)
for i in range(1,x_axis_set+1):
    x_neg.append(-i)
for i in range(1,len(x_neg)+1):
    x_neg_sort.append(x_neg[-i])
for i in x_neg_sort:
    x_all.append(i)
for i in x_pos:
    x_all.append(i)
x_out=[]
for i in x_all:
    tmp1=[]
    for j in range(1,poly+1):
        tmp=i**j
        tmp1.append(tmp)
    x_out.append(tmp1)
out=[]
for i in range(0,len(x_all)):
    a=x_out[i]
    temp=np.matmul(a, coeff)
    temp=temp+cons
    print (temp)
    out.append(temp)
plt.plot(x_all,out)
plt.show()


