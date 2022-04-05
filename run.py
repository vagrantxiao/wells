#!/usr/bin/env python
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math

A = 0.5
u = 60
B = 200
 
def return_r_p(r):
  # my_m = float(r)/10
  my_m = r
  my_p = math.cos((my_m-6)/6*math.pi)*math.log(my_m+1, 100)*12+60
  my_r = A*math.exp(-(my_p-u)**2/B)
 
  return my_p, my_r 
  
  #print(my_x)
  #print(my_y)

def raw_pic(x, y, r, num):
  fig = plt.figure(figsize=(10, 10), dpi=80)
  ax = fig.add_subplot(111)
  for i in range(N):
    for j in range(N):
      shape = matplotlib.patches.Circle((x[i][j], y[i][j]), r[i][j], edgecolor='blue', fill=False)
      ax.add_patch(shape)
  
  plt.xlim([-1, N])
  plt.ylim([-1, N])
  plt.savefig("./output/p"+str(num)+".png", bbox_inches='tight')
  # plt.show()

if __name__ == '__main__':
  MON = 960
 
  my_m = np.ones(MON)
  my_p = np.ones(MON)
  my_r = np.ones(MON)
  
  for i in range(MON):
    my_m[i] = float(i)/10
    my_p[i], my_r[i] = return_r_p(my_m[i])
    
 
  #fig = plt.figure(figsize=(10, 10), dpi=80)
  #ax = fig.add_subplot(121)

  #plt.plot(my_m, my_r)
  #plt.plot(my_m, my_p/240+0.25)

  #ax = fig.add_subplot(122)
  #plt.plot(my_m, my_p)
  #plt.savefig("my_fig.png", bbox_inches='tight')
  #plt.show()

  N = 50                 # The grid number
  G = 100                # How many years you want to predict
  dec_others = 0.005     # The impact factor by others 
  inc_global = 0.1       # The global factor: precipitation 

  np.random.seed(123)


  # water wells coordinates
  wells=np.array([[15, 40],
                [23, 30],
                [22, 34]])


  x = np.random.rand(N,N)*N
  x = x.astype(int)
  y = np.random.rand(N,N)*N
  y = y.astype(int)
  r = np.random.rand(N,N)
  
  for i in range(N):
    for j in range(N):
      x[i, j] = i;
      y[i, j] = j;

  for month in range(1, MON, 10):
    for i in range(N):
      for j in range(N):
        r[i][j] = my_r[month]
    

    raw_pic(x, y, r, int(month/10))
    print(month/10, r[0][0])