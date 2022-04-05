#!/usr/bin/env python
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
w1 = 1               # weights for interaction
w2 = 0               # weights for percipitation


A = 0.5
u = 60
B = 200
N = 50                 # The grid number
G = 100                # How many years you want to predict
dec_others = 0.5     # The impact factor by others 
inc_global = 0.5      # The global factor: nutrient from soil



def return_r_p(r):
  # my_m = float(r)/10
  my_m = r
  my_p = math.cos((my_m-6)/6*math.pi)*math.log(my_m+1, 100)*12+60
  my_r = A*math.exp(-(my_p-u)**2/B)
 
  return my_p, my_r 
  
  #print(my_x)
  #print(my_y)

def distance(x1,y1,x2, y2):
  dist = np.sqrt((x1-x2)**2+(y1-y2)**2);
  return dist

def return_radius(x, y, r, row, col, dec_others, inc_global):
    sum = 0;
    dim_in = x.shape

    #for i in range(dim_in[0]):
      #for j in range(dim_in[1]):
    v = 1
    if row - v < 0: i_min = 0
    else          : i_min = row-v
    if row + v > dim_in[0]: i_max = dim_in[0] 
    else                  : i_max = row+v
    if col - v < 0: j_min = 0
    else          : j_min = col-v
    if col + v > dim_in[1]: j_max = dim_in[1]
    else                  : j_max = col+v

    for i in range(i_min, i_max):
      for j in range(j_min, j_max):
            dist = distance(row, col, x[i,j], y[i,j]);
            if dist != 0:
                sum = sum + (1/dist)*r[i][j]

    tmp =r[row][col] - sum * dec_others + inc_global
    if tmp < 0:
        return 0
    else:
        return tmp



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

def raw_pic_well(x, y, r, num, wells):
  fig = plt.figure(figsize=(10, 10), dpi=80)
  ax = fig.add_subplot(111)
  for i in range(N):
    for j in range(N):
      shape = matplotlib.patches.Circle((x[i][j], y[i][j]), r[i][j], edgecolor='blue', fill=False)
      ax.add_patch(shape)
  
  shape = matplotlib.patches.Circle((wells[0][0], wells[0][1]), 0.5, color='red')
  ax.add_patch(shape)
  shape = matplotlib.patches.Circle((wells[1][0], wells[1][1]), 0.5, color='red')
  ax.add_patch(shape)
  shape = matplotlib.patches.Circle((wells[2][0], wells[2][1]), 0.5, color='red')
  ax.add_patch(shape)

  plt.xlim([-1, N])
  plt.ylim([-1, N])
  plt.savefig("./output/p"+str(num)+".png", bbox_inches='tight')
  # plt.show()


if __name__ == '__main__':
  MON = 9600
 
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



  np.random.seed(123)


  # water wells coordinates
  wells=np.array([[5, 5],
                [20, 40],
                [40, 10]])


  x = np.random.rand(N,N)*N
  x = x.astype(int)
  y = np.random.rand(N,N)*N
  y = y.astype(int)
  r = np.random.rand(N,N)
  
  for i in range(N):
    for j in range(N):
      x[i, j] = i;
      y[i, j] = j;


  #for month in range(1, MON, 10):
  #  for i in range(N):
  #    for j in range(N):
  #      r[i][j] = my_r[month]
  #  raw_pic(x, y, r, int(month/10))
  #  print(month/10, r[0][0])

  r_max = 0
  r_min = 1000
  for i in range(N):
    for j in range(N):

      # calculate the distance to the closest well
      wells_dim = wells.shape
      dist_min = distance(wells[0][0],wells[0][1], i, j)
      for k in range(wells_dim[0]):
        dist = distance(wells[k][0], wells[k][1], i, j)
        if dist < dist_min: dist_min = dist
    
      #if dist_min != 0:
      r[i, j] = dist_min
      if r_max < r[i,j]:
        r_max = r[i, j]
        
      if r_min > r[i, j]:
        r_min = r[i, j]

  for i in range(N):
    for j in range(N):
      r[i, j] = (r_max-r[i, j])/r_max*0.5

  
  print ("r_max=", r_max)
  print ("r_min=", r_min)


  raw_pic_well(x, y, r, "_intial", wells)

  for num in range(200):
    for i in range(N):
      for j in range(N):
        r[i][j] = w1*return_radius(x, y, r, i, j, dec_others, inc_global) + w2*my_r[num*10]
        if r[i][j] > 0.7: r[i][j] = 0.7
  
    print (num)
    #if(num%10 == 0):
    raw_pic_well(x, y, r, 'p_'+str(num), wells)


    
