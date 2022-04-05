#!/usr/bin/env python
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math

def return_r_p(
  MON = 960
  A = 0.5
  u = 60
  B = 2000
  
  my_m = np.ones(MON)
  my_p = np.ones(MON)
  my_r = np.ones(MON)
  
  for i in range(MON):
    my_m[i] = float(i)/10
    #my_y2[i] = math.log(my_x2[i], 100)
    my_p[i] = math.cos((my_m[i]-6)/6*math.pi)*math.log(my_m[i]+1, 100)*12+60
    my_r[i] = A*math.exp(-(my_p[i]-u)**2/B)
  
  
  #print(my_x)
  #print(my_y)
 
if __name__ == '__main__':

 fig = plt.figure(figsize=(10, 10), dpi=80)
  plt.plot(my_m, my_r)
  plt.plot(my_m, my_p/240+0.25)
  plt.show()



