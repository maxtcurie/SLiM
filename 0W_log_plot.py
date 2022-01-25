import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data=pd.read_csv('W_auto.log',header=None)

w_list=[]
for i in data[1]:
    if i[:len('(nan')]!='(nan':
        w_list.append(eval(i))

plt.clf()
plt.scatter(np.real(w_list),np.imag(w_list))
plt.xlabel('omega')
plt.ylabel('gamma')
plt.ylim(0,100)
plt.show()

