import numpy as np
import matplotlib.pyplot as plt


theta=np.arange(0,7,0.0001)
x=np.cos(theta)
y=np.sin(theta)

plt.figure(figsize=(15,15),dpi=96)


plt.clf()

kappa=2.2
R=1.42322
plt.plot(R+x,kappa*y,label='NSTX')

kappa=1.6333
R=2.877
plt.plot(R+x,kappa*y,label='DIII-D')
plt.xlabel('R/r',fontsize=15)
plt.ylabel('kappa',fontsize=15)
plt.xlim(0,5)
plt.ylim(-2.5,2.5)
plt.legend()
plt.show()
