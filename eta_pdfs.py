import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

plt.close('all')

plt.style.use('seaborn-deep')

HOME = os.path.abspath(os.path.dirname(__file__))

#time scale
Omega = 7.292*10**-5
day = 24*60**2*Omega

axis = np.linspace(0.0, 2.0*np.pi, N)
[x , y] = np.meshgrid(axis , axis)

#start, end time (in days) + time step
t_end = 300.0*day

store_ID = 'test'

fname = HOME + '/samples/' + store_ID + '_t_' + str(np.around(t_end/day, 1)) + '.hdf5'

print('Loading samples from ', fname)

if os.path.exists(HOME + '/samples') == False:
    os.makedirs(HOME + '/samples')

#create HDF5 file
h5f = h5py.File(fname, 'r')

#store numpy sample arrays as individual datasets in the hdf5 file
for key in h5f.keys():
    print(key)
    vars()[key] = h5f[key][:]
    
h5f.close()    

fig = plt.figure()
plt.hist([Eta1.flatten(), Eta2.flatten()], 30, label=[r'$\eta_1$', r'$\eta_2$'])
leg = plt.legend(loc=0)
plt.yticks([])

fig = plt.figure()
plt.hist([Theta1.flatten() - Theta2.flatten()], 300, label=[r'$\Delta\theta$'])
leg = plt.legend(loc=0)
plt.yticks([])

plt.show()