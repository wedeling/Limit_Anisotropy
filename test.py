#compute spectral filter
def get_P(cutoff):

    P = np.ones([N, np.int(N/2+1)])

    for i in range(N):
        for j in range(np.int(N/2+1)):

            if np.abs(kx[i, j]) > cutoff or np.abs(ky[i, j]) > cutoff:
                P[i, j] = 0.0

    return P

#compute spectral filter
def get_P_full(cutoff):

    P = np.ones([N, N])

    for i in range(N):
        for j in range(N):

            if np.abs(kx_full[i, j]) > cutoff or np.abs(ky_full[i, j]) > cutoff:
                P[i, j] = 0.0

    return P

#compute the energy and enstrophy at t_n
def compute_E_and_Z(w_hat_n, verbose=True):

    psi_hat_n = w_hat_n/k_squared_no_zero
    psi_hat_n[0,0] = 0.0
    psi_n = np.fft.irfft2(psi_hat_n)
    w_n = np.fft.irfft2(w_hat_n)

    e_n = -0.5*psi_n*w_n
    z_n = 0.5*w_n**2

    E = simps(simps(e_n, axis), axis)/(2*np.pi)**2
    Z = simps(simps(z_n, axis), axis)/(2*np.pi)**2

    if verbose:
        print('Energy = ', E, ', enstrophy = ', Z)
    return E, Z

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

plt.close('all')
plt.rcParams['image.cmap'] = 'seismic'

#number of gridpoints in 1D
I = 7
N = 2**I

#2D grid
h = 2*np.pi/N
axis = h*np.arange(1, N+1)
axis = np.linspace(0.0, 2.0*np.pi, N)
[x , y] = np.meshgrid(axis , axis)

#frequencies
k = np.fft.fftfreq(N)*N

kx = np.zeros([N, np.int(N/2+1)]) + 0.0j
ky = np.zeros([N, np.int(N/2+1)]) + 0.0j

for i in range(N):
    for j in range(np.int(N/2+1)):
        kx[i, j] = 1j*k[j]
        ky[i, j] = 1j*k[i]
        
kx_full = np.zeros([N, N]) + 0.0j
ky_full = np.zeros([N, N]) + 0.0j

for i in range(N):
    for j in range(N):
        kx_full[i, j] = 1j*k[j]
        ky_full[i, j] = 1j*k[i]

k_squared = kx**2 + ky**2
k_squared_no_zero = np.copy(k_squared)
k_squared_no_zero[0,0] = 1.0

#cutoff in pseudospectral method
Ncutoff = N/3
Ncutoff_LF = 2**(I-1)/3

#spectral filter
P = get_P(Ncutoff)
P_LF = get_P(Ncutoff_LF)
P_U = P - P_LF

#spectral filter
P = get_P(Ncutoff)

P_full = get_P_full(Ncutoff)

#initial condition
w = np.sin(4.0*x)*np.sin(4.0*y)*np.exp(-x) + 0.4*np.cos(3.0*x)*np.cos(3.0*y) + \
    0.3*np.cos(5.0*x)*np.cos(5.0*y) + 0.02*np.sin(x) + 0.02*np.cos(y)

w_squared = w**2
w_squared_hat = P_full*np.fft.fft2(w_squared)
print(0.5*np.sum(w_squared_hat**2)/N**4)

w_hat = P_LF*np.fft.rfft2(w)

compute_E_and_Z(w_hat)

#map
shift = np.zeros(N).astype('int')
for i in range(1,N):
    shift[i] = np.int(N-i)

I = range(N);J = range(np.int(N/2+1))

map_I, map_J = np.meshgrid(shift[I], shift[J])
I, J = np.meshgrid(I, J)

w_hat2 = np.zeros([N, N]) + 0.0j
w_hat2[0:N, 0:np.int(N/2+1)] = w_hat
w_hat2[map_I, map_J] = np.conjugate(w_hat[I, J])

print(np.sum(0.5*w_hat2*np.conjugate(w_hat2))/(N)**4)


plt.show()