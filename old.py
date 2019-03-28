"""
*************************
* S U B R O U T I N E S *
*************************
"""

#pseudo-spectral technique to solve for Fourier coefs of Jacobian
def compute_VgradW_hat(w_hat_n, P):
    
    #compute streamfunction
    psi_hat_n = w_hat_n/k_squared_no_zero
    psi_hat_n[0,0] = 0.0
    
    #compute jacobian in physical space
    u_n = np.fft.irfft2(-ky*psi_hat_n)
    w_x_n = np.fft.irfft2(kx*w_hat_n)

    v_n = np.fft.irfft2(kx*psi_hat_n)
    w_y_n = np.fft.irfft2(ky*w_hat_n)
    
    VgradW_n = u_n*w_x_n + v_n*w_y_n
    
    #return to spectral space
    VgradW_hat_n = np.fft.rfft2(VgradW_n)
    
    VgradW_hat_n *= P
    
    return VgradW_hat_n

#compute streamfunction
def get_psi_hat(w_hat):

    psi_hat = w_hat/k_squared_no_zero
    psi_hat[0,0] = 0.0

    return psi_hat

def rst_eigs(w_hat_LF, delta_w_hat):

    #compute streamfunctions
    psi_hat_LF = w_hat_LF/k_squared_no_zero
    psi_hat_LF[0,0] = 0.0
    delta_psi_hat = delta_w_hat/k_squared_no_zero
    delta_psi_hat[0,0] = 0.0
    
    #compute full and projected velocities
    u_LF = np.fft.irfft2(-ky*psi_hat_LF)
    v_LF = np.fft.irfft2(kx*psi_hat_LF)
    delta_u = np.fft.irfft2(-ky*delta_psi_hat)
    delta_v = np.fft.irfft2(kx*delta_psi_hat)

    #compute tensor entries
    c11 = delta_u*delta_u
    c12 = delta_u*delta_v
    c22 = delta_v*delta_v

    c11_hat_LF = P_LF*np.fft.rfft2(c11)
    c12_hat_LF = P_LF*np.fft.rfft2(c12)
    c22_hat_LF = P_LF*np.fft.rfft2(c22)

    c11_LF = np.fft.irfft2(c11_hat_LF)
    c12_LF = np.fft.irfft2(c12_hat_LF)
    c22_LF = np.fft.irfft2(c22_hat_LF)

    #print np.where(c22_LF < 0.0)[0].size/np.double(N**2)
    idx0 = np.where(c11_LF < 0.0)
    c11_LF[idx0[0], idx0[1]] = 0.0
    idx0 = np.where(c22_LF < 0.0)
    c22_LF[idx0[0], idx0[1]] = 0.0

    tke = 0.5*(c11_LF + c22_LF)
    tke = np.sqrt(tke**2)

    dM_2 = 0.5*(c11_LF - c22_LF)
    dN_2 = c12_LF

    C = np.zeros([N**2, 2, 2])
    C[:, 0, 0] = c11_LF.flatten()
    C[:, 0, 1] = c12_LF.flatten()
    C[:, 1, 0] = c12_LF.flatten()
    C[:, 1, 1] = c22_LF.flatten()

    #NOTE: the use of eigh seems to already sort the eigenvalues, whereas eig does not
    lambda_i, V = np.linalg.eigh(C)
    lambda_i = np.sqrt(lambda_i**2)
    #lambda_i = np.sort(lambda_i, axis=1)

    #extract the angle of the eigenvectors, using the components of the 1st vector
    V11 = V[:, 0, 0] 
    V21 = V[:, 1, 0]
    theta = -np.arctan2(V11, V21)

    #idx0 = np.where(lambda_i[:, 0] < 0.0)[0]
    #lambda_i[idx0, 0] = 1e-12
    #idx0 = np.where(lambda_i[:, 1] < 0.0)[0]
    #lambda_i[idx0, 1] = 1e-12

    return lambda_i, theta, tke, dM_2, dN_2, C

def R1_eigs(w_hat_LF, delta_w_hat):

    #compute streamfunctions
    psi_hat_LF = w_hat_LF/k_squared_no_zero
    psi_hat_LF[0,0] = 0.0
    delta_psi_hat = delta_w_hat/k_squared_no_zero
    delta_psi_hat[0,0] = 0.0
    
    #compute full and projected velocities
    u_LF = np.fft.irfft2(-ky*psi_hat_LF)
    v_LF = np.fft.irfft2(kx*psi_hat_LF)
    delta_u = np.fft.irfft2(-ky*delta_psi_hat)
    delta_v = np.fft.irfft2(kx*delta_psi_hat)

    #compute tensor entries
    c11 = u_LF*delta_u + delta_u*u_LF 
    c12 = u_LF*delta_v + delta_u*v_LF 
    c22 = v_LF*delta_v + delta_v*v_LF 

    c11_hat_LF = P_LF*np.fft.rfft2(c11)
    c12_hat_LF = P_LF*np.fft.rfft2(c12)
    c22_hat_LF = P_LF*np.fft.rfft2(c22)

    c11_LF = np.fft.irfft2(c11_hat_LF)
    c12_LF = np.fft.irfft2(c12_hat_LF)
    c22_LF = np.fft.irfft2(c22_hat_LF)
    
    C = np.zeros([N**2, 2, 2])
    C[:, 0, 0] = c11_LF.flatten()
    C[:, 0, 1] = c12_LF.flatten()
    C[:, 1, 0] = c12_LF.flatten()
    C[:, 1, 1] = c22_LF.flatten()

    #NOTE: the use of eigh seems to already sort the eigenvalues, whereas eig does not
    lambda_i, V = np.linalg.eigh(C)
    #lambda_i = np.sort(lambda_i, axis=1)
    
    #extract the angle of the eigenvectors, using the components of the 1st vector
    V11 = V[:, 0, 0] 
    V21 = V[:, 1, 0]
    theta = -np.arctan2(V11, V21)

    return lambda_i, theta

def tensor_eddy_force_hat(w_hat_LF, delta_w_hat, P):

    #compute streamfunctions
    psi_hat_LF = w_hat_LF/k_squared_no_zero
    psi_hat_LF[0,0] = 0.0
    delta_psi_hat = delta_w_hat/k_squared_no_zero
    delta_psi_hat[0,0] = 0.0
    
    #compute full and projected velocities
    u_LF = np.fft.irfft2(-ky*psi_hat_LF)
    v_LF = np.fft.irfft2(kx*psi_hat_LF)
    delta_u = np.fft.irfft2(-ky*delta_psi_hat)
    delta_v = np.fft.irfft2(kx*delta_psi_hat)

    #compute tensor entries
    c11 = u_LF*delta_u + delta_u*u_LF + delta_u*delta_u
    c12 = u_LF*delta_v + delta_u*v_LF + delta_u*delta_v
    c22 = v_LF*delta_v + delta_v*v_LF + delta_v*delta_v

    c11_hat = P*np.fft.rfft2(c11)
    c12_hat = P*np.fft.rfft2(c12)
    c22_hat = P*np.fft.rfft2(c22)
   
    #return eddy forcing 
    return (kx**2 - ky**2)*c12_hat + kx*ky*(c22_hat - c11_hat)

def tensor_eddy_force_hat1(w_hat_LF, delta_w_hat, P):

    #compute streamfunctions
    psi_hat_LF = w_hat_LF/k_squared_no_zero
    psi_hat_LF[0,0] = 0.0
    delta_psi_hat = delta_w_hat/k_squared_no_zero
    delta_psi_hat[0,0] = 0.0
    
    #compute full and projected velocities
    u_LF = np.fft.irfft2(-ky*psi_hat_LF)
    v_LF = np.fft.irfft2(kx*psi_hat_LF)
    delta_u = np.fft.irfft2(-ky*delta_psi_hat)
    delta_v = np.fft.irfft2(kx*delta_psi_hat)

    #compute tensor entries
    c11 = u_LF*delta_u + delta_u*u_LF
    c12 = u_LF*delta_v + delta_u*v_LF
    c22 = v_LF*delta_v + delta_v*v_LF

    c11_hat = P*np.fft.rfft2(c11)
    c12_hat = P*np.fft.rfft2(c12)
    c22_hat = P*np.fft.rfft2(c22)
   
    #return eddy forcing 
    return (kx**2 - ky**2)*c12_hat + kx*ky*(c22_hat - c11_hat)

def tensor_eddy_force_hat2(w_hat_LF, delta_w_hat, P):

    #compute streamfunctions
    psi_hat_LF = w_hat_LF/k_squared_no_zero
    psi_hat_LF[0,0] = 0.0
    delta_psi_hat = delta_w_hat/k_squared_no_zero
    delta_psi_hat[0,0] = 0.0
    
    #compute full and projected velocities
    #u_LF = np.fft.irfft2(-ky*psi_hat_LF)
    #v_LF = np.fft.irfft2(kx*psi_hat_LF)
    delta_u = np.fft.irfft2(-ky*delta_psi_hat)
    delta_v = np.fft.irfft2(kx*delta_psi_hat)

    #compute tensor entries
    c11 = delta_u*delta_u
    c12 = delta_u*delta_v
    c22 = delta_v*delta_v

    c11_hat = P*np.fft.rfft2(c11)
    c12_hat = P*np.fft.rfft2(c12)
    c22_hat = P*np.fft.rfft2(c22)
   
    #return eddy forcing 
    return (kx**2 - ky**2)*c12_hat + kx*ky*(c22_hat - c11_hat)

def tensor_eddy_force_hat3(w_hat_HF, w_hat_LF):
    
    #compute streamfunctions
    psi_hat_HF = w_hat_HF/k_squared_no_zero
    psi_hat_HF[0,0] = 0.0
    psi_hat_LF = w_hat_LF/k_squared_no_zero
    psi_hat_LF[0,0] = 0.0
    
    #compute full and projected velocities
    u_HF = np.fft.irfft2(-ky*psi_hat_HF)
    u_LF = np.fft.irfft2(-ky*psi_hat_LF)
    v_HF = np.fft.irfft2(kx*psi_hat_HF)
    v_LF = np.fft.irfft2(kx*psi_hat_LF)
    
    C1_11 = u_HF*u_HF
    C1_12 = u_HF*v_HF
    C1_22 = v_HF*v_HF

    C2_11 = u_LF*u_LF
    C2_12 = u_LF*v_LF
    C2_22 = v_LF*v_LF
    
    C1_11_hat = P_LF*np.fft.rfft2(C1_11)
    C1_12_hat = P_LF*np.fft.rfft2(C1_12)
    C1_22_hat = P_LF*np.fft.rfft2(C1_22)
    
    C2_11_hat = P_LF*np.fft.rfft2(C2_11)
    C2_12_hat = P_LF*np.fft.rfft2(C2_12)
    C2_22_hat = P_LF*np.fft.rfft2(C2_22)
    
    EF1_hat = (kx**2 - ky**2)*C1_12_hat + kx*ky*(C1_22_hat - C1_11_hat)
    EF2_hat = (kx**2 - ky**2)*C2_12_hat + kx*ky*(C2_22_hat - C2_11_hat)
    
    return EF1_hat - EF2_hat

#pseudo-spectral technique to solve for Fourier coefs of BCD components
def compute_MN_hat(w_hat_HF, w_hat_LF):
    
    #compute streamfunctions
    psi_hat_HF = w_hat_HF/k_squared_no_zero
    psi_hat_HF[0,0] = 0.0
    psi_hat_LF = w_hat_LF/k_squared_no_zero
    psi_hat_LF[0,0] = 0.0
    
    #compute full and projected velocities
    u_HF = np.fft.irfft2(-ky*psi_hat_HF)
    u_LF = np.fft.irfft2(-ky*psi_hat_LF)
    v_HF = np.fft.irfft2(kx*psi_hat_HF)
    v_LF = np.fft.irfft2(kx*psi_hat_LF)
    
    M_HF = 0.5*(u_HF**2 - v_HF**2)
    M_LF = 0.5*(u_LF**2 - v_LF**2)
    N_HF = u_HF*v_HF
    N_LF = u_LF*v_LF
    
    dM_hat = P_LF*np.fft.rfft2(M_HF - M_LF)
    dN_hat = P_LF*np.fft.rfft2(N_HF - N_LF)
    M_LF_hat = P_LF*np.fft.rfft2(M_LF)
    N_LF_hat = P_LF*np.fft.rfft2(N_LF)
    
    return dM_hat, dN_hat, M_LF_hat, N_LF_hat, u_LF, v_LF

#pseudo-spectral technique to solve for Fourier coefs of RST components
def compute_rst_hat(w_hat):
    
    #compute streamfunction
    psi_hat = w_hat/k_squared_no_zero
    psi_hat[0,0] = 0.0
    
    #compute full and projected velocities
    u = np.fft.irfft2(-ky*psi_hat)
    u_bar = np.fft.irfft2(-P_LF*ky*psi_hat)
    v = np.fft.irfft2(kx*psi_hat)
    v_bar = np.fft.irfft2(P_LF*kx*psi_hat)
    
    #compute subgrid velocities
    u_prime = u - u_bar
    v_prime = v - v_bar
    
    #return resolved part of the RST components (\bar{u_iu_j})
    uu_hat = P_LF*np.fft.rfft2(u_prime*u_prime)
    uv_hat = P_LF*np.fft.rfft2(u_prime*v_prime)
    vv_hat = P_LF*np.fft.rfft2(v_prime*v_prime)

    return uu_hat, uv_hat, vv_hat

#get Fourier coefficient of the vorticity at next (n+1) time step
def get_w_hat_np1(w_hat_n, w_hat_nm1, VgradW_hat_nm1, P, norm_factor, sgs_hat = 0.0):
    
    #compute jacobian
    VgradW_hat_n = compute_VgradW_hat(w_hat_n, P)
    
    #solve for next time step according to AB/BDI2 scheme
    w_hat_np1 = norm_factor*P*(2.0/dt*w_hat_n - 1.0/(2.0*dt)*w_hat_nm1 - \
                               2.0*VgradW_hat_n + VgradW_hat_nm1 + mu*F_hat - sgs_hat)
    
    return w_hat_np1, VgradW_hat_n

#compute spectral filter
def get_P(cutoff):
    
    P = np.ones([N, N/2+1])
    
    for i in range(N):
        for j in range(N/2+1):
            
            if np.abs(kx[i, j]) > cutoff or np.abs(ky[i, j]) > cutoff:
                P[i, j] = 0.0
                
    return P

#store samples in hierarchical data format, when sample size become very large
def store_samples_hdf5():
  
    fname = HOME + '/samples/' + store_ID + '_t_' + str(np.around(t_end/day, 1)) + '.hdf5'
    
    print 'Storing samples in ', fname
    
    if os.path.exists(HOME + '/samples') == False:
        os.makedirs(HOME + '/samples')
    
    #create HDF5 file
    h5f = h5py.File(fname, 'w')
    
    #store numpy sample arrays as individual datasets in the hdf5 file
    for q in QoI:
        h5f.create_dataset(q, data = samples[q])
        
    h5f.close()    

def draw_2w():
    plt.subplot(121, aspect = 'equal', title=r'$Q_1\; ' + r't = '+ str(np.around(t/day, 2)) + '\;[days]$')
    #plt.xlim([-0.0001, 0.0001])
    #plt.ylim([-0.0001, 0.0001])
    #plt.contourf(x, y, L_over_K1.reshape([N,N]), 100)
    plt.contourf(x, y, EF_nm1_check, 100)
    #plt.tricontourf(mu_i[:,1], mu_i[:,0], L_over_K1, 500)
    plt.colorbar()
    plt.subplot(122, aspect = 'equal', title=r'$Q_2$')
    #plt.contourf(x, y, lambda_i[:,1].reshape([N, N]) - lambda_i[:,0].reshape([N, N]), 100)
    #ll = (lambda_i[:,1].reshape([N, N]) - lambda_i[:,0].reshape([N, N]))/(lambda_i[:,1].reshape([N, N]) + lambda_i[:,0].reshape([N, N]))
    #plt.contourf(x, y, beta1*EF_nm1_tensor_mod, 100)
    #plt.contourf(x, y, L_over_K.reshape([N, N]), 100)
    plt.contourf(x, y, EF_nm1_exact, 100)
    #plt.subplot(122, title=r'$Q_2$')
    #plt.plot(ll[:,90])
    
    #plt.subplot(122, title=r'$Q_2$')

    #least squares estimate for beta1 only (beta0 = 0 a priori)
    #beta1 = np.sum(EF_nm1_exact*EF_n_mod)/np.sum(EF_n_mod*EF_n_mod)
    #print beta1
    # plt.contourf(x, y, beta1*EF_n_mod, 100)
    plt.colorbar()
    plt.tight_layout()

def draw_3w():
    plt.subplot(131, aspect='equal', title=r'$Q_1\; ' + r't = '+ str(np.around(t/day,2)) + '\;[days]$')
    plt.contourf(x, y, r, 100)
    plt.subplot(132, aspect='equal', title=r'$Q_2$')
    plt.contourf(x, y, EF_n_mod, 100)    
    plt.subplot(133, aspect='equal', title=r'$Q_3$')
    plt.contourf(x, y, EF_nm1_exact, 100) 
    plt.tight_layout()
    
#compute the spatial correlation coeffient at a given time
def spatial_corr_coef(X, Y):
    return np.mean((X - np.mean(X))*(Y - np.mean(Y)))/(np.std(X)*np.std(Y))

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
        print 'Energy = ', E, ', enstrophy = ', Z
    return E, Z

"""
***************************
* M A I N   P R O G R A M *
***************************
"""

import numpy as np
import matplotlib.pyplot as plt
import os, cPickle
import h5py
from drawnow import drawnow
from scipy.integrate import simps

plt.close('all')
plt.rcParams['image.cmap'] = 'seismic'

HOME = os.path.abspath(os.path.dirname(__file__))

#plt.close('all')
#plt.rcParams['image.cmap'] = 'seismic'

#number of gridpoints in 1D
I = 7
N = 2**I

#2D grid
h = 2*np.pi/N
axis = h*np.arange(1, N+1)
[x , y] = np.meshgrid(axis , axis)

#frequencies
k = np.fft.fftfreq(N)*N

kx = np.zeros([N, N/2+1]) + 0.0j
ky = np.zeros([N, N/2+1]) + 0.0j

for i in range(N):
    for j in range(N/2+1):
        kx[i, j] = 1j*k[j]
        ky[i, j] = 1j*k[i]

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

#time scale
Omega = 7.292*10**-5
day = 24*60**2*Omega

#viscosities
decay_time_nu = 5.0
decay_time_mu = 90.0
nu = 1.0/(day*Ncutoff**2*decay_time_nu)
#nu_LF = 1.0/(day*Ncutoff_LF**2*decay_time_nu)
nu_LF = 1.0/(day*Ncutoff**2*decay_time_nu)
mu = 1.0/(day*decay_time_mu)

#start, end time (in days) + time step
t = 250.0*day
t_end = (t + 5.0*365)*day
#t_end = 350.0*day
t_data = 500.0*day

dt = 0.01
n_steps = np.ceil((t_end-t)/dt).astype('int')

#############
# USER KEYS #
#############

sim_ID = 'RST_TEST5'
store_ID = 'deterministic_decay_time5.0'
plot_frame_rate = np.floor(1.0*day/dt).astype('int')
store_frame_rate = np.floor(0.05*day/dt).astype('int')
S = np.floor(n_steps/store_frame_rate).astype('int')

state_store = False
restart = True
store = False
plot = True
eddy_forcing_type = 'exact'

#QoI to store, First letter in caps implies an NxN field, otherwise a scalar 

#training data QoI
#QoI = ['DM_LF', 'DN_LF', 'M_LF', 'N_LF', 'M_HF', 'N_HF', 'Jac_HF', 'Jac_LF', 't']
#QoI = ['W_LF', 'Jac_HF', 'Jac_LF', 'EF_MOD', 't']

#prediction data QoI
QoI = ['e_HF', 'z_HF', 'e_LF', 'z_LF', 'rho', 't']
Q = len(QoI)

#allocate memory
samples = {}

if store == True:
    samples['S'] = S
    samples['N'] = N
    
    for q in range(Q):
        
        #a field
        if QoI[q][0].isupper():
            samples[QoI[q]] = np.zeros([S, N, N/2+1]) + 0.0j
        #a scalar
        else:
            samples[QoI[q]] = np.zeros(S)

#forcing term
F = 2**1.5*np.cos(5*x)*np.cos(5*y);
F_hat = np.fft.rfft2(F);

if restart == True:
    
    state = cPickle.load(open(HOME + '/restart/' + sim_ID + '_t_' + str(np.around(t/day, 1)) + '.pickle'))
    for key in state.keys():
        print key
        vars()[key] = state[key]
else:
    
    #initial condition
    w = np.sin(4.0*x)*np.sin(4.0*y) + 0.4*np.cos(3.0*x)*np.cos(3.0*y) + \
        0.3*np.cos(5.0*x)*np.cos(5.0*y) + 0.02*np.sin(x) + 0.02*np.cos(y)

    #initial Fourier coefficients at time n and n-1
    w_hat_n_HF = P*np.fft.rfft2(w)
    w_hat_nm1_HF = np.copy(w_hat_n_HF)
    
    w_hat_n_LF = P_LF*np.fft.rfft2(w)
    w_hat_nm1_LF = np.copy(w_hat_n_LF)
    
    #initial Fourier coefficients of the jacobian at time n and n-1
    VgradW_hat_n_HF = compute_VgradW_hat(w_hat_n_HF, P)
    VgradW_hat_nm1_HF = np.copy(VgradW_hat_n_HF)
    
    VgradW_hat_n_LF = compute_VgradW_hat(w_hat_n_LF, P_LF)
    VgradW_hat_nm1_LF = np.copy(VgradW_hat_n_LF)

#constant factor that appears in AB/BDI2 time stepping scheme   
norm_factor = 1.0/(3.0/(2.0*dt) - nu*k_squared + mu)
norm_factor_LF = 1.0/(3.0/(2.0*dt) - nu_LF*k_squared + mu)

j = 0; j2 = 0; idx = 0

if plot == True:
    plt.figure()

rho1 = []; Beta1 = []

#beta1_mean = -0.001276      #nu decay time of 1
beta1_mean = -0.0004894     #nu decay time of 5

#time loop
for n in range(n_steps):    
    
    #solve for next time step
    w_hat_np1_HF, VgradW_hat_n_HF = get_w_hat_np1(w_hat_n_HF, w_hat_nm1_HF, VgradW_hat_nm1_HF, P, norm_factor)
  
    EF_hat_nm1_exact = P_LF*VgradW_hat_nm1_HF - VgradW_hat_nm1_LF 

    #LHS of the LF model (Euler)
    VgradW_hat_nm1_full = compute_VgradW_hat(w_hat_nm1_LF, P)
    lhs_hat_n_LF = (w_hat_n_LF - w_hat_nm1_LF)/dt + VgradW_hat_nm1_full
    
    #LHS of the LF model (AB/BDI2)
    #VgradW_hat_nm1_full = compute_VgradW_hat(w_hat_nm1_LF, P)
    #VgradW_hat_n_full = compute_VgradW_hat(w_hat_n_LF, P)
    #lhs_hat_n_LF = (w_hat_n_LF - w_hat_nm1_LF)/dt + 2.0*VgradW_hat_n_full - VgradW_hat_nm1_full

    #model eddy forcing
    EF_hat_n_mod = (kx**2 + ky**2)*lhs_hat_n_LF

    #EXACT eddy forcing (for reference)
    if eddy_forcing_type == 'exact':
        EF_hat_nm1 = EF_hat_nm1_exact
    #model eddy forcing
    if eddy_forcing_type == 'model':
        EF_hat_nm1 = beta1_mean*tensor_eddy_force_hat(w_hat_nm1_LF, -EF_hat_n_mod, P_LF)
    #NO eddy forcing
    elif eddy_forcing_type == 'unparam':
        EF_hat_nm1 = np.zeros([N, N/2+1])

    w_hat_np1_LF, VgradW_hat_n_LF = get_w_hat_np1(w_hat_n_LF, w_hat_nm1_LF, VgradW_hat_nm1_LF, P_LF, norm_factor_LF, EF_hat_nm1)
    
    #plot results to screen during iteration
    if j == plot_frame_rate and plot == True:
        j = 0

        w_np1_HF = np.fft.irfft2(P_LF*w_hat_np1_HF)
        w_np1_LF = np.fft.irfft2(w_hat_np1_LF)
        
        EF_nm1 = np.fft.irfft2(EF_hat_nm1)
        EF_nm1_exact = np.fft.irfft2(EF_hat_nm1_exact)

        #eddy forcing computed from the tensor
        EF_hat_nm1_tensor_mod = tensor_eddy_force_hat(w_hat_nm1_LF, -EF_hat_n_mod, P_LF)
        EF_nm1_tensor_mod = np.fft.irfft2(EF_hat_nm1_tensor_mod)
        beta1 = np.sum(EF_nm1_exact*EF_nm1_tensor_mod)/np.sum(EF_nm1_tensor_mod**2)
        print beta1
        Beta1.append(beta1)

        delta_w_hat_nm1 = w_hat_nm1_HF - w_hat_nm1_LF
        lambda_i, theta, tke, dM_2, dN_2, C = rst_eigs(w_hat_nm1_LF, delta_w_hat_nm1)
        
        ######
        # R1 #
        ######
        
        mu_i, theta1 = R1_eigs(w_hat_nm1_LF, delta_w_hat_nm1)
        L_over_K1 = (mu_i[:,1] - mu_i[:,0])/(mu_i[:,1] + mu_i[:,0])
        
        K1 = 0.5*(mu_i[:,1] + mu_i[:,0])
        c11_star1 = K1*(np.cos(2.0*theta1)*L_over_K1 + np.ones(N**2))
        c22_star1 = K1*(-np.cos(2.0*theta1)*L_over_K1 + np.ones(N**2))
        c12_star1 = K1*(np.sin(2.0*theta1)*L_over_K1)
       
        c11_hat_star1 = P_LF*np.fft.rfft2(c11_star1.reshape([N, N]))
        c22_hat_star1 = P_LF*np.fft.rfft2(c22_star1.reshape([N, N]))
        c12_hat_star1 = P_LF*np.fft.rfft2(c12_star1.reshape([N, N]))

        #the reconstructed part of the eddy forcing due to the cross terms
        EF_hat_nm1_star1 = (kx**2 - ky**2)*c12_hat_star1 + kx*ky*(c22_hat_star1 - c11_hat_star1)
        EF_nm1_star1 = np.fft.irfft2(EF_hat_nm1_star1)
        
        #the exact part of the eddy forcing due to the cross terms
        check_hat1 = tensor_eddy_force_hat1(w_hat_nm1_LF, delta_w_hat_nm1, P_LF)
        check1 = np.fft.irfft2(check_hat1)

        ######
        # R2 #
        ######

        L2 = 0.5*(lambda_i[:, 1] - lambda_i[:, 0])
        L2 = L2.reshape([N, N])
        L = np.sqrt(dM_2**2 + dN_2**2)

        L_over_K = (lambda_i[:,1] - lambda_i[:,0])/(lambda_i[:,1] + lambda_i[:,0])
        alpha = 0.0
        L_over_K_star = L_over_K + alpha*(np.zeros(N**2) - L_over_K)

        K = 0.5*(lambda_i[:,1] + lambda_i[:,0])

        c11_star = K*(np.cos(2.0*theta)*L_over_K_star + np.ones(N**2))
        c22_star = K*(-np.cos(2.0*theta)*L_over_K_star + np.ones(N**2))
        c12_star = K*(np.sin(2.0*theta)*L_over_K_star)
       
        c11_hat_star = P_LF*np.fft.rfft2(c11_star.reshape([N, N]))
        c22_hat_star = P_LF*np.fft.rfft2(c22_star.reshape([N, N]))
        c12_hat_star = P_LF*np.fft.rfft2(c12_star.reshape([N, N]))

        #the reconstructed part of the eddy forcing due to the RST
        EF_hat_nm1_star = (kx**2 - ky**2)*c12_hat_star + kx*ky*(c22_hat_star - c11_hat_star)
        EF_nm1_star = np.fft.irfft2(EF_hat_nm1_star)
        
        #the exact part of the eddy forcing due to the RST
        check_hat = tensor_eddy_force_hat2(w_hat_nm1_LF, delta_w_hat_nm1, P_LF)
        check = np.fft.irfft2(check_hat)

        rho1 = spatial_corr_coef(EF_nm1_exact, beta1*EF_nm1_tensor_mod)
        print rho1
        print '-----'
        
        ######
        # R3 #
        ######
        
        EF_hat_nm1_check = tensor_eddy_force_hat3(w_hat_nm1_HF, w_hat_nm1_LF)
        EF_nm1_check = np.fft.irfft2(EF_hat_nm1_check)
        
        E_HF, Z_HF = compute_E_and_Z(P_LF*w_hat_np1_HF)
        E_LF, Z_LF = compute_E_and_Z(w_hat_np1_LF)
        
        drawnow(draw_2w)
        
    #store samples to dict
    if j2 == store_frame_rate and store == True:
        j2 = 0
        
        print 'n = ', n, ' of ', n_steps

        #################
        # training data #
        #################
        #samples['DN'][idx, :, :] = dN_hat_n 
        #samples['DM'][idx, :, :] = dM_hat_n
        #samples['DN_LF'][idx, :, :] = dN_hat_n_LF 
        #samples['DM_LF'][idx, :, :] = dM_hat_n_LF
        #samples['M_LF'][idx, :, :] = M_hat_n_LF  
        #samples['N_LF'][idx, :, :] = N_hat_n_LF
        #samples['M_R'][idx, :, :] = M_hat_n_R  
        #samples['N_R'][idx, :, :] = N_hat_n_R
        #samples['M_HF'][idx, :, :] = M_hat_n_HF 
        #samples['N_HF'][idx, :, :] = N_hat_n_HF
        #samples['Jac_R'][idx, :, :] = VgradW_hat_n_R 
        #samples['EF_MOD'][idx, :, :] = EF_hat_n_mod 
        #samples['Jac_LF'][idx, :, :] = VgradW_hat_nm1_LF 
        #samples['Jac_HF'][idx, :, :] = VgradW_hat_nm1_HF
        #samples['W_LF'][idx, :, :] = w_hat_nm1_LF

        ###################
        # prediction data #
        ###################

        E_HF, Z_HF = compute_E_and_Z(P_LF*w_hat_np1_HF)
        E_LF, Z_LF = compute_E_and_Z(w_hat_np1_LF)
       
        samples['e_HF'][idx] = E_HF
        samples['z_HF'][idx] = Z_HF
        samples['e_LF'][idx] = E_LF
        samples['z_LF'][idx] = Z_LF

        EF_nm1 = np.fft.irfft2(EF_hat_nm1)
        EF_nm1_exact = np.fft.irfft2(EF_hat_nm1_exact)

        if eddy_forcing_type != 'unparam':
            samples['rho'][idx] = spatial_corr_coef(EF_nm1, EF_nm1_exact)

        #print samples['rho'][idx]

        samples['t'][idx] = t
        
        idx += 1  
        
    #update variables
    w_hat_nm1_HF = np.copy(w_hat_n_HF)
    w_hat_n_HF = np.copy(w_hat_np1_HF)
    VgradW_hat_nm1_HF = np.copy(VgradW_hat_n_HF)

    w_hat_nm1_LF = np.copy(w_hat_n_LF)
    w_hat_n_LF = np.copy(w_hat_np1_LF)
    VgradW_hat_nm1_LF = np.copy(VgradW_hat_n_LF)
    
    t += dt
    j += 1
    j2 += 1
    
#store the state of the system to allow for a simulation restart at t > 0
if state_store == True:
    
    keys = ['t', 'w_hat_nm1_HF', 'w_hat_n_HF', 'VgradW_hat_nm1_HF', \
            'w_hat_nm1_LF', 'w_hat_n_LF', 'VgradW_hat_nm1_LF']
    
    state = {}
    
    for key in keys:
        state[key] = vars()[key]
    
    if os.path.exists(HOME + '/restart') == False:
        os.makedirs(HOME + '/restart')
    
    cPickle.dump(state, open(HOME + '/restart/' + sim_ID + '_t_' + str(np.around(t_end/day,1)) + '.pickle', 'w'))

#store the samples
if store == True:
    store_samples_hdf5() 

plt.figure()
plt.plot(Beta1)
print 'Mean beta1 = ', np.mean(Beta1)

plt.show()
