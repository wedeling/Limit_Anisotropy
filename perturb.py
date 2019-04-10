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

#pseudo-spectral technique to solve for Fourier coefs of Jacobian
def compute_VgradEta_hat(w_hat_n, eta_hat_n, P):
    
    #compute streamfunction
    psi_hat_n = w_hat_n/k_squared_no_zero
    psi_hat_n[0,0] = 0.0
    
    #compute jacobian in physical space
    u_n = np.fft.irfft2(-ky*psi_hat_n)
    eta_x_n = np.fft.irfft2(kx*eta_hat_n)

    v_n = np.fft.irfft2(kx*psi_hat_n)
    eta_y_n = np.fft.irfft2(ky*eta_hat_n)
    
    VgradEta_n = u_n*eta_x_n + v_n*eta_y_n
    
    #return to spectral space
    VgradEta_hat_n = np.fft.rfft2(VgradEta_n)
    
    VgradEta_hat_n *= P
    
    return VgradEta_hat_n

def pos_def_tensor_eddy_force(w_hat, P):
    
    #compute streamfunctions
    psi_hat = w_hat/k_squared_no_zero
    psi_hat[0,0] = 0.0
    
    #compute full and projected velocities
    u = np.fft.irfft2(-ky*psi_hat)
    v = np.fft.irfft2(kx*psi_hat)
    
    R_11 = u*u
    R_12 = u*v
    R_22 = v*v

    R_11_hat = P*np.fft.rfft2(R_11)
    R_12_hat = P*np.fft.rfft2(R_12)
    R_22_hat = P*np.fft.rfft2(R_22)

    R = np.zeros([N**2, 2, 2])

    R[:, 0, 0] = np.fft.irfft2(R_11_hat).flatten()
    R[:, 0, 1] = np.fft.irfft2(R_12_hat).flatten()
    R[:, 1, 0] = R[:, 0, 1]
    R[:, 1, 1] = np.fft.irfft2(R_22_hat).flatten()

    EF_hat = (kx**2 - ky**2)*R_12_hat + kx*ky*(R_22_hat - R_11_hat)
    
    return EF_hat, R

def perturbed_eddy_forcing(R1, R2, P):

    R1_11_hat = P*np.fft.rfft2(R1[:, 0, 0].reshape([N, N]))
    R1_22_hat = P*np.fft.rfft2(R1[:, 1, 1].reshape([N, N]))
    R1_12_hat = P*np.fft.rfft2(R1[:, 0, 1].reshape([N, N]))

    R2_11_hat = P*np.fft.rfft2(R2[:, 0, 0].reshape([N, N]))
    R2_22_hat = P*np.fft.rfft2(R2[:, 1, 1].reshape([N, N]))
    R2_12_hat = P*np.fft.rfft2(R2[:, 0, 1].reshape([N, N]))

    EF1_hat = (kx**2 - ky**2)*R1_12_hat + kx*ky*(R1_22_hat - R1_11_hat)
    EF2_hat = (kx**2 - ky**2)*R2_12_hat + kx*ky*(R2_22_hat - R2_11_hat)

    return EF1_hat - EF2_hat

#compute the eigenvalues, eigenvector angle and tke of R \in [N**2, 2, 2]
def eigs(R):

    #zero out any non-physical entries
    idx0 = np.where(R[:, 0, 0] < 0.0)[0]
    R[idx0, 0, 0] = 0.0
    idx0 = np.where(R[:, 1, 1] < 0.0)[0]
    R[idx0, 1, 1] = 0.0

    tke = 0.5*(R[:, 0, 0] + R[:, 1, 1])
    tke = np.sqrt(tke**2)

    #NOTE: the use of eigh seems to already sort the eigenvalues, whereas eig does not
    lambda_i, V = np.linalg.eigh(R)
    lambda_i = np.sqrt(lambda_i**2)

    #extract the angle of the eigenvectors, using the components of the 1st vector
    V11 = V[:, 0, 0] 
    V21 = V[:, 1, 0]
    theta = -np.arctan2(V11, V21)

    return lambda_i, theta, tke

#reconstruct the R tensor, given the kinetic energy, eigenvalue ratio eta and eigenvector angle theta
def reconstruct_R(tke, eta, theta):

    R11 = tke*(np.cos(2.0*theta)*eta + np.ones(N**2))
    R22 = tke*(-np.cos(2.0*theta)*eta + np.ones(N**2))
    R12 = tke*(np.sin(2.0*theta)*eta)

    R = np.zeros([N**2, 2, 2])

    R[:, 0, 0] = R11
    R[:, 0, 1] = R12
    R[:, 1, 0] = R12
    R[:, 1, 1] = R22

    return R

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
    
    P = np.ones([N, int(N/2+1)])
    
    for i in range(N):
        for j in range(int(N/2+1)):
            
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

#store samples in hierarchical data format, when sample size become very large
def store_samples_hdf5():
  
    fname = HOME + '/samples/' + store_ID + '_t_' + str(np.around(t_end/day, 1)) + '.hdf5'
    
    print('Storing samples in ', fname)
    
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
    plt.contourf(x, y, test1, 100)
    plt.colorbar()
    plt.subplot(122, aspect = 'equal', title=r'$Q_2$')
    plt.contourf(x, y, test2, 100)
    plt.colorbar()
    plt.tight_layout()
    
def draw_stats():
    plt.subplot(121, xlabel=r't')
    plt.plot(T, energy_HF, label=r'$E^{HF}$')
    plt.plot(T, energy_LF, label=r'$E^{LF}$')
    plt.legend(loc=0)
    plt.subplot(122, xlabel=r't')
    plt.plot(T, enstrophy_HF, label=r'$Z^{HF}$')
    plt.plot(T, enstrophy_LF, label=r'$Z^{LF}$')
    plt.legend(loc=0)
    plt.tight_layout()
    
#compute the spatial correlation coeffient at a given time
def spatial_corr_coef(X, Y):
    return np.mean((X - np.mean(X))*(Y - np.mean(Y)))/(np.std(X)*np.std(Y))

#compute the energy and enstrophy at t_n
def compute_E_and_Z(w_hat_n, verbose=True):
    
#Compute stats using Simpson's integration rule    
#    psi_hat_n = w_hat_n/k_squared_no_zero
#    psi_hat_n[0,0] = 0.0
#    psi_n = np.fft.irfft2(psi_hat_n)
#    w_n = np.fft.irfft2(w_hat_n)
#    
#    e_n = -0.5*psi_n*w_n
#    z_n = 0.5*w_n**2
#
#    E = simps(simps(e_n, axis), axis)/(2*np.pi)**2
#    Z = simps(simps(z_n, axis), axis)/(2*np.pi)**2

    #compute stats using Fourier coefficients - is faster
    #convert rfft2 coefficients to fft2 coefficients
    w_hat_full = np.zeros([N, N]) + 0.0j
    w_hat_full[0:N, 0:int(N/2+1)] = w_hat_n
    w_hat_full[map_I, map_J] = np.conjugate(w_hat_n[I, J])
    w_hat_full *= P_full
    
    #compute Fourier coefficients of stream function
    psi_hat_full = w_hat_full/k_squared_no_zero_full
    psi_hat_full[0,0] = 0.0

    #compute energy and enstrophy (density)
    Z = 0.5*np.sum(w_hat_full*np.conjugate(w_hat_full))/N**4
    E = -0.5*np.sum(psi_hat_full*np.conjugate(w_hat_full))/N**4

    if verbose:
        #print 'Energy = ', E, ', enstrophy = ', Z
        print('Energy = ', E.real, ', enstrophy = ', Z.real)

    return E.real, Z.real

"""
***************************
* M A I N   P R O G R A M *
***************************
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import _pickle as cPickle
import h5py
from drawnow import drawnow
from scipy.integrate import simps

plt.close('all')
plt.rcParams['image.cmap'] = 'seismic'

HOME = os.path.abspath(os.path.dirname(__file__))

#number of gridpoints in 1D
N = 2**7

#2D grid
h = 2*np.pi/N
#axis = h*np.arange(1, N+1)
axis = np.linspace(0.0, 2.0*np.pi, N)
[x , y] = np.meshgrid(axis , axis)

#frequencies
k = np.fft.fftfreq(N)*N

kx = np.zeros([N, int(N/2+1)]) + 0.0j
ky = np.zeros([N, int(N/2+1)]) + 0.0j

for i in range(N):
    for j in range(int(N/2+1)):
        kx[i, j] = 1j*k[j]
        ky[i, j] = 1j*k[i]

k_squared = kx**2 + ky**2
k_squared_no_zero = np.copy(k_squared)
k_squared_no_zero[0,0] = 1.0

kx_full = np.zeros([N, N]) + 0.0j
ky_full = np.zeros([N, N]) + 0.0j

for i in range(N):
    for j in range(N):
        kx_full[i, j] = 1j*k[j]
        ky_full[i, j] = 1j*k[i]

k_squared_full = kx_full**2 + ky_full**2
k_squared_no_zero_full = np.copy(k_squared_full)
k_squared_no_zero_full[0,0] = 1.0

#cutoff in pseudospectral method
Ncutoff = N/3
Ncutoff_LF = 2**6/3 

#spectral filter for the real FFT2
P = get_P(Ncutoff)
P_LF = get_P(Ncutoff_LF)
P_U = P - P_LF

#spectral filter for the full FFT2 (used in compute_E_Z)
P_full = get_P_full(Ncutoff_LF)

#map from the rfft2 coefficient indices to fft2 coefficient indices
#Use: see compute_E_Z subroutine
shift = np.zeros(N).astype('int')
for i in range(1,N):
    shift[i] = np.int(N-i)
I = range(N);J = range(np.int(N/2+1))
map_I, map_J = np.meshgrid(shift[I], shift[J])
I, J = np.meshgrid(I, J)

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
#t_end = (t + 5.0*365)*day
t_end = 300.0*day

#time step
dt = 0.01
n_steps = np.ceil((t_end-t)/dt).astype('int')

#number of step after which the eddy forcing is perturbed
perturb_step = np.int(0.5*day/dt)
#perturb_step = 1

#############
# USER KEYS #
#############

sim_ID = 'LIMIT_0'
store_ID = 'test'
plot_frame_rate = np.floor(1.0*day/dt).astype('int')
store_frame_rate = np.floor(0.5*day/dt).astype('int')
S = np.floor(n_steps/store_frame_rate).astype('int')

state_store = False
restart = True
store = False
plot = True
eddy_forcing_type = 'lag'

alpha = 0.9
eta_limit = 0.5
tau = 100.0

#QoI to store, First letter in caps implies an NxN field, otherwise a scalar 

#prediction data QoI
QoI = ['Eta1', 'Eta2', 'Theta1', 'Theta2', 't']
Q = len(QoI)

#allocate memory
samples = {}

if store == True:
    samples['S'] = S
    samples['N'] = N
    
    for q in range(Q):
        
        #a field
        if QoI[q][0].isupper():
            #samples[QoI[q]] = np.zeros([S, N, N/2+1]) + 0.0j
            samples[QoI[q]] = np.zeros([S, N, N])
        #a scalar
        else:
            samples[QoI[q]] = np.zeros(S)

#forcing term
F = 2**1.5*np.cos(5*x)*np.cos(5*y);
F_hat = np.fft.rfft2(F);

if restart == True:
    
    fname = HOME + '/restart/' + sim_ID + '_t_' + str(np.around(t/day,1)) + '.hdf5'
    
    #create HDF5 file
    h5f = h5py.File(fname, 'r')
    
    for key in h5f.keys():
        print(key)
        vars()[key] = h5f[key][:]
        
    h5f.close()
   
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
    
#initialize the lag pde
if eddy_forcing_type == 'lag':
    
    #compute the pos semi-def tensor R2
    EF1_hat, R1 = pos_def_tensor_eddy_force(w_hat_n_LF, P_LF)
    
    #eigenvalue decomposition of the closed, pos-semi-def part of the eddy forcing, expensive
    lambda2_i, theta2, tke2 = eigs(R1)
    
    #L/K
    eta = (lambda2_i[:,1] - lambda2_i[:,0])/(lambda2_i[:,1] + lambda2_i[:,0])
    eta_hat_n_tilde = np.fft.rfft2(eta.reshape([N, N]))
    eta_hat_nm1_tilde = np.copy(eta_hat_n_tilde)
    
    VgradEta_hat_nm1 = compute_VgradEta_hat(w_hat_nm1_LF, eta_hat_nm1_tilde, P_LF)

#constant factor that appears in AB/BDI2 time stepping scheme   
norm_factor = 1.0/(3.0/(2.0*dt) - nu*k_squared + mu)
norm_factor_LF = 1.0/(3.0/(2.0*dt) - nu_LF*k_squared + mu)

j = 0; j2 = 0; idx = 0

if plot == True:
    plt.figure()
    energy_HF = []; energy_LF = []; enstrophy_HF = []; enstrophy_LF = []; T = []

#time loop
for n in range(n_steps):    
    
    #solve for next time step
    w_hat_np1_HF, VgradW_hat_n_HF = get_w_hat_np1(w_hat_n_HF, w_hat_nm1_HF, VgradW_hat_nm1_HF, P, norm_factor)
  
    EF_hat_nm1_exact = P_LF*VgradW_hat_nm1_HF - VgradW_hat_nm1_LF #+ (nu_LF - nu)*k_squared*w_hat_nm1_LF

    #EXACT eddy forcing (for reference)
    if eddy_forcing_type == 'exact':
        EF_hat = EF_hat_nm1_exact

    #perturbed model eddy forcing    
    elif eddy_forcing_type == 'perturb' and np.mod(n, perturb_step) == 0:
        
        #compute the pos semi-def tensor R2
        EF1_hat, R1 = pos_def_tensor_eddy_force(w_hat_nm1_HF, P)
        EF2_hat, R2 = pos_def_tensor_eddy_force(w_hat_nm1_LF, P_LF)

        #eigenvalue decomposition of the closed, pos-semi-def part of the eddy forcing, expensive
        lambda1_i, theta1, tke1 = eigs(R1)
        lambda2_i, theta2, tke2 = eigs(R2)

        #L/K
        eta1 = (lambda1_i[:,1] - lambda1_i[:,0])/(lambda1_i[:,1] + lambda1_i[:,0])
        eta2 = (lambda2_i[:,1] - lambda2_i[:,0])/(lambda2_i[:,1] + lambda2_i[:,0])
        
        #perturb eta towards one of its limits (0 or 1)
        eta_star = eta2 + alpha*(eta_limit - eta2) 

        #construct the modelled R1
        R1_star = reconstruct_R(tke1, eta1, theta1)
        
        #compute the model eddy forcing
        EF_hat = perturbed_eddy_forcing(R1_star, R2, P_LF) #+ (nu_LF - nu)*k_squared*w_hat_nm1_LF
    
    #solve a lag pde for eta
    elif eddy_forcing_type == 'lag':   
        
        if np.mod(n, perturb_step) == 0:
            #compute the pos semi-def tensor R2
            EF2_hat, R2 = pos_def_tensor_eddy_force(w_hat_n_LF, P_LF)
            
            #eigenvalue decomposition of the closed, pos-semi-def part of the eddy forcing, expensive
            lambda2_i, theta2, tke2 = eigs(R2)
            
            #L/K
            eta = (lambda2_i[:,1] - lambda2_i[:,0])/(lambda2_i[:,1] + lambda2_i[:,0])
            eta_hat = np.fft.rfft2(eta.reshape([N, N]))
        
        #solve the lag pde
        VgradEta_hat_n = compute_VgradEta_hat(w_hat_n_LF, eta_hat_n_tilde, P_LF)
        
        eta_hat_np1_tilde = 2.0*dt/3.0*P_LF*(2.0/dt*eta_hat_n_tilde - eta_hat_nm1_tilde/(2.0*dt) - \
                                             2.0*VgradEta_hat_n + VgradEta_hat_nm1 + \
                                             tau*eta_hat - tau*eta_hat_n_tilde)
        
        #update variables
        eta_hat_nm1_tilde = np.copy(eta_hat_n_tilde)
        eta_hat_n_tilde = np.copy(eta_hat_np1_tilde)
        VgradEta_hat_nm1 = np.copy(VgradEta_hat_n)
        
        EF_hat = EF_hat_nm1_exact
        
    #NO eddy forcing
    elif eddy_forcing_type == 'unparam':
        EF_hat = np.zeros([N, int(N/2+1)])

    w_hat_np1_LF, VgradW_hat_n_LF = get_w_hat_np1(w_hat_n_LF, w_hat_nm1_LF, VgradW_hat_nm1_LF, P_LF, norm_factor_LF, EF_hat)
    
    #plot results to screen during iteration
    if j == plot_frame_rate and plot == True:
        j = 0
        
        #HF and LF vorticities
        w_np1_HF = np.fft.irfft2(P_LF*w_hat_np1_HF)
        w_np1_LF = np.fft.irfft2(w_hat_np1_LF)
        
        test1 = np.fft.irfft2(eta_hat)
        test2 = np.fft.irfft2(eta_hat_np1_tilde)
        
#        w_hat2 = np.zeros([N, N]) + 0.0j
#        w_hat2[0:N, 0:np.int(N/2+1)] = w_hat_np1_HF
#        w_hat2[map_I, map_J] = np.conjugate(w_hat_np1_HF[I, J])
#        test = np.fft.ifft2(P_full*w_hat2)
#        
#        #exact eddy forcing
#        EF_nm1_exact = np.fft.irfft2(EF_hat_nm1_exact)
#        
#        EF = np.fft.irfft2(EF_hat)
#
#        #pos semi-def eddy forcing tensor of the HF part
#        EF1_hat, R1 = pos_def_tensor_eddy_force(w_hat_nm1_HF)
#        
#        #pos semi-def eddy forcing tensor of the LF part
#        EF2_hat, R2 = pos_def_tensor_eddy_force(w_hat_nm1_LF)
#        
#        #check: should be the same as the exact eddy forcing, if the system
#        #is forced by the exact eddy forcing
#        EF_nm1_check = np.fft.irfft2(EF1_hat - EF2_hat)
#        
#        #eigenvalue decomposition of the closed, pos-semi-def part of the eddy forcing
#        lambda_i, theta, tke = eigs(R1)
#        eta = (lambda_i[:,1] - lambda_i[:,0])/(lambda_i[:,1] + lambda_i[:,0])
#        
#        #perturbed eta
#        eta_star = eta + alpha*(eta_limit - eta) 
#
#        #reconstuct R1 with the perturbed eta
#        R1_star = reconstruct_R(tke, eta_star, theta)
#        
#        #compute the perturbed eddy forcing
#        EF_star_hat = perturbed_eddy_forcing(R1_star, R2)
#        EF_star = np.fft.irfft2(EF_star_hat)

        #compute stats
        E_HF, Z_HF = compute_E_and_Z(P_LF*w_hat_np1_HF)
        E_LF, Z_LF = compute_E_and_Z(w_hat_np1_LF)
        print('------------------')
        
        energy_HF.append(E_HF); energy_LF.append(E_LF)
        enstrophy_HF.append(Z_HF); enstrophy_LF.append(Z_LF)
        T.append(t)

        #drawnow(draw_stats)
        drawnow(draw_2w)
        
    #store samples to dict
    if j2 == store_frame_rate and store == True:
        j2 = 0
        
        print('n = ', n, ' of ', n_steps)

#        E_HF, Z_HF = compute_E_and_Z(P_LF*w_hat_np1_HF)
#        E_LF, Z_LF = compute_E_and_Z(w_hat_np1_LF)
#       
#        samples['e_HF'][idx] = E_HF
#        samples['z_HF'][idx] = Z_HF
#        samples['e_LF'][idx] = E_LF
#        samples['z_LF'][idx] = Z_LF
        
        #pos semi-def eddy forcing tensor of the HF part
        EF1_hat, R1 = pos_def_tensor_eddy_force(w_hat_nm1_HF)
        
        #pos semi-def eddy forcing tensor of the LF part
        EF2_hat, R2 = pos_def_tensor_eddy_force(w_hat_nm1_LF)

        lambda_i, theta1, tke = eigs(R1)
        eta1 = (lambda_i[:,1] - lambda_i[:,0])/(lambda_i[:,1] + lambda_i[:,0])

        lambda_i, theta2, tke = eigs(R2)
        eta2 = (lambda_i[:,1] - lambda_i[:,0])/(lambda_i[:,1] + lambda_i[:,0])
        
        samples['Eta1'][idx,:,:] = eta1.reshape([N, N])
        samples['Eta2'][idx,:,:] = eta2.reshape([N, N])
        samples['Theta1'][idx,:,:] = theta1.reshape([N, N])
        samples['Theta2'][idx,:,:] = theta2.reshape([N, N])
        
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
    
    keys = ['w_hat_nm1_HF', 'w_hat_n_HF', 'VgradW_hat_nm1_HF', \
            'w_hat_nm1_LF', 'w_hat_n_LF', 'VgradW_hat_nm1_LF']
    
    if os.path.exists(HOME + '/restart') == False:
        os.makedirs(HOME + '/restart')
    
    #cPickle.dump(state, open(HOME + '/restart/' + sim_ID + '_t_' + str(np.around(t_end/day,1)) + '.pickle', 'w'))
    
    fname = HOME + '/restart/' + sim_ID + '_t_' + str(np.around(t_end/day,1)) + '.hdf5'
    
    #create HDF5 file
    h5f = h5py.File(fname, 'w')
    
    #store numpy sample arrays as individual datasets in the hdf5 file
    for key in keys:
        qoi = eval(key)
        h5f.create_dataset(key, data = qoi)
        
    h5f.close()   

#store the samples
if store == True:
    store_samples_hdf5() 

plt.show()
