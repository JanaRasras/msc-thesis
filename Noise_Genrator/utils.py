import numpy as np
from numpy import linalg as alg
from numpy import pi as Pi
# from numpy.random import randn

from scipy.stats import norm
from scipy.signal import coherence, csd
from scipy.special import jv

import matplotlib.pyplot as plt

np.random.seed(13)
jrandn = lambda n: norm.ppf(np.random.rand(*n))

class NoiseGenerator:
    def __init__(self, L: int, n_mics: int, n_sources: int,
                       mode='C3D',                           
                       C=340, Fs=8000
            ) -> None:
        self.L = L                                          # Make L a power of 2 for efficient FFT calculation
        self.n_mics = n_mics
        self.n_sources = n_sources
        self.mode = mode
        self.C = C
        self.Fs = Fs
        self.NFFT = int( 2 ** np.ceil( np.log2(L) )  )

    def __call__(self, rx_pos: list, tx_pos=None):
        """ Generate isotropic noise as it will be received by the microphone array. """
        
        # Mics relative positions & norms w.r.t. 1st sensor
        rx_pos = np.array(rx_pos)
        P = rx_pos - rx_pos[0]                  # TODO: need Transpose?
        # print(P)
        
        if self.mode == 'C1D':
            if tx_pos is None:
                tx_pos = 2*Pi * np.arange(0, 1  , 1/self.n_sources)       # Phi only
            z = self._cylinderical_noise_1d(P, phi=tx_pos)

        elif self.mode == 'C3D':
            if tx_pos is None:
                tx_pos = 2*Pi * np.linspace(0, 1  , self.n_sources)       # Phi only
            z = self._cylinderical_noise_3d(P, phi=tx_pos)
        
        elif self.mode == 'S1D':
            if tx_pos is None:
                tx_pos = np.arccos(2* np.arange(0, 1+1/self.n_sources , 1/self.n_sources)-1)      #@
                # print(tx_pos)
            z = self._spherical_noise_1d(P, phi=tx_pos)

        elif self.mode == 'S3D':
            if tx_pos is None:
                N = 256
                phi = np.zeros(N)
                theta = np.zeros(N)

                for k in range(0,N):
                    h = -1 + 2*(k)/(N-1)
                    phi[k] = np.arccos(h)
                    if k==0 or k==(N-1):
                        theta[k] = 0
                    else:
                        theta[k] = (theta[k-1] + 3.6/np.sqrt(N*(1-h**2))) % (2*Pi)

                # k = np.arange(0, N)
                # h = -1 + 2 *(k-1)/  (N-1)

                # phi = np.arccos(h)
                # theta = 3.6/np.sqrt(N*(1-h**2))
                # theta[0] = 0
                # theta[-1] = 0

                tx_pos = (phi, theta)
                # print(tx_pos)
            z = self._spherical_noise_3d(P, tx_pos=tx_pos)

        else:
            raise Exception('Error: unkowon mode!')
        
        sc_sim, sc_theory, F = self._coherence(P, z)
        # return sc_sim, sc_theory, F           # For plotting
        return z.T                              # For audio save

    def _cylinderical_noise_1d(self, P, phi):
        d = alg.norm(P - P[0], axis=1)
        w = 2*Pi * self.Fs * np.linspace(0, 0.5, self.NFFT//2 + 1)
        X = np.zeros( [self.n_mics, int(self.NFFT//2 +1)] , dtype='complex64')         # for complex analysis: add type or specify imaginary

        for p in phi:
            X_prime = jrandn([self.NFFT//2 + 1]) + 1j * jrandn([self.NFFT//2 + 1])
            X[0,:] = X[0,:] + X_prime
            
            for m in range(1, self.n_mics):                                   # mic index
                Delta = d[m] * np.cos(p)   
                X[m,:] = X[m,:] + X_prime * np.exp(-1j*Delta*w/self.C) 
        
        
        X = X / np.sqrt(self.n_sources)
        NFFT = self.NFFT
        X = np.hstack([
            np.dot(np.sqrt(NFFT)         , np.real(  np.expand_dims(X[:,0], 1)  )),                          # (N, 1)
            np.dot(np.sqrt(int(NFFT//2)) , X[:,1:int(NFFT//2)] ),                                             # (N, M-1)
            np.dot(np.sqrt(NFFT)         , np.real(  np.expand_dims(X[:,int(NFFT//2)], 1)) ),                # (N, M)
            np.dot(np.sqrt(NFFT//2)      , np.conj(X[:, int(NFFT//2-1):0:-1]) )                               # (N, M-1)
        ])

        z =  np.real(np.fft.ifft(X, NFFT, 1))
        z = z[:, 0:self.L+1]                                  # Truncate the output signals

        return z

    def _cylinderical_noise_3d(self, P, phi):
        M = np.array(P).shape[0]                                                       #@ same as n_mics
        # print(M)                                                            
        NFFT = self.NFFT
        X = np.zeros( [self.n_mics, int(self.NFFT//2 +1)] , dtype='complex64')         # for complex analysis: add type or specify imaginary
        w = 2*Pi * self.Fs * np.linspace(0, 0.5, self.NFFT//2 + 1)

        for p in phi:
            X_prime = jrandn([self.NFFT//2 + 1]) + 1j * jrandn([self.NFFT//2 + 1])
            X[0,:] = X[0,:] + X_prime
            v  = np.array([ np.cos(p) , np.sin(p) , 0 ])                            #@

            for m in range(1, self.n_mics):                                         # mic index
                Delta = np.dot((np.transpose(v)) ,  (P[m,:] ))                     # @
                X[m,:] = X[m,:] + X_prime * np.exp(-1j*Delta*w/self.C) 
        
        X = X / np.sqrt(self.n_sources)

        NFFT = self.NFFT
        X = np.hstack([
            np.dot(np.sqrt(NFFT)         , np.real(  np.expand_dims(X[:,0], 1)  )),                          # (N, 1)
            np.dot(np.sqrt(int(NFFT//2)) , X[:,1:int(NFFT//2)] ),                                             # (N, M-1)
            np.dot(np.sqrt(NFFT)         , np.real(  np.expand_dims(X[:,int(NFFT//2)], 1)) ),                # (N, M)
            np.dot(np.sqrt(NFFT//2)      , np.conj(X[:, int(NFFT//2-1):0:-1]) )                               # (N, M-1)
        ])

        z =  np.real(np.fft.ifft(X, NFFT, 1))
        z = z[:, 0:self.L+1]    

        return z

    def _spherical_noise_1d(self, P, phi):
        d = alg.norm(P - P[0], axis=1)
        w = 2*Pi * self.Fs * np.linspace(0, 0.5, self.NFFT//2 + 1)
        X = np.zeros( [self.n_mics, int(self.NFFT//2 +1)] , dtype='complex64')         # for complex analysis: add type or specify imaginary

        for p in phi:
            X_prime = jrandn([self.NFFT//2 + 1]) + 1j * jrandn([self.NFFT//2 + 1])
            X[0,:] = X[0,:] + X_prime
            
            for m in range(1, self.n_mics):                                   # mic index
                Delta = d[m] * np.cos(p)   
                X[m,:] = X[m,:] + X_prime * np.exp(-1j*Delta*w/self.C) 
        
        
        X = X / np.sqrt(self.n_sources)

        NFFT = self.NFFT
        X = np.hstack([
            np.dot(np.sqrt(NFFT)         , np.real(  np.expand_dims(X[:,0], 1)  )),                          # (N, 1)
            np.dot(np.sqrt(int(NFFT//2)) , X[:,1:int(NFFT//2)] ),                                             # (N, M-1)
            np.dot(np.sqrt(NFFT)         , np.real(  np.expand_dims(X[:,int(NFFT//2)], 1)) ),                # (N, M)
            np.dot(np.sqrt(NFFT//2)      , np.conj(X[:, int(NFFT//2-1):0:-1]) )                               # (N, M-1)
        ])

        z =  np.real(np.fft.ifft(X, NFFT, 1))
        z = z[:, 0:self.L+1]                                  # Truncate the output signals

        return z

    def _spherical_noise_3d(self, P, tx_pos):
        phi, theta = tx_pos
        M = np.array(P).shape[0]                                                       #@ same as n_mics
        # print(M)                                                            
        NFFT = self.NFFT
        X = np.zeros( [self.n_mics, int(self.NFFT//2 +1)] , dtype='complex64')         # for complex analysis: add type or specify imaginary
        w = 2*Pi * self.Fs * np.linspace(0, 0.5, self.NFFT//2 + 1)

        for ph,th in zip(phi, theta):
            X_prime = jrandn([self.NFFT//2 + 1]) + 1j * jrandn([self.NFFT//2 + 1])
            X[0,:] = X[0,:] + X_prime
            v  = [[np.cos(th) * np.sin(ph)] , [np.sin(ph) * np.sin(th)], [ np.cos(ph)] ]                          #@

            for m in range(1, self.n_mics):                                         # mic index
                Delta = np.dot((np.transpose(v)) ,  (P[m,:] ))                     # @
                X[m,:] = X[m,:] + X_prime * np.exp(-1j*Delta*w/self.C) 
        
        X = X / np.sqrt(self.n_sources)

        NFFT = self.NFFT
        X = np.hstack([
            np.dot(np.sqrt(NFFT)         , np.real(  np.expand_dims(X[:,0], 1)  )),                          # (N, 1)
            np.dot(np.sqrt(int(NFFT//2)) , X[:,1:int(NFFT//2)] ),                                             # (N, M-1)
            np.dot(np.sqrt(NFFT)         , np.real(  np.expand_dims(X[:,int(NFFT//2)], 1)) ),                # (N, M)
            np.dot(np.sqrt(NFFT//2)      , np.conj(X[:, int(NFFT//2-1):0:-1]) )                               # (N, M-1)
        ])

        z =  np.real(np.fft.ifft(X, NFFT, 1))
        z = z[:, 0:self.L+1]    

        return z


    def _coherence(self, P, z, NFFT=256):
        d = alg.norm(P - P[0], axis=1)
        w = 2*Pi * self.Fs * np.linspace(0, 0.5, NFFT//2 + 1)

        sc_sim = np.zeros([self.n_mics-1, int(NFFT//2+1)])
        for m in range(0, self.n_mics-1):
            # [F,sc] = coherence( z[0, :-1].T, z[m+1,:-1].T, fs=8000, window=np.hanning(NFFT), noverlap=0.75*NFFT, nfft=NFFT)
            win = np.hanning(NFFT+2)
            F, pyx = csd(z[0, :-1].T, z[m+1,:-1].T, fs=8000, window=win[1:-1], noverlap=0.75*NFFT )
            F, pxx = csd(z[0, :-1].T, z[0, :-1].T, fs=8000, window=win[1:-1], noverlap=0.75*NFFT )
            F, pyy = csd(z[m+1,:-1].T, z[m+1,:-1].T, fs=8000, window=win[1:-1], noverlap=0.75*NFFT )

            sc = pyx/ np.sqrt(pxx * pyy)

            # sc, F = plt.cohere(z[0, :-1], z[m+1,:-1], Fs=8000, window=win[1:-1], noverlap=0.75*NFFT)
            # F = F / self.Fs * 2 * Pi
            sc_sim[m,:] = np.real(sc)
        
        sc_theory = np.zeros([self.n_mics-1, int(NFFT/2+1)])

        for m in range(0, self.n_mics-1):

            # for Cylinder
            # a = w * d[m+1]/self.C
            # sc_theory[m,:] = jv(0, a)

            # for sphere
            a = w * d[m+1]/self.C/Pi
            sc_theory[m,:] = np.sinc(a)


        return sc_sim, sc_theory, F


