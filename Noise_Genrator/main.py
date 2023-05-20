"""
Jana Rasras
Noise Generator
June 2021
"""

'''
Note
1. uncomment section (Cindf_1D line[31,32], Cindf_3D, Sindf_1D, Sindf_3D line[43,44]) you want to run and comment other sections.[Follwing sec: ## Generate sensor signals]
2. To run the spherical part, in utils.py file line[232,233] should be uncomment.| and for Cylinderical line[228,229] should be uncoomented and line[232,233] should be commented from the same file.
'''
import csv

from matplotlib import pyplot
from utils import NoiseGenerator
import soundfile as sf
import numpy as np

## Converting to X, Y, Z
def sph2cart(phi, theta, r):
    """ Convert spherical(in deg) to cartesian. """
    theta *= np.pi/180          # from xy to z
    phi *= np.pi/180            # from x to y

    x = r * np.cos(theta) * np.cos(phi)
    y = r * np.cos(theta) * np.sin(phi)
    z = r * np.sin(theta)

    return np.array([x, y, z])

def main():
    
    # Parameters
    c = 340                                         # Speed of sound
    fs = 16000                                       # Sample frequency
    NFFT = 256                                      # Number of frequency bins (for analysis)
    L = 137549                                       # Data length

    # r = [[2, 1.5, 2], [2.1, 1.5, 2]]                # P = Receiver Absolute position [x y z] (m) per mic (row)
    s = None
    N_phi = 64                                      # Number of noise sources (rather than there positions)
    N = 256                                         # for Sinf_3D

    count = 0
    ## Read params & let x = wav
    reader = csv.reader(
                    open("configs_without_SNR.csv"), 
                    delimiter=";"
                )

    csv_header = next(reader)                                           # Assume CSV has a header
    for line in reader:
        count += 1                                                      # for numbering the noise files
        # 1. parse params
        room_dim, RT, rx_rel_pos, d, rx_rot, r, phi, snr = list(map(eval, line))

        center_mic = np.array(room_dim) * np.array(rx_rel_pos)    
        rx_pos = np.array([center_mic - sph2cart(rx_rot, 0, d/2), center_mic + sph2cart(rx_rot, 0, d/2)])
        r = rx_pos
        ## Generate sensor signals
        ## cinf_1D
        #cinf_1D = NoiseGenerator(L=L, n_mics=len(r), n_sources=N_phi, C=c, Fs=fs, mode='C1D')   
        # sc_sim, sc_theory, F = cinf_1D(rx_pos=r, tx_pos=s) 
                                
        ## cinf_3D                                                                        
        # cinf_3D = NoiseGenerator(L=L, n_mics=len(r), n_sources=N_phi, C=c, Fs=fs, mode='C3D')   
        # sc_sim, sc_theory, F = cinf_3D(rx_pos=r, tx_pos=s)  
        
        ## sinf_1D
        #sinf_1D = NoiseGenerator(L=L, n_mics=len(r), n_sources=N_phi, C=c, Fs=fs, mode='S1D')   
        # sc_sim, sc_theory, F = sinf_1D(rx_pos=r, tx_pos=s) 
        
        ## sinf_3D
        sinf_3D = NoiseGenerator(L=L, n_mics=len(r), n_sources=N_phi, C=c, Fs=fs, mode='S3D')   
        # sc_sim, sc_theory, F = sinf_3D(rx_pos=r, tx_pos=s)            ## For Plotting the Coherence


        #### For noise Generation
        noise = sinf_3D(rx_pos=r, tx_pos=s) 
        #noise = (noise-np.min(noise))/(np.max(noise)-np.min(noise))

        # Plot the results
        # fig, ax = pyplot.subplots(1,1)
        # ax.plot(F/1000, sc_sim[0])
        # ax.plot(F/1000,sc_theory[0]) 

        # fig2, ax2 = pyplot.subplots(1,1)
        # ax2.plot(F/1000, sc_sim[1])
        # ax2.plot(F/1000,sc_theory[1]) 
        # pyplot.show()

        # Save noise file
        
        source_info = f'Noise{phi:03d}_'
        if rx_rel_pos == [.5,.5,.5]: source_info += 'center'
        elif rx_rel_pos == [.1,.1,.5]: source_info += 'corner'
        else: source_info += 'edge'
        source_info += f"{rx_rot}_d{d}"
        
        fname = f"DS/{source_info}.wav"
        sf.write(fname, noise, fs) 

if __name__ == '__main__':
    main()
