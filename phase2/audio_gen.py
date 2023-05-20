""" Generate Clean mono speach using TIMIT dataset."""

import numpy as np

import soundfile as sf
from scipy.io import loadmat  # values are used for label

from phase1.utilities import FullArcPath
from phase2.utilities import PartyGenerator, PartyOfTow, PartyOfOne, PartyGeneratorOne


def generate_labels(arc, pt: PartyOfTow) -> np.ndarray:
    """ generate the labels from the the path """

    # -----------Read the files-------

    # A. audio
    srcA, fs = sf.read(f"{pt.srcA.folder}/{pt.srcA.fname}", always_2d=True)
    srcB, _ = sf.read(f"{pt.srcB.folder}/{pt.srcB.fname}", always_2d=True) # _ : to ignore fs
    noise, _ = sf.read(f"{pt.noise.folder}/{pt.noise.fname}", always_2d=True)
    noise = noise[:len(srcA), :]  # just to make sure that it has the same size          
    assert srcA.shape == srcB.shape == noise.shape, \
        "Error: In generate_labels(): expects all audio inputs to have same shape.."

    # B. Path (for DOA)
    pathA_obj = loadmat(
        f"{pt.srcA.folder}/{pt.srcA.fname}".replace('.wav', '.mat'),
        squeeze_me=True,
        struct_as_record=False
    )
    pathA = FullArcPath(
        pathA_obj['speaker'].rho,
        pathA_obj['speaker'].phi[:, None], # expand (unsqueeze) volumn vector 
        pathA_obj['speaker'].theta
    )

    pathB_obj = loadmat(
        f"{pt.srcB.folder}/{pt.srcB.fname}".replace('.wav', '.mat'),
        squeeze_me=True,
        struct_as_record=False
    )
    pathB = FullArcPath(
        pathB_obj['speaker'].rho,
        pathB_obj['speaker'].phi[:, None],
        pathB_obj['speaker'].theta
    )

    # Generate Labels
    # a. quantize
    bounds = np.arange(0+5/2, 180-5/2+1, 5)
    phiA_q = np.digitize(np.copy(pathA.phi), bounds)*5
    phiB_q = np.digitize(np.copy(pathB.phi), bounds)*5
    # print(pathA.phi[40000:40010], phiA_q[40000:40010])

    # b. replace the silence with 999
    amp = max(np.max(srcA), np.max(srcB))  # normalize before threshold


    ## when there is silence and we want to deal with it.
    # maskA = np.all(abs(srcA/amp) <= sn.threshold, axis=1, keepdims=True)
    # maskB = np.all(abs(srcB/amp) <= sn.threshold, axis=1, keepdims=True)
    # print(pathA.phi.shape, maskA.shape)

    # phiA_q[maskA] = 999
    # phiB_q[maskB] = 999

    labels = np.hstack([phiA_q, phiB_q])
    # print(pathA.phi[80000], pathB.phi[80000])
    return labels


def generate_noisy_mixture(pt: PartyOfTow) -> np.ndarray:
    "Run the part and combine the 2 srcs with the noise"

    srcA, fs = sf.read(f"{pt.srcA.folder}/{pt.srcA.fname}", always_2d=True)
    srcB, _ = sf.read(f"{pt.srcB.folder}/{pt.srcB.fname}", always_2d=True)
    noise, _ = sf.read(f"{pt.noise.folder}/{pt.noise.fname}", always_2d=True)
    noise = noise[:len(srcA), :]            
    assert srcA.shape == srcB.shape == noise.shape, \
        "Error: generate_labels() expects all audio inputs to have same shape!"

    # Used to rormalize
    amp = max(np.max(srcA), np.max(srcB))
    mixture = (srcA + srcB) / amp

    # Normalize by left RMS
    ls_rms = np.sqrt(sum(np.abs(mixture[:, 0]**2))/len(mixture[:, 0]))
    mixture /= ls_rms

    if pt.noise.desired_snr is not None:
        snr = 10 ** (pt.noise.desired_snr/20)

        ln_rms = np.sqrt(sum(np.abs(noise[:, 0]**2))/len(noise[:, 0]))
        noise /= ln_rms

        noisy_mixture = mixture + 1/snr * noise

    else:
        noisy_mixture = mixture
    # make sure that the signal is  in [-1:1]
    amp = np.max(np.abs(noisy_mixture))
    noisy_mixture /= amp

    return noisy_mixture



def generate_labelsOne(arc, pt: PartyOfOne) -> np.ndarray:
    """ generate the labels from the the path 

        for one speaker
    
    """

    # -- Read the files
    # A. audio (for silence)
    srcA, fs = sf.read(f"{pt.srcA.folder}/{pt.srcA.fname}", always_2d=True)
    noise, _ = sf.read(f"{pt.noise.folder}/{pt.noise.fname}", always_2d=True)
    noise = noise[:len(srcA), :]            
    # assert srcA.shape == srcB.shape == noise.shape, \
    #     "Error: generate_labels() expects all audio inputs to have same shape!"

    # B. Path (for DOA)
    pathA_obj = loadmat(
        f"{pt.srcA.folder}/{pt.srcA.fname}".replace('.wav', '.mat'),
        squeeze_me=True,
        struct_as_record=False
    )
    pathA = FullArcPath(
        pathA_obj['speaker'].rho,
        pathA_obj['speaker'].phi[:, None],
        pathA_obj['speaker'].theta
    )

    # Generate Labels
    # a. quantize
    bounds = np.arange(0+5/2, 180-5/2+1, 5)
    phiA_q = np.digitize(np.copy(pathA.phi), bounds)*5

    amp = np.max(srcA)  # normalize before threshold

    # maskA = np.all(abs(srcA/amp) <= sn.threshold, axis=1, keepdims=True)
    # maskB = np.all(abs(srcB/amp) <= sn.threshold, axis=1, keepdims=True)
    # print(pathA.phi.shape, maskA.shape)

    # phiA_q[maskA] = 999
    # phiB_q[maskB] = 999

    labels = np.hstack([phiA_q])
    # print(pathA.phi[80000], pathB.phi[80000])
    return labels


def generate_noisy_mixtureOne(pt: PartyOfOne) -> np.ndarray:
    "Run the part and combine the 2 srcs with the noise"

    srcA, fs = sf.read(f"{pt.srcA.folder}/{pt.srcA.fname}", always_2d=True)
    # srcB, _ = sf.read(f"{pt.srcB.folder}/{pt.srcB.fname}", always_2d=True)
    noise, _ = sf.read(f"{pt.noise.folder}/{pt.noise.fname}", always_2d=True)
    noise = noise[:len(srcA), :]            # TODO: check noise length files.
    # assert srcA.shape == srcB.shape == noise.shape, \
    #     "Error: generate_labels() expects all audio inputs to have same shape!"

    # Used to rormalize
    amp = np.max(srcA)
    mixture = (srcA) / amp

    # Normalize by left RMS
    ls_rms = np.sqrt(sum(np.abs(mixture[:, 0]**2))/len(mixture[:, 0]))
    mixture /= ls_rms

    if pt.noise.desired_snr is not None:
        snr = 10 ** (pt.noise.desired_snr/20)

        ln_rms = np.sqrt(sum(np.abs(noise[:, 0]**2))/len(noise[:, 0]))
        noise /= ln_rms

        noisy_mixture = mixture + 1/snr * noise

    else:
        noisy_mixture = mixture
    # make sure that the signal is  in [-1:1]
    amp = np.max(np.abs(noisy_mixture))
    noisy_mixture /= amp

    return noisy_mixture
