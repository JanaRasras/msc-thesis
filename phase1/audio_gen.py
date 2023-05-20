""" Generate Clean mono speach using TIMIT dataset.

    It supports adding silence with:
    - random length  : based on Exponential Distribution
    - random interval: based on TIMIT file length
"""
import os
import numpy as np
import random as rn
import soundfile as sf

from phase1.utilities import FullArcPath, Room, Speaker


class AudioGenerator:
    """ Generate clean audio in specified directory. """

    def __init__(self, output_folder: str) -> None:
        os.makedirs(output_folder, exist_ok=True)
        self.output_folder = output_folder

    def __call__(self, wav_folder: str, t_sec: int):
        """ """
        fs = 16000
        threshold = 0.01                # select a threshold to remove the silence
        nb_files = 500 + 50             # for training & testing {...}

        # Process folder
        all_wav = os.listdir(wav_folder)

        # Generate many files
        for i in range(nb_files):

            audio = np.zeros(shape=[1, 1])
            done = False

            # Construct one file
            while not done:

                # load random files
                f_wav = rn.choice(all_wav)

                wav, fs1 = sf.read(f"{wav_folder}/{f_wav}", always_2d=True)

                # pre-process TIMIT
                wav /= np.max(np.abs(wav))        # normalize
                wav = wav[np.abs(wav) > threshold][:, None]    # remove silence

                # Concate
                # audio = np.vstack([audio, wav, silence])
                # TIMIT, No pauses and No generted silence addded.
                audio = np.vstack([audio, wav])  # 800000*1

                if len(audio) > t_sec * fs:  # 50*16000
                    done = True

            sf.write(
                f"{self.output_folder}/clean_audio_{i:03d}.wav",
                audio[1: 1+int(t_sec*fs), :],   # from1 to remove the zero sample we added at the begining
                fs
            )


def generate_path(sp: Speaker, room: Room) -> FullArcPath:
    """ Generate `Arc` path of a Speaker in `spherical` coordinates. """
    # Static
    if sp.speed == 0:
        sp_path = sp.phi_start * np.ones(shape=[sp.len, 1])
        return FullArcPath(sp.rho, sp_path, sp.theta)

    # Moving
    phi = sp.phi_start
    d_phi = sp.speed * (room.hop/sp.fs)   # dist[deg] = speed[deg/s] * time[s] # how many deg per sample the speaker moves

    sp_path = np.zeros(shape=[sp.len, 1])
    for n in range(0, sp.len, room.hop):
        sp_path[n:n+room.hop] = phi

        # Reset
        if not sp.is_rev:
            phi += d_phi
            if phi > 180:
                phi = 0
        else:
            phi -= d_phi
            if phi < 0:
                phi = 180

    return FullArcPath(sp.rho, sp_path, sp.theta)
