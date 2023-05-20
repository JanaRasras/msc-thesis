""" 
Jana Rasras


Phase 1 of dataset generation for RIR-SL project. 
     Generate clean signals (wav & noise). {Noise generator is not merged}
     Calculate RIR (scenarios: python | wav: matlab) 
    output :
    directional wav signal (single speaker each)
 
    Phase 2:
        Create multi-source scenarios (mix & match)
         output :       
         multi_speaker+ noise in npz format
    
    Phase 3: Generate STFT frames
        - Train & test 

     Phase 4:   NN
"""

# Importing libraries
import os                       # for reading and writing files
from dataclasses import asdict  # cleaner way to store paramerts (autmatically write init)
from tqdm import tqdm
import numpy as np
import random as rn

from scipy.io import savemat # For phase1 (save wanted scenarios in .mat format, then genreate them using matlab)

import pytorch_lightning as pl  # pytorch lightning
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary  # a way to add extra steps to the main workflow

from phase1.audio_gen import generate_path, AudioGenerator
from phase2.audio_gen import generate_labels, generate_labelsOne, generate_noisy_mixture, generate_noisy_mixtureOne
from phase4.data import StftDoaDataModule
from phase4.model import NNetModule

from einops.layers.torch import Rearrange # reshaping the data inside the NN (this is as a layer, inside sequential)


def setup() -> None:
    " generate clean audio(mono) by concat multiple short files so the desired audio have a desired length(50s)"
    from phase1 import scenarios as sc

    generate_audio = AudioGenerator(f"{sc.assets_folder}/output")
    generate_audio(
        wav_folder=f"{sc.assets_folder}/wgn",
        t_sec=50
    )


def main_phase1() -> None:
    from phase1 import scenarios as sc

    wav_train_test = sorted([
        f"{sc.assets_folder}/output/{fname}"
        for fname in os.listdir(f"{sc.assets_folder}/wgn")
    ])                                                              # os.listdir read the files randomly. we sort them in order to have same list everytime

    # Create Training scenarios for RIR
    for room in sc.train_rooms:
        for mic in sc.train_microphones:
            for sp in sc.train_speakers:
                # Generate path
                arc = generate_path(sp, room)

                # Create Parameters
                rir_dict = {                    # dictnary
                    "room": asdict(room),       # dataclass(obj) to dic(string) = struct in matlab  # Matlab doesnt understant dataclass
                    "mic": asdict(mic),        # has relative position
                    "speaker": asdict(arc),    # will need sph2cart in radians
                    "clean_wav": rn.choice(wav_train_test[:21]),
                }

                # Save for MATLAB                                                           
                f_output = f"{room.folder}/{room.fname}_" \
                    + f"{mic.fname}_{sp.fname}_" \
                    + f"Wav{rir_dict['clean_wav'][-7:-4]}.mat"
                savemat(f_output, rir_dict, do_compression=True)


def main_pahse2() -> None:
    from phase1 import scenarios as sc1
    from phase2.scenarios import all_sc_slow_15

    for sc_slow_15 in all_sc_slow_15:
        for pt in tqdm(sc_slow_15):
            print(pt)
            signals = generate_noisy_mixture(pt)
            labels = generate_labels(None, pt)
            # signals = generate_noisy_mixtureOne(pt)   # this for generting audio with one speaker only 
            # labels = generate_labelsOne(None, pt)      # this for generting labels with one speaker only 


            f_output = f"{pt.folder}/{pt.f_output}".replace('.wav', '.npz')

            # input(f_output)
            # plt.plot(signals)
            # plt.ylabel('Amplitude [Hz]')
            # plt.xlabel('Samples')
            # # plt.title('2 White_NoiseDirectional moving sourecs ')

            # plt.show()
            # input('/')
            np.savez_compressed(f_output, x=signals, y=labels)  #npz



def main_phase3() -> None:
    """ Frame-based dataset. """
    from phase3 import scenarios as sc
    from phase3.frames_gen import plot_label
    from phase3.frames_gen import LabelFrameGenerator, generate_feature_frames

    generate_label_frames = LabelFrameGenerator(5)

    for fname in os.listdir(sc.te_ds_folder):

        # if fname == "Tr_ws_Room_rt0.200_snr15_Mic_x0.5y0.5_Src_phi000,120_rho1.000,1.000_omega005,005_Fwd,Fwd_Wav005,016.npz":
        #     continue

        data = np.load(f"{sc.te_ds_folder}/{fname}")
        try: # catch if an error happend when reading the files.
            x_frame = generate_feature_frames(data['x'], sc.param)
            y_frame = generate_label_frames(data['y'], sc.param)
            fname_new = fname[: -4]

            # print(x_frame, y_frame)
            f_output = f"phase3/dataset/speed140/test/{fname_new}_stft{sc.param.window_sec:.2f},{sc.param.freq_bins:d},{sc.param.hop_sec:.2f}.npz"
            # plot_label(data['x'], y_frame)

            # input(f_output)

            # plt.matshow(x_frame)
            # plt.show()
            np.savez_compressed(f_output, x=x_frame, y=y_frame)

        except:
            print("Err:", fname)

"""
    To see results type in CMD:>    tensorboard --logdir ./lightning_logs
"""

def main_phase_4a(
        model_type='rnn',
        ckpt_path=None,#"logs/lightning_logs/version_1/checkpoints/epoch=25-step=4750.ckpt", # None: training, else: testing
        seq_len=9, n_classes=37, batch_size=512,
        many_to_one=True, download_data=False, # downlad :true if the dataset is not on the computer and want to downdload it.
        device = 'cuda:0' , #  options: ['cuda:0','cpu'] # cpu is used for testing # cuda is not working for testing 
        ):
    """ NN for FNN """
    # Prepare trainer
    trainer = pl.Trainer(
        # resume_from_checkpoint=ckpt_path,
        max_epochs=25,
        auto_lr_find=True, # it takes a small batch of the DS and it automatically find the best learning rate. 
        default_root_dir='logs', # if it's not there, the code will generate one.
        log_every_n_steps=50,    # 
        gpus = None if device == 'cpu' else [0],
        # strategy='dp',
        # limit_train_batches=.1,
        callbacks=[
            LearningRateMonitor(logging_interval='step'),# each batch
            ModelSummary(max_depth=2), # print two levels(the models and a submodel inside it)
        ],
    )


    if ckpt_path is None:
        # choose dataset
        data = StftDoaDataModule('phase3/dataset/speed0_win0.9',
                                batch_size=batch_size, n_classes=n_classes,
                                n_steps=seq_len, many_to_one=many_to_one,
                                download=download_data)

        # Define model
        model = NNetModule(
            model_type,
            n_inputs=15*2, n_outputs=n_classes, n_steps=seq_len, learning_rate=0.0001, win_size=0.2, gpu=device) # n_inputs:fbins*2ratio&diff
    else:
        # choose dataset     # for testing only
        data = StftDoaDataModule('phase3/dataset/speed0_win0.9',
                             batch_size=batch_size, n_classes=n_classes,
                             n_steps=seq_len, many_to_one=many_to_one,
                             download=download_data)
        
        # Define model
        model = NNetModule.load_from_checkpoint(ckpt_path, win_size=0.2,  gpu=device)

    # Train / Test
    test_only = False if ckpt_path is None else True
    if not test_only:
        # trainer.tune(model, data)  # this is for the auto lr 
        trainer.fit(model, data)     # 
        ckpt_path = 'best'           # choose the best model in training and test on it

    trainer.test(model, datamodule=data)


if __name__ == "__main__":
    # setup()
    # main_phase_1()
    # main_pahse2()
    # main_phase3()
    main_phase_4a()


