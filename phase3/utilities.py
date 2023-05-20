from dataclasses import dataclass, field
from matplotlib import colorbar

import torch
from torch import nn
from nnAudio.features.stft import STFT

import matplotlib.pyplot as plt


@dataclass
class FrameParams:
    """ Parameters used to convert samples to frames. """
    window_sec: float
    overlap: float = field(init=True, repr=False)
    freq_bins: int
    fs: int

    hop_sec: float = field(init=False, repr=True)
    window_sa: int = field(init=False, repr=True)
    hop_sa: int = field(init=False, repr=True)

    def __post_init__(self):
        self.hop_sec = self.window_sec * (1 - self.overlap)

        self.window_sa = int(self.window_sec * self.fs)
        self.hop_sa = int(self.hop_sec * self.fs)


class Spectrogram(nn.Module):
    """ Wrapper for STFT to allow MicRatio as features for M=2.(as postprocessing for nnaudio.stft) """

    def __init__(self, params: FrameParams, mode: str) -> None:
        """ Initialize NN Module. 
            Inputs: mode: possible values are (ReIm | AbsArgRatio)

            always use `output_format='Complex'` in STFT layer to have (Re,Im). i.e: mode is not for nnaudio output_format
            mode is used in forward()  as a postprocessing step to decide on whether to compute MicRatios
            as features or leave the output as ReIm.
        """
        super().__init__() # because its from nn.Module that does Gpu initalization first before creating the class.

        assert mode in ['ReIm', "AbsArgRatio"],  \
            "Error: unknown `mode` in Spectrogram, Use ReIm | AbsArgRatio"

        self.fp = params
        self.mode = mode            # To be used in forward
        self.stft = STFT(
            n_fft=self.fp.window_sa,
            freq_bins=self.fp.freq_bins,
            hop_length=self.fp.hop_sa,      # stride
            window='hann',
            freq_scale='linear',
            center=True,                    # STFT keneral
            pad_mode='reflect',
            sr=self.fp.fs,
            verbose=False,
            output_format="Complex"         # Always hard-coded
        )

        """
        self.stft2 = lambda x: torch.stft(  # pytorch also has stft that can be used, but nnaudio is used here (this part is just for comap. and it is not used)
            x,
            n_fft=self.fp.window_sa,
            hop_length=self.fp.hop_sa,      # stride
            window=torch.hann_window(self.fp.window_sa),
            center=True,                    # STFT keneral
            pad_mode='reflect',
            normalized=True,
            onesided=True,
            return_complex=False
        )
        """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Compute STFT of a two-channel signal w.r.t left channel. 

            Inputs  :    (N, M) #nb. samples, mics
            Outputs :    STFT_ReIm or STFT_Ratios 
        """

        if self.mode == "ReIm":
            output = self.stft(x)

        elif self.mode == "AbsArgRatio":
            eps = 1e-8
            complex_stft = self.stft(x)

            # Split the signal
            left_re = complex_stft[0, :, :, 0]# [left mic, all freq bins, all time win, real part] #
            left_im = complex_stft[0, :, :, 1] # [left mic, all freq bins, all time win, complex]

            right_re = complex_stft[1, :, :, 0]
            right_im = complex_stft[1, :, :, 1]

            # Calculate magnitude-ratios
            magnitude_ref = torch.sqrt(
                left_re.pow(2) + left_im.pow(2)
            )
            magnitude_right_ch = torch.sqrt(
                right_re.pow(2) + right_im.pow(2))

            magnitude_ratio = magnitude_right_ch / (magnitude_ref + eps)

            # Calculate Phase-difference
            phase_ref = torch.atan2(
                -left_im + 0.0,     # Fix: +0.0 removes -0.0 elements,
                left_re
            )
            phase_right_ch = torch.atan2(-right_im +
                                         0.0, right_re)

            phase_diffrence = phase_right_ch - phase_ref

            # Return output
            output = torch.cat([
                magnitude_ratio.unsqueeze(-1),# conver tthe vector (shape=[n,]) to 1D array(shape = [n,1],)
                phase_diffrence.unsqueeze(-1)
            ], dim=-1)


            

        else:
            raise ValueError(f"Unknown mode={self.mode}!")

            

        ## FOR Visulalizatiion
        from numpy import pi as PI
        a = output[:, :, 0]
        b = output[:, :, 1]
        # print(a.shape, b.shape, torch.min(a), torch.max(a), torch.min(b), torch.max(b))
        # fig, (ax1, ax2) = plt.subplots(2,1,  figsize=(12,9))

        # c = ax1.matshow(a, vmin=0, vmax=10, aspect=4)
        # d = ax2.matshow(b, vmin=-2*PI, vmax=2*PI, aspect=4)

        # fig.colorbar(c, ax = ax1)
        # fig.colorbar(d, ax = ax2)

        # ax1.set_title('Magnitude Ratio ')
        # ax1.set_ylabel('Frequency bins')
        # ax1.set_xlabel('Time frames')     

        # ax2.set_title('Phase Difference ')
        # ax2.set_ylabel('Frequency bins')
        # ax2.set_xlabel('Time frames')     

        # plt.tight_layout()
        # plt.show()
        return output
