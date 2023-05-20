
""" Generate frames for NN as (STFT, Labels).

"""

import torch    # to convert numpy to pytorch tensor             
from torch import nn # torch.nn. conv is used in label generation to match the nnaudio implemention of STFT that used conv
import numpy as np

from phase3.utilities import FrameParams, Spectrogram


def generate_feature_frames(x: np.ndarray, fp: FrameParams) -> np.ndarray:
    """ Create STFT features for a single file.
    """

    gen_stft = Spectrogram(fp, mode='AbsArgRatio') # this is wraper(a class to simplify nnaudio spectrogram class)
    features = gen_stft(
        torch.from_numpy(x.T).float()
    ).numpy()  # In phase3, try to keep all arrays as numpy not tensors (phase4 is for NN everything there is tensor)

    return features


class LabelFrameGenerator:
    """ Generate Frame labels for a single file. """

    def __init__(self, resolution: int = 5) -> None:
        self.d_phi = resolution           # degrees

    def __call__(self, y: np.ndarray, fp: FrameParams) -> np.ndarray:
        """ Generate Frame labels for a single file.
          """
        # Pad labels (as in STFT)
        padding = nn.ReflectionPad1d(fp.window_sa//2)  #copied from the torch.conv that is used inside nnaudio.STFT

        y = padding(
            torch.from_numpy(y.T).float()  # apply the padding thing on the labels(labels: numpy_> torch : tensor , so I converted to tensor of type flaot)
        ).numpy().T                        # padding needs row vector

        # STEP 1: Slice labels (into frames)
        W, K = fp.window_sa, fp.hop_sa  # to shorten the names
        F = int(1 + (len(y) - W) / K)   # nb of frames in the y list 

        frames = []
        for i in range(F):
            frames.append(y[i*K: i*K+W])     # split the long 1D y (in samples) into a list of frames(blocks)
        frames = np.stack(frames, axis=0)    # convert the list of frames into 2D numpy array    # (F, W, ...)

        # STEM 2: Calculate Mode (as Frame label)
        frame_labels, _ = torch.mode(           # conv take the avg, this take the most frequent
            torch.from_numpy(frames), dim=1     # it returns the most frequent values and its indecies(_: because indx is not required)
        )

        return frame_labels.numpy()


def plot_label(y, yf):

    from matplotlib import pyplot
    from numpy import ma  # the masked array library is used to ignore (mask) some elements in the array when processing it (e.g., during plotting)

    fig, (ax1, ax2) = pyplot.subplots(2, 1)

    ym = ma.masked_array(y, y == 999)       # to not plot silence
    ymf = ma.masked_array(yf, yf > 180)     # to not plot DOA outside [0, 180] range
    ax1.plot(ym)
    ax2.matshow(ymf, vmin=0, vmax=1, cmap='gray',
                origin='lower', aspect='auto')
    pyplot.show()
