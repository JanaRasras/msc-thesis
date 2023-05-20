import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage as ni


class OneHotEncode:
    """ Encode Targets according to `nn.LossFunction` for C classes. """

    def __init__(self, n_classes: int):
        self.n_classes = n_classes

    def __call__(self, labels: torch.Tensor) -> torch.Tensor:
        """ Expect `index` as labels with silence_label=`n_classes`. """

        # Temporarly encode silence then remove it
        temp = F.one_hot(labels, num_classes=self.n_classes + 1)
        output = torch.any(temp[:, :, :-1], dim=1)

        # Has to return float() for criterion
        return output.long()


class TargetEncoderForMCML:
    """ Encode Targets according to `TorchMetric` for upto S-speakers. """

    def __init__(self, n_classes: int, n_labels: int, resolution: int, silence_label: int):
        self.n_classes = n_classes          # nb. of possible DOAs
        self.n_label = n_labels             # max nb. of active Speakers
        self.resolution = resolution
        self.silence_label = silence_label

    def __call__(self, target: torch.Tensor) -> torch.Tensor:
        """ Convert targets into index  (type: `int`) 
        """
        # Find silene
        mask = target == self.silence_label

        # Encode `classes` (including silence-class)
        index = torch.div(target, self.resolution,
                          rounding_mode='trunc')
        index[mask] = self.n_classes

        # Pad `labels` up to max nb. of speakers
        S = index.shape[1]
        S_pad = self.n_label - S

        if S_pad != 0:
            output = F.pad(index, (0, S_pad), mode='constant',
                           value=self.n_classes)

        # Sort speakers
        output, _ = torch.sort(output, dim=-1)

        # Has to return long() for the metrics
        return output.long()


class PredictionEncoderForMCML:
    """ Encode Predictions according to `TorchMetric` for upto S-speakers. """

    def __init__(self, n_classes: int, n_labels: int, threshold: float):
        self.n_classes = n_classes
        self.n_labels = n_labels    # # Max nb. of speakers
        self.threshold = threshold

    def __call__(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """ Convert predictions into index (type: `int`).

        """
        # Find silene & Binarize
        peaks, loc = torch.topk(predictions, k=self.n_labels, dim=-1)
        silence_mask = torch.sigmoid(peaks) < self.threshold
        loc[silence_mask] = self.n_classes

        # Sort & Aligh prediction to targets
        temp, _ = torch.sort(loc, dim=-1)
        output = torch.zeros_like(temp) + self.n_classes

        for i, (rp, rl) in enumerate(zip(temp, labels)):
            idx = list(range(self.n_labels))

            for cp in rp:
                if cp in rl:
                    output[i, idx.pop(0)] = cp      # pop(pos in rl)
                else:
                    output[i, idx.pop(-1)] = cp     # check(neighbours)


        # Has to return long() for the metrics
        return output.long()


class Percentile:
    def __init__(self, p=.5, len_sec=1, frame_sec=0.1, update_sec=0.05) -> None:
        self.p = p
        self.len_sec = len_sec
        self.frame_sec = frame_sec
        self.update_sec = update_sec

    def __call__(self, x_data: torch.Tensor):
        """ Return the un-normalized filtered signal."""
        x_data = x_data.cpu().numpy()

        # Calculate nb. past samples based on update_sec
        L = 1 + np.floor((self.len_sec - self.frame_sec) /
                         self.update_sec).astype('int')
        K = L // 2            # (L//2-1)

        origin = (K, 0)                 # (right, center) = causal 2D ----
        kernel = np.ones((2*K+1, 1))   # this way it is always odd

        y_data = ni.percentile_filter(
            x_data, percentile=self.p*100, footprint=kernel, origin=origin, mode='constant', cval=0)

        # Remove Transient while keep dim (N, DOA)
        output = np.zeros_like(x_data)
        # in matlab output(1:end-k) = y_data(k:end)
        output[:-K] = y_data[K:]

        return torch.from_numpy(output).long()


# [1 0 0 1 0]    #[0.9 0.1 0 0]
                # [1 0 0 0]
# [0 15]       #[999 0]
# [0 3]        # [37 0]