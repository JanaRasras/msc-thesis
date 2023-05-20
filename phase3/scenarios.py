""" Scenarios for Phase 3 """

from phase1 import scenarios as sc1
from phase3.utilities import FrameParams


tr_ds_folder = "phase2/dataset/train/speed140"
te_ds_folder = "phase2/dataset/test/speed140"

param = FrameParams(0.2, 0.5, 15, fs=sc1.FS)
