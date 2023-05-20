""" Generate scenarios for RIR-SL project. """
from phase1.utilities import MicArray, Room, Speakers

# ::--------------------  PHASE I  -------------------- ::

ds_folder = "phase1/output_dataset"
assets_folder = "phase1/assets"

FS = 16000
T_SEC = 50


train_rooms = [
    Room([5, 4, 6], 0.2, 1024, folder=f"{ds_folder}/train/room1"),

]

train_speakers = Speakers(
    phi_start=range(0, 180-20, 5),
    phi_start_rev=range(180, 0+20, -5),
    rho=1.0,
    speed=[0, 5, 15,  45, 95, 140],
    are_reverse=[False, True],
    t_sec=50,
    fs=FS,
)
test_speakers = Speakers(
    phi_start=range(0, 180-20, 5),
    phi_start_rev=range(180, 0+20, -5),
    rho=1.0,
    speed=[0, 5,  15,  45, 95, 140],
    are_reverse=[False, True],
    t_sec=10,
    fs=FS,
)


train_microphones = [
    MicArray(2, [.5, .5, .5], .05)
]
test_microphones = [
    MicArray(2, [.5, .5, .5], .05)
]
