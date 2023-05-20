""" Define low-level functions & classes used for RIR-SL project. 

    This file represent a way to deal with `configuration`:

"""

import os
from dataclasses import dataclass, field
import numpy as np
from numpy import pi as PI


@dataclass
class Room:
    """ Define a room class to store RIR parameters. """
    dim: list[float]
    rt60: float
    len: int                # N: len(h) = nb. samples
    folder: str
    order: int = -1
    C: int = 340            # defult val
    hop: int = 32           # air refresh rate [sa]

    fname: str = field(init=False, repr=False)

    def __post_init__(self):
        """ Create a room folder inside the dataset root directory. """   # derived val
        os.makedirs(self.folder, exist_ok=True)
        self.fname = f"Tr_Room_rt{self.rt60:.3f}"


@dataclass
class Speaker:
    """ Define path using spherical coordinates (phi, theta, rho). """
    phi_start: int      # DOA [deg]
    rho: float          # distance [m] b/w speaker & MicArray center
    speed: int          # angular velocity [deg/sec]
    is_rev: bool
    t_sec: int

    fs: int
    theta: int          # elevation [deg]
    len: int = field(init=False, repr=False)        # T*Fs: len(s)
    fname: str = field(init=False, repr=False)

    def __post_init__(self):
        self.len = int(self.t_sec * self.fs)
        self.fname = f"Src_phi{self.phi_start:03d}_rho{self.rho:.3f}_omega{self.speed:03d}_" \
            + ("Rev" if self.is_rev else "Fwd")


@dataclass
class Speakers:
    " used to generate all the indivual speaker class"
    phi_start: list[int]    # DOA [deg]
    phi_start_rev: list[int]
    rho: float              # distance [m] b/w speaker & MicArray center
    speed: list[int]        # angular velocity [deg/sec]
    are_reverse: list[bool]
    t_sec: int

    fs: int = 16000         # sample frequency [sa/sec]
    theta: int = 0          # elevation [deg]

    speaker_list: list[Speaker] = field(
        init=False, repr=False, default_factory=list)

    def __post_init__(self):
        """ Generate scenarios for speakers. """

        for omega in self.speed:
            for is_rev in self.are_reverse:

                phi_list = self.phi_start if not is_rev else self.phi_start_rev

                for phi_s in phi_list:
                    self.speaker_list.append(
                        Speaker(phi_s, self.rho, omega, is_rev, self.t_sec,
                                self.fs, self.theta)
                    )

    def __getitem__(self, idx: int):
        return self.speaker_list[idx]

    def __len__(self):
        return len(self.speaker_list)


@dataclass
class MicArray:
    len: int            # M: number of microphones
    rel_position: list[float]  # (x,y,z) w.r.t. room dimensions [m]
    distance: float     # microphone seperation [m]
    fname: str = field(init=False, repr=False)

    def __post_init__(self):
        """ Convert parameters to np.ndArray """
        self.rel_position = np.array(self.rel_position)
        self.fname = f"Mic_x{self.rel_position[0]:.1f}y{self.rel_position[1]:.1f}"


@dataclass
class FullArcPath:
    rho: float
    phi: np.ndarray
    theta: int










### This a simple example to show the difference between the class and dataclass
# class FullArchPath2:
#     def __init__(self, r, p, t) -> None:
#         self.rho = r
#         self.phi = p
#         self.theta = t

#     def __repr__(self) -> str:
        
#         return f"FullArchPath2(rho={self.rho})"

# if __name__== '__main__':
#     a = FullArcPath(1,1,1)
#     b = FullArchPath2(1,1,1)
#     print(a)
#     print(b)