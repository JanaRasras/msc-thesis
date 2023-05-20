import os
import random as rn
from dataclasses import dataclass, field # feild is a placeholder for varibles (define but dont assign a value)

## Defining the varibles

@ dataclass
class ScenarioFile:
    """ Decode file name string for single clean source into Dataclass. """
    folder: str         # e.g.: dataset/train/roomX
    fname: str

    rt60: float = field(init=False, repr=False)     # init: not required when creating an instance, repre: doesnt appear when printing the class
    phi_start: int = field(init=False, repr=False)  # [deg]
    rho: float = field(init=False, repr=False)
    speed: int = field(init=False, repr=False)      # [deg/sec]
    is_rev: bool = field(init=False, repr=False)
    mic_pos: list[float] = field(init=False, repr=False)
    wav_id: int = field(init=False, repr=False)

    def __post_init__(self):
        """ Parse the file name string. 

        Example:
            Tr_Room_rt0.200_Mic_x0.5y0.5_Src_phi150_rho1.000_omega015_Fwd_Wav015.wav
        """
        specs = self.fname.find("rt")  # temporary value
        self.rt60 = float(self.fname[specs+2:specs+7])

        specs = self.fname.find("Mic_x")
        self.mic_pos = [
            float(self.fname[specs+5:specs+8]),
            float(self.fname[specs+9:specs+12])
        ]

        specs = self.fname.find("phi")
        self.phi_start = int(self.fname[specs+3:specs+6])

        specs = self.fname.find("rho")
        self.rho = float(self.fname[specs+3:specs+6])

        specs = self.fname.find("omega")
        self.speed = int(self.fname[specs+5:specs+8])

        self.is_rev = True if "Rev" in self.fname else False

        specs = self.fname.find("Wav")
        self.wav_id = int(self.fname[specs+3:specs+6])


@dataclass
class NoiseFile:
    folder: str         # e.g.: assets/noise_only
    fname: str

    desired_snr: int    # in dB (after normalization)


@dataclass
class PartyOfOne:
    """ A party of two speakers under noise. """
    srcA: ScenarioFile = field(init=True, repr=False)
    noise: NoiseFile = field(init=True, repr=False)

    folder: str = field(init=True, repr=True)
    f_output: str = field(init=False, repr=True)
    # srcB: ScenarioFile = srcA

    def __post_init__(self):
        """ Create output file name for one party. 

        Example:
            Tr_Room_rt0.200_snr10_Mic_x0.5y0.5_Src_phi030,150_rho1.000,0.500_omega,005,015_Fwd,Rev_Wav015,010.wav
        """
        # assert self.srcA.rt60 == self.srcB.rt60, \
        #     "Error: all speakers should have same room parameters!"
        snr = 'Inf' if self.noise.desired_snr is None else f"{self.noise.desired_snr:02d}"
        self.f_output = f"Tr_Room_rt{self.srcA.rt60:.3f}_"\
            + f"snr{snr}_" \
            + f"Mic_x{self.srcA.mic_pos[0]:.1f}y{self.srcA.mic_pos[1]:.1f}_" \
            + f"Src_phi{self.srcA.phi_start:03d}_" \
            + f"rho{self.srcA.rho:.3f}_" \
            + f"omega{self.srcA.speed:03d}_" \
            + ("Rev" if self.srcA.is_rev else "Fwd") + "_" \
            + f"Wav{self.srcA.wav_id:03d}.wav"


@dataclass
class PartyOfTow:
    """ A party of two speakers under noise. """
    srcA: ScenarioFile = field(init=True, repr=False)
    srcB: ScenarioFile = field(init=True, repr=False)
    noise: NoiseFile = field(init=True, repr=False)

    folder: str = field(init=True, repr=True)
    f_output: str = field(init=False, repr=True)

    def __post_init__(self):
        """ Create output file name for one party. 

        Example:
            Tr_Room_rt0.200_snr10_Mic_x0.5y0.5_Src_phi030,150_rho1.000,0.500_omega,005,015_Fwd,Rev_Wav015,010.wav
        """
        assert self.srcA.rt60 == self.srcB.rt60, \
            "Error: all speakers should have same room parameters!"
        snr = 'Inf' if self.noise.desired_snr is None else f"{self.noise.desired_snr:02d}"
        self.f_output = f"Tr_Room_rt{self.srcA.rt60:.3f}_"\
            + f"snr{snr}_" \
            + f"Mic_x{self.srcA.mic_pos[0]:.1f}y{self.srcA.mic_pos[1]:.1f}_" \
            + f"Src_phi{self.srcA.phi_start:03d},{self.srcB.phi_start:03d}_" \
            + f"rho{self.srcA.rho:.3f},{self.srcB.rho:.3f}_" \
            + f"omega{self.srcA.speed:03d},{self.srcB.speed:03d}_" \
            + ("Rev" if self.srcA.is_rev else "Fwd") + "," \
            + ("Rev" if self.srcB.is_rev else "Fwd") + "_" \
            + f"Wav{self.srcA.wav_id:03d},{self.srcB.wav_id:03d}.wav"

        # self.folder = self.srcA.folder

    @property
    def is_duplicate(self):
        """ Identify `Static` cases.

            Is used to remove dumplicates since `Forward` & `Reverse` 
            is not applicable 
            i.e.,
                DOA : [0-20] + [25-155] + [160-180]
                Note: Unique   Duplicate   Unique
                Keep:  Fwd        Fwd        Rev  
        """
        speed_match = (self.srcA.speed == self.srcB.speed == 0)
        doa_match = (
            25 <= min(self.srcA.phi_start, self.srcB.phi_start)
            and max(self.srcA.phi_start, self.srcB.phi_start) <= 155
        )
        direction_match = (not self.srcA.is_rev) and (not self.srcB.is_rev)

        return speed_match and doa_match and (not direction_match)


@dataclass
class PartyGenerator:
    """ Generate scenarios for `Party`. """
    n_speakers: int
    rt60: float
    snr: int
    phi_desired: list[int]
    speed_desired: list[int]
    same_dir: bool

    ds_folder: str              # contains dataset for train/test
    asset_folder: str           # Contains noise files
    output_folder: str          # use as scenario name

    party_list: list[PartyOfTow] = field(
        init=False, repr=False, default_factory=list)

    def __post_init__(self):
        """ Loop over all possible cases and generate Speaker() class for each one. """
        # create output folder
        os.makedirs(self.output_folder, exist_ok=True)

        #
        rooms = self._rt60_to_room(self.rt60)
        # Vmin, Vmax = min(self.speed_desired), max(self.speed_desired)

        for room in rooms:

            # Get files (single source)
            all_files = [
                ScenarioFile(f"{self.ds_folder}/{room}", fname)
                for fname in sorted(os.listdir(f"{self.ds_folder}/{room}"))
                if fname.endswith(".wav")
            ]

            all_noise = os.listdir(self.asset_folder)

            # Main code
            while len(all_files) > 1:
                # Remove srcA from the list
                srcA: ScenarioFile = all_files.pop(0)

                # 1) check srcA speed
                # if not (Vmin <= srcA.speed <= Vmax):
                if srcA.speed not in self.speed_desired:
                    continue

                # 2.a) find a match as srcB
                # TODO: loop over other guests: srcB, srcC, ...n_speakers)
                for srcB in all_files:
                    if self._is_valid_match(srcA, srcB):
                        # 2.B) select noise as srcX
                        srcX = NoiseFile(self.asset_folder,
                                         rn.choice(all_noise), self.snr)

                        # 3) Store as party as (srcs & noise)
                        self.party_list.append(
                            PartyOfTow(srcA, srcB, noise=srcX,
                                       folder=self.output_folder)
                        )

        # Remove Duplicates
        self.party_list = [
            party 
            for party in self.party_list 
            if not party.is_duplicate]

    def __getitem__(self, idx: int) -> PartyOfTow:
        return self.party_list[idx]

    def __len__(self):
        return len(self.party_list)

    def _rt60_to_room(self, rt60):
        """ Select room folder based on RT60 value. """
        folder = []
        spec = f"{rt60:.3f}"
        for f_room in os.listdir(self.ds_folder):
            # Check for invalid folder
            if "room" not in f_room:
                continue

            # Check for empty folder
            all_files = os.listdir(f"{self.ds_folder}/{f_room}")
            if len(all_files) == 0:
                continue

            # Check RT in first file
            if spec in all_files[0]:
                folder.append(f_room)

        return folder

    def _is_valid_match(self, srcA, srcB):
        """ Check if Speaker match specification.
        """
        # Vmin, Vmax = min(self.speed_desired), max(self.speed_desired)
        # speed_match = Vmin <= srcB.speed <= Vmax
        speed_match = srcB.speed in self.speed_desired
        phi_match = abs(srcB.phi_start - srcA.phi_start) in self.phi_desired

        direction_match = (srcA.is_rev == srcB.is_rev) & self.same_dir \
            or (srcA.is_rev != srcB.is_rev) & (not self.same_dir)

        return (speed_match and phi_match and direction_match)


@dataclass
class PartyGeneratorOne:
    """ Generate scenarios for `Party`. """
    n_speakers: int
    rt60: float
    snr: int
    phi_desired: list[int]      #
    speed_desired: list[int]
    same_dir: bool              # 

    ds_folder: str              # contains dataset for train/test
    asset_folder: str           # Contains noise files
    output_folder: str          # use as scenario name

    party_list: list[PartyOfOne] = field(
        init=False, repr=False, default_factory=list)

    def __post_init__(self):
        """ Loop over all possible cases and generate Speaker() class for each one. """
        # create output folder
        os.makedirs(self.output_folder, exist_ok=True)

        #
        rooms = self._rt60_to_room(self.rt60)

        for room in rooms:

            # Get files (single source)
            all_files = [
                ScenarioFile(f"{self.ds_folder}/{room}", fname)
                for fname in sorted(os.listdir(f"{self.ds_folder}/{room}"))
                if fname.endswith(".wav")
            ]

            all_noise = os.listdir(self.asset_folder)

            # Main code
            while len(all_files) > 1:
                # Remove srcA from the list
                srcA: ScenarioFile = all_files.pop(0)

                # 1) check srcA speed
                if srcA.speed not in self.speed_desired:
                    continue
                srcX = NoiseFile(self.asset_folder,
                                 rn.choice(all_noise), self.snr)

                # 3) Store as party as (srcs & noise)
                self.party_list.append(
                    PartyOfOne(srcA,  noise=srcX,
                               folder=self.output_folder)
                )

        # Remove Duplicates
        self.party_list = [
            party 
            for party in self.party_list 
            if 'Fwd' in party.f_output
        ] + self.party_list[-5:]

    def __getitem__(self, idx: int) -> PartyOfOne:
        return self.party_list[idx]

    def __len__(self):
        return len(self.party_list)

    def _rt60_to_room(self, rt60):
        """ Select room folder based on RT60 value. """
        folder = []
        spec = f"{rt60:.3f}"
        for f_room in os.listdir(self.ds_folder):
            # Check for valid folder
            if "room" not in f_room:
                continue

            # Check RT in first file
            all_files = os.listdir(f"{self.ds_folder}/{f_room}")
            if len(all_files) == 0:
                continue

            if spec in all_files[0]:
                folder.append(f_room)

        return folder
