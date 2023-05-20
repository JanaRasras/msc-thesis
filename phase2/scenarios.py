""" Scenarios for Phase 2 """

from phase1 import scenarios as sc1
from phase2.utilities import PartyGenerator, PartyGeneratorOne
import numpy as np

# Generate whatever combinations are required ..


# # 2 src
# all_sc_slow_15 = [

#  PartyGenerator(
#     2, 0.2, 15,np.arange(20,181,25), [0,5], True,
#     ds_folder=f"{sc1.ds_folder}/train",
#     asset_folder=f"{sc1.assets_folder}/noise_only",
#     output_folder='phase2/dataset/train/speed5'
# ),

#  PartyGenerator(
#     2, 0.2, 10,np.arange(20,181,25), [0,5], False,
#     ds_folder=f"{sc1.ds_folder}/train",
#     asset_folder=f"{sc1.assets_folder}/noise_only",
#     output_folder='phase2/dataset/train/speed5'
# ),

# PartyGenerator(
#   2, 0.2, 5,np.arange(20,181,25), [0,5], True,
#     ds_folder=f"{sc1.ds_folder}/train",
#     asset_folder=f"{sc1.assets_folder}/noise_only",
#     output_folder='phase2/dataset/train/speed5'
# ),

#  PartyGenerator(
#     2, 0.2, 15,np.arange(20,181,25), [5,5], False,
#     ds_folder=f"{sc1.ds_folder}/train",
#     asset_folder=f"{sc1.assets_folder}/noise_only",
#     output_folder='phase2/dataset/train/speed5'
# ),
# PartyGenerator(
#     2, 0.2, 10,np.arange(20,181,25), [5,5], False,
#     ds_folder=f"{sc1.ds_folder}/train",
#     asset_folder=f"{sc1.assets_folder}/noise_only",
#     output_folder='phase2/dataset/train/speed5'
# ),

# PartyGenerator(
#   2, 0.2, 5,np.arange(20,181,25), [5,5], True,
#     ds_folder=f"{sc1.ds_folder}/train",
#     asset_folder=f"{sc1.assets_folder}/noise_only",
#     output_folder='phase2/dataset/train/speed5'
# ),]

#  PartyGenerator(
#     2, 0.2, 15,np.arange(20,181,25), [15,15], True,
#     ds_folder=f"{sc1.ds_folder}/train",
#     asset_folder=f"{sc1.assets_folder}/noise_only",
#     output_folder='phase2/dataset/train/speed5'
# ),
# PartyGenerator(
#     2, 0.2, 10,np.arange(20,181,25), [15,15], True,
#     ds_folder=f"{sc1.ds_folder}/train",
#     asset_folder=f"{sc1.assets_folder}/noise_only",
#     output_folder='phase2/dataset/train/speed5'
# ),

# PartyGenerator(
#   2, 0.2, 5,np.arange(20,181,25), [15,15], False,
#     ds_folder=f"{sc1.ds_folder}/train",
#     asset_folder=f"{sc1.assets_folder}/noise_only",
#     output_folder='phase2/dataset/train/speed5'
# ),


#  PartyGenerator(
#     2, 0.2, 15,np.arange(20,181,25), [15,15], True,
#     ds_folder=f"{sc1.ds_folder}/train",
#     asset_folder=f"{sc1.assets_folder}/noise_only",
#     output_folder='phase2/dataset/train/speed5'
# ),
# PartyGenerator(
#     2, 0.2, 10,np.arange(20,181,25), [15,15], True,
#     ds_folder=f"{sc1.ds_folder}/train",
#     asset_folder=f"{sc1.assets_folder}/noise_only",
#     output_folder='phase2/dataset/train/speed5'
# ),

# PartyGenerator(
#   2, 0.2, 5,np.arange(20,181,25), [15,15], False,
#     ds_folder=f"{sc1.ds_folder}/train",
#     asset_folder=f"{sc1.assets_folder}/noise_only",
#     output_folder='phase2/dataset/train/speed5'
# ),


# ]


# # 1 src
# all_sc_slow_15= [


# PartyGeneratorOne(
#     1, 0.2, 15, None, [5], True,
#     ds_folder=f"{sc1.ds_folder}/train",
#     asset_folder=f"{sc1.assets_folder}/noise_only",
#     output_folder='phase2/dataset/train/speed5'
# ),
# PartyGeneratorOne(
#     1, 0.2, 10, None, [5], True,
#     ds_folder=f"{sc1.ds_folder}/train",
#     asset_folder=f"{sc1.assets_folder}/noise_only",
#     output_folder='phase2/dataset/train/speed5'
# ),
# PartyGeneratorOne(
#     1, 0.2, 5, None, [5], True,
#     ds_folder=f"{sc1.ds_folder}/train",
#     asset_folder=f"{sc1.assets_folder}/noise_only",
#     output_folder='phase2/dataset/train/speed5'
# ),
# ]




##### Testing

# 2 src
all_sc_slow_15 = [

    
 PartyGenerator(
    2, 0.2, 15, [60], [5,5], False,
    ds_folder=f"{sc1.ds_folder}/train",
    asset_folder=f"{sc1.assets_folder}/noise_only",
    output_folder='phase2/dataset/test/speed5'
),




 PartyGenerator(
    2, 0.2, 15, [60], [5,0], False,
    ds_folder=f"{sc1.ds_folder}/train",
    asset_folder=f"{sc1.assets_folder}/noise_only",
    output_folder='phase2/dataset/test/speed5'
),
 
  PartyGenerator(
    2, 0.2, 15, [60], [5,5], True,
    ds_folder=f"{sc1.ds_folder}/train",
    asset_folder=f"{sc1.assets_folder}/noise_only",
    output_folder='phase2/dataset/test/speed5'
),]



# # 1 src
# all_sc_slow_15 = [



# PartyGeneratorOne(
#     1, 0.2, 15, None, [15], False,
#     ds_folder=f"{sc1.ds_folder}/train",
#     asset_folder=f"{sc1.assets_folder}/noise_only",
#     output_folder='phase2/dataset/train/speed5'
# ),


# PartyGeneratorOne(
#     1, 0.2, 10, None, [15], True,
#     ds_folder=f"{sc1.ds_folder}/train",
#     asset_folder=f"{sc1.assets_folder}/noise_only",
#     output_folder='phase2/dataset/train/speed5'
# ),

# PartyGeneratorOne(
#     1, 0.2, 5, None, [15], True,
#     ds_folder=f"{sc1.ds_folder}/train",
#     asset_folder=f"{sc1.assets_folder}/noise_only",
#     output_folder='phase2/dataset/train/speed5'
# ),

# ]