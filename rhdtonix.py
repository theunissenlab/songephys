import os
#import quantities as pq
#from zeebeez.tdt2neo import stim
#from neo.core import Epoch
from neo.io import RHDIO, NixIO
#from neo.io.tools import convert_digital_to_metadata
#from neosound import sound_manager
#from neosound.sound_store import HDF5Store
import sys

      
# Check to see if RHD file exists.
experiment_file = sys.argv[1]
if not(os.path.isfile(experiment_file)):
    print 'Error:', experiment_file, 'does not exist'
    sys.exit()
    
# Check to see if NIX file exists.
nix_filename = sys.argv[2]
if os.path.isfile(nix_filename):
   print 'Error:', nix_filename, 'exists'
   sys.exit()
    
# Read RHD file.   
rhd_in = RHDIO(experiment_file)
block = rhd_in.read_block()

# stimulus_nix_filename = os.path.join(experiment_dir,
#                                    os.path.basename(
#                                        experiment_file).replace(".rhd", "_stimuli.h5"))
# if stimulus_file is not None:
#     csv_reader = stim.StimCSVReader()
#
#     # Read all of the metadata and trial data
#     csv_reader.read(stimulus_file, root_dir=stimulus_dir)
#     csv_reader.write(stimulus_nix_filename)
#
#     write_playback_stim_to_nix(block, stimulus_nix_filename)
    
# Write NIX File  
nix_out = NixIO(nix_filename)
nix_out.write_block(block)

# print 'Block path is', block.hdf5_path
nix_out.close()
