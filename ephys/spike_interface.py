import spikeinterface.full as si  # Flat import (full library)
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# global_job_kwargs = dict(n_jobs=4, chunk_duration="1s")
# si.set_global_job_kwargs(**global_job_kwargs)

session_id = '007_2024-06-25_15-54-23'
base_folder = Path() / 'D:'
oe_folder = base_folder / session_id

# IBL destripping
rec = si.read_openephys(folder_path=oe_folder, stream_id='0')
rec = si.highpass_filter(recording=rec)
rec = si.phase_shift(recording=rec)
bad_channel_ids = si.detect_bad_channels(recording=rec)
# rec = si.interpolate_bad_channels(recording=rec, bad_channel_ids=bad_channel_ids[0])
rec = si.highpass_spatial_filter(recording=rec)













# References
# Preprocessing module - How to implement “IBL destriping” or “SpikeGLX CatGT” in SpikeInterface
# (https://spikeinterface.readthedocs.io/en/stable/modules/preprocessing.html#ibl-destripe)

# Analyze Neuropixels datasets
# (https://spikeinterface.readthedocs.io/en/stable/how_to/analyze_neuropixels.html)




