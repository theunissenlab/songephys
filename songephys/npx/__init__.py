"""Neuropixels (SpikeGLX) support for songephys.

Recordings are acquired with SpikeGLX (imec probe + NI DAQ), preprocessed with
CatGT, sorted with Kilosort 4, and packaged into NWB files whose layout mirrors
the Intan-based songephys NWB files.

All times written to NWB are seconds on the NI (nidq) clock, measured along the
CatGT-concatenated (tcat) NI file. See NWB_SCHEMA_NPX.md for details.
"""
