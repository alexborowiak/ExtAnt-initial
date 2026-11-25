"""Where the data are on JASMIN. Every notebook takes its folders from here, so moving a folder is one change.

The notebook test runner (tests/run_notebooks.py) points SCRATCH_DIR at a temporary folder.
"""

from pathlib import Path

#(t): JASMIN's scratch: notebook calc_01's seasonal means (seasonal/) and every notebook's results (results/)
SCRATCH_DIR = Path('/work/scratch-pw4/aborow/ExtAnt')

#(t): The monthly LESFMIP stores notebook calc_01 makes (section 1); calc_04 reads the sea-ice ones
MONTHLY_DIR = Path('/work/scratch-pw4/aborow/LESFMIP/monthly_v2')

#(t): The raw files: LESFMIP on the common 2.5° grid, ERA5's hourly fields, the SAM index, and CMIP6's sea ice on CEDA
LESFMIP_RAW_DIR = Path('/gws/ssde/j25b/leader_epesc/CMIP6_SinglForcHistSimul/InterpolatedFlds')
ERA5_RAW_DIR = Path('/badc/ecmwf-era5/data/oper/an_sfc')
SAM_RAW_DIR = Path('/gws/ssde/j25b/leader_epesc/wg1/monthly/NAM_SAM_new_update_LastVersion_UseThese/SAM_zonal_new_jasmin')
CEDA_DIRS = [Path('/badc/cmip6/data/CMIP6/DAMIP'), Path('/badc/cmip6/data/CMIP6/CMIP'),
             Path('/badc/cmip6/data/CMIP6Plus/LESFMIP')]

#(t): The station observations (notebook plot_05, section 1)
OBS_PATH = Path('/gws/ssde/j25a/extant/users/rywill/Extremes_Database/Obs')

RECON_PATH = Path('/gws/ssde/j25a/extant/users/aborow/recon_t2m_1958-2025_ano.final.nc')


BERKLEY_PATH = Path('/gws/ssde/j25a/esmeval/obsdata-v2/Tier2/BerkeleyEarth/OBS_BerkeleyEarth_reanaly_2020_Amon_tas_185001-201912.nc')

EXTANT_PATH = Path('/gws/ssde/j25a/extant/users/aborow')