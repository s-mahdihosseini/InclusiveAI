"""
Path override for running kappa-heterogeneity calibration with data that
lives under '<repo>/Claude Folder/Archive/...'.

Import this BEFORE importing simple_dynamic_model (or before calling
build_model_data), and it will patch the module-level path constants so
build_model_data() can find all required inputs.
"""

import os
import simple_dynamic_model as sdm

# Resolve the Archive directory from the mounted workspace.
# The workspace was mounted at '/sessions/.../mnt/AI and Expertise'.
_CANDIDATES = [
    "/sessions/nifty-focused-bell/mnt/AI and Expertise/Claude Folder/Archive",
]
_ARCHIVE = None
for c in _CANDIDATES:
    if os.path.isdir(c):
        _ARCHIVE = c
        break
if _ARCHIVE is None:
    raise FileNotFoundError(
        "Could not locate 'Claude Folder/Archive'. "
        "Mount the parent project folder and/or update paths_override.py."
    )

_OLG_MODEL_DIR   = os.path.join(_ARCHIVE, "OLG Model")
_LLM_DIR         = os.path.join(_ARCHIVE, "LLM Output and Old Datasets")

sdm._OLG_MODEL_DIR   = _OLG_MODEL_DIR
sdm._CALIBRATED_JSON = os.path.join(_OLG_MODEL_DIR, "calibrated_parameters.json")
sdm._BLS_OES_XLSX    = os.path.join(_OLG_MODEL_DIR, "national_M2024_dl.xlsx")
sdm._EXPERTISE_CSV   = os.path.join(_ARCHIVE, "prepare education data",
                                    "expertise_by_soc3.csv")
sdm._LLM_DIR         = _LLM_DIR
sdm._FINAL_OCC_XLSX  = os.path.join(_LLM_DIR, "Final_Occupation_Dataset.xlsx")
sdm._RETRAIN_NO_AI   = os.path.join(_LLM_DIR,
                                    "occ2occ_retraining_merged_without_ai.csv")
sdm._RETRAIN_WITH_AI = os.path.join(_LLM_DIR,
                                    "occ2occ_retraining_merged_with_ai.csv")
sdm._COLLEGE_CSV     = os.path.join(_ARCHIVE, "Create education by occupation data",
                                    "college_share_by_soc3.csv")

ARCHIVE = _ARCHIVE
