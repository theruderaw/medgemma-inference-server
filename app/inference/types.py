import json

from fastapi import UploadFile
from enum import StrEnum

ImageInput = bytes | str | UploadFile


class ChestXrayEntity(StrEnum):
    """
    Canonical labels from the NIH ChestX-ray14 dataset
    (Wang et al., 2017 — https://arxiv.org/abs/1705.02315).
    14 finding classes + "No Finding" = 15 total.
    """
    NO_FINDING = "No Finding"
    ATELECTASIS = "Atelectasis"
    CARDIOMEGALY = "Cardiomegaly"
    EFFUSION = "Effusion"
    INFILTRATION = "Infiltration"
    MASS = "Mass"
    NODULE = "Nodule"
    PNEUMONIA = "Pneumonia"
    PNEUMOTHORAX = "Pneumothorax"
    CONSOLIDATION = "Consolidation"
    EDEMA = "Edema"
    EMPHYSEMA = "Emphysema"
    FIBROSIS = "Fibrosis"
    PLEURAL_THICKENING = "Pleural_Thickening"
    HERNIA = "Hernia"


OUTPUT_JSON_STRUCTURE = json.dumps(
    {
        "summary": "string",
        "entities": [ChestXrayEntity.NO_FINDING.value],
        "notes": {
            "projection": None,
            "image_quality": None,
            "comparison": None,
            "limitations": [],
        },
    },
    indent=2,
)