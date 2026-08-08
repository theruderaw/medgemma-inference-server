import json

from fastapi import UploadFile
from enum import StrEnum

ImageInput = bytes | str | UploadFile

class ChestXrayEntity(StrEnum):
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
    