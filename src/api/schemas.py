from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field


class BuildingType(str, Enum):
    BRICK = "BRICK"
    PANEL = "PANEL"
    WOODEN = "WOODEN"
    LOW_ENERGY = "LOW_ENERGY"
    UNDEFINED = "UNDEFINED"


class Condition(str, Enum):
    NEW = "NEW"
    VERY_GOOD = "VERY_GOOD"
    GOOD = "GOOD"
    BAD = "BAD"
    UNDER_CONSTRUCTION = "UNDER_CONSTRUCTION"
    BEFORE_RECONSTRUCTION = "BEFORE_RECONSTRUCTION"
    AFTER_RECONSTRUCTION = "AFTER_RECONSTRUCTION"
    UNDEFINED = "UNDEFINED"


class Equipment(str, Enum):
    UNFURNISHED = "UNFURNISHED"
    PARTIALLY = "PARTIALLY"
    FURNISHED = "FURNISHED"
    UNDEFINED = "UNDEFINED"


class EnergyClass(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"


class Disposition(str, Enum):
    DISP_1_KK = "DISP_1_KK"
    DISP_1_1 = "DISP_1_1"
    DISP_2_KK = "DISP_2_KK"
    DISP_2_1 = "DISP_2_1"
    DISP_3_KK = "DISP_3_KK"
    DISP_3_1 = "DISP_3_1"
    DISP_4_KK = "DISP_4_KK"
    DISP_4_1 = "DISP_4_1"
    DISP_5_KK = "DISP_5_KK"
    DISP_5_1 = "DISP_5_1"
    DISP_6_KK = "DISP_6_KK"
    DISP_OTHER = "DISP_OTHER"


class ApartmentInput(BaseModel):
    city: Annotated[str, Field(min_length=1, max_length=100)]
    region: Annotated[str, Field(min_length=1, max_length=100)]
    floor_space: Annotated[float, Field(gt=0, le=500)]
    land_space: Annotated[float, Field(ge=0)] = 0
    disposition: Disposition
    building_type: BuildingType
    condition: Condition
    equipment: Equipment
    penb: EnergyClass

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "city": "Praha",
                    "region": "Hlavní město Praha",
                    "floor_space": 55.0,
                    "land_space": 0.0,
                    "disposition": "DISP_2_KK",
                    "building_type": "BRICK",
                    "condition": "GOOD",
                    "equipment": "FURNISHED",
                    "penb": "C",
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    predicted_rent: float
    currency: str = "CZK"


class FeatureContribution(BaseModel):
    feature: str
    contribution: float


class PredictionWithExplanationResponse(PredictionResponse):
    base_value: float
    top_features: list[FeatureContribution]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
