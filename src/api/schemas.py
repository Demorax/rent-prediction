from pydantic import BaseModel, Field


class ApartmentInput(BaseModel):
    city: str
    region: str
    floor_space: float = Field(gt=0, le=500)
    land_space: float = Field(ge=0, default=0)
    disposition: str
    building_type: str
    condition: str
    equipment: str
    penb: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "city": "Praha",
                    "region": "Hlavní město Praha",
                    "floor_space": 55,
                    "land_space": 0,
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


class PredictionWithShapResponse(PredictionResponse):
    shap_base_value: float
    top_features: list[tuple[str, float]]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
