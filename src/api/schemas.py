from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Literal

# How Pydantic works:
# 1. Python type hints alone do not check incoming data.
# 2. Inheriting from BaseModel gives our class Pydantic's validation machinery.
# 3. When the class is defined, Pydantic reads its types and Field constraints
#    and builds a validation schema: a description of acceptable data.
# 4. Field supplies extra rules (e.g. gt=0); it does not perform checks itself.
#    Field(gt=0) supplies no default value, so that field is still required.
# 5. When FastAPI receives a body declared as PredictionRequest, Pydantic
#    validates the parsed data, including every observation inside history.
# 6. Success produces an object with attributes such as request.engine_id;
#    invalid data produces a validation error instead of reaching prediction.
#    Some compatible values may be converted, e.g. "42" into the integer 42.
# Remember: BaseModel provides the machinery; types and Field define the rules.


# BaseModel reads our types and Field rules, then validates incoming data.
class CycleObservation(BaseModel):  # Inherit Pydantic's validation machinery for one cycle.
    
    model_config = ConfigDict(
        allow_inf_nan=False,
        strict=True,
        extra="forbid",
    )

    time_in_cycles : int =  Field(gt=0)  # Required integer; Field adds the rule: greater than 0.
    operational_setting_1 : float  # No default means required; Pydantic validates it as a float.
    operational_setting_2 : float
    operational_setting_3 : float
    sensor_measurement_1 : float
    sensor_measurement_2 : float
    sensor_measurement_3 : float
    sensor_measurement_4 : float
    sensor_measurement_5 : float
    sensor_measurement_6 : float
    sensor_measurement_7 : float
    sensor_measurement_8 : float
    sensor_measurement_9 : float
    sensor_measurement_10 : float
    sensor_measurement_11 : float
    sensor_measurement_12 : float
    sensor_measurement_13 : float
    sensor_measurement_14 : float
    sensor_measurement_15 : float
    sensor_measurement_16 : float
    sensor_measurement_17 : float
    sensor_measurement_18 : float
    sensor_measurement_19 : float
    sensor_measurement_20 : float
    sensor_measurement_21 : float

class PredictionRequest(BaseModel):  # Validate the whole request using the definitions below.

    model_config = ConfigDict(strict=True, extra="forbid") #it means strict types(the user must provide an int for engine_id 42 and not string "42" ), and no additional unexpected fields

    
    engine_id: int = Field(gt=0)  # Field describes the constraint; Pydantic enforces it.
    # Validate every item as a CycleObservation; require at least 20 items.
    history: list[CycleObservation] = Field(min_length=20)

    @field_validator("history", mode="after") #Registers this function as an additional check for history. "after" means it runs after Pydantic validates the list and its observations, so we can safely access element.time_in_cycles
    @classmethod #Makes this a method associated with the class. Python supplies the class as cls; Pydantic supplies the validated history. We don’t need to use cls inside this particular check.
    def cycle_check(cls, history):
        cycles = []

        for element in history:
            cycles.append(element.time_in_cycles)

        for i in range(len(cycles)):
            if cycles[i] != i + 1:
                raise ValueError(
                    "Cycles must start at 1 and be ordered and consecutive."
                )

        return history

class BatchPredictionRequest(BaseModel):

    model_config = ConfigDict(strict=True, extra="forbid")

    engines: list[PredictionRequest] = Field(min_length=1)  # Require at least one engine in the batch.

class PredictionFactor(BaseModel):
    feature: str
    value: float = Field(allow_inf_nan=False)
    shap_value: float = Field(allow_inf_nan=False)

#The request schema defines what we accept; the response schema defines what we promise to return.
class PredictionResponse(BaseModel):
    engine_id: int = Field(gt=0)
    cycle: int = Field(gt=0)
    predicted_rul: float = Field(ge=0, allow_inf_nan=False)
    status: Literal["healthy", "warning", "critical"]
    top_factors: list[PredictionFactor]
    model_version: str

