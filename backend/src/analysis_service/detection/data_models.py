from pydantic import BaseModel, Field, PositiveFloat, PositiveInt


class Suspect(BaseModel):
    candidate_id: str
    detection_threshold: PositiveFloat
    observed_similarity: PositiveInt
    p_value: float | None = Field(ge=0, le=1)
