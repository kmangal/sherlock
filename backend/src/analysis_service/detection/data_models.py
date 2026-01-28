from pydantic import BaseModel, Field, PositiveFloat, PositiveInt


class Suspect(BaseModel):
    candidate_id: str
    detection_threshold: PositiveFloat
    observed_similarity: PositiveInt
    p_value: float | None = Field(ge=0, le=1)


# class CheatingDetectionResults(BaseModel):
#     similarity_threshold: int = Field(
#         description="Similarity threshold used for the analysis"
#     )
#     observed_distribution: dict[set[str], int] = Field(
#         description="Map from candidate IDs to # of observed similarities"
#     )
#     suspects: set[frozenset[str]] = Field(
#         description="set of groups of candidate IDs that are flagged as suspicious"
#     )
