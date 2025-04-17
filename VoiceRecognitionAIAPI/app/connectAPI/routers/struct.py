from pydantic import BaseModel
class TextResponse(BaseModel):
    text: str
    job_id: str
    processing_time: float
