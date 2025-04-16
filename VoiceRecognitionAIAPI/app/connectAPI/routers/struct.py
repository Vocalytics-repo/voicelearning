from pydantic import BaseModel
import os
'''
TEMP_DIR = "./temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)
'''
class TextResponse(BaseModel):
    text: str
    job_id: str
    processing_time: float
