from fastapi import HTTPException
import os

class errorhandling:
    def __init__(self):
        self.AllowedType = [".mp3", ".wav"]

    def soundcheck(self, file):
        _, file_extension = os.path.splitext(file.filename.lower())
        if file_extension not in self.AllowedType:
            return False
        else:
            return True