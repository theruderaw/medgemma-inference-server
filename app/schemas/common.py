from pydantic import BaseModel,ConfigDict

class ORMBase(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
class ErrorResponse(BaseModel):
    error: str
    message: str