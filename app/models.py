from sqlalchemy import Column,Integer,String,Float,DateTime
from sqlalchemy.sql import func #get the current server time 
from .database import Base

class Prediction(Base):
    __tablename__= 'predictions'

    id=Column(Integer,primary_key=True,index=True)
    emotion=Column(String,index=True)
    confidence=Column(Float)

    ## This automatically sets the creation time on the database server
    created_at=Column(DateTime(timezone=True), server_default=func.now())