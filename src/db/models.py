from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime
from .database import Base

class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, index=True)
    text_hash = Column(String(64), index=True, nullable=False)
    label = Column(String(32), nullable=False)
    confidence = Column(Float, nullable=False)
    probabilities = Column(JSONB, nullable=False)
    reasons = Column(JSONB, nullable=False)
    #proof_packet = Column(JSONB, nullable=True)
    raw_text = Column(Text, nullable=True)
    model_version = Column(String(32), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

#