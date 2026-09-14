from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
from app.db.database import Base

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, default="New Investigation")
    is_pinned = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")
    documents = relationship("UploadedDocument", back_populates="session", cascade="all, delete-orphan")

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("chat_sessions.id"))
    sender = Column(String)
    content = Column(Text)
    evidence_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("ChatSession", back_populates="messages")

class UploadedDocument(Base):
    __tablename__ = "uploaded_documents"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("chat_sessions.id"))
    filename = Column(String)
    file_type = Column(String)
    file_path = Column(String)
    is_rulebook = Column(String, default="false")
    processed_status = Column(String, default="pending")
    extracted_text = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("ChatSession", back_populates="documents")

class EmployeeAssessment(Base):
    __tablename__ = "employee_assessments"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String, default="default_corp")
    employee_name = Column(String)
    job_role = Column(String)
    department = Column(String)
    status = Column(String, default="Enrolled")
    overall_suitability = Column(Float, default=0.0)
    competency_scores = Column(JSON, nullable=True)
    proctoring_status = Column(String, default="Clean")
    hr_decision = Column(String, default="Pending")
    created_at = Column(DateTime, default=datetime.utcnow)