from enum import Enum


class AnalysisStatus(str, Enum):
    READY = "ready"
    ANALYZING = "analyzing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    COMPLETE = "complete"
    FAILED = "failed"
    DELETED = "deleted"


class ChunkType(str, Enum):
    TEXT = "TEXT"
    TABLE = "TABLE"
    IMAGE = "IMAGE"
    CAPTION = "CAPTION"
    METADATA = "METADATA"


class MessageRole(str, Enum):
    SYSTEM = "SYSTEM"
    USER = "USER"
    ASSISTANT = "ASSISTANT"