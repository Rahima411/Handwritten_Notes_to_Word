"""State objects used by the handwritten-notes agent."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentConfig:
    """User-controllable settings for one conversion run."""

    output_mode: str = "structured"
    autonomy_level: str = "semi"
    allow_memory: bool = False
    use_cloud_models: bool = False
    preserve_chemistry: bool = True
    prefer_exact_transcription: bool = False


@dataclass
class PageObservation:
    """What the agent observes before deciding how to process a page."""

    filename: str
    width: int
    height: int
    orientation: str
    aspect_ratio: float
    likely_lined_paper: bool
    likely_chemistry: bool
    likely_table: bool
    notes: List[str] = field(default_factory=list)


@dataclass
class AgentDecision:
    """The agent's chosen processing strategy."""

    strategy: str
    prompt: str
    actions: List[str]
    requires_review: bool
    rationale: List[str] = field(default_factory=list)


@dataclass
class AgentResult:
    """Final result for a processed page or batch."""

    raw_text: str
    document_bytes: bytes
    filename: str
    observation: PageObservation
    decision: AgentDecision
    feedback_items: List[str]
    audit_log: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchAgentResult:
    """Final result for multiple pages."""

    results: List[AgentResult]
    output_bytes: bytes
    filename: str
    output_type: str
    audit_log: List[str] = field(default_factory=list)
    feedback_items: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
