from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.model import CompanionModel
    from core.session import ConversationDB, Session
    from lora.scheduler import LoRAScheduler


@dataclass
class AppState:
    db: Optional[ConversationDB] = None
    model: Optional[CompanionModel] = None
    scheduler: Optional[LoRAScheduler] = None
    active_session: Optional[Session] = None
    startup_done: bool = False
    generation_lock: Lock = field(default_factory=Lock)
    total_tokens: int = 0
    total_elapsed: float = 0.0
    companion_id: Optional[str] = None
    active_model_id: Optional[str] = None
    training_active: bool = False
    training_progress: dict = field(default_factory=dict)  # epoch/step/total/loss
    last_user_activity: float = field(default_factory=time.time)

    @property
    def avg_tokens_per_second(self) -> float:
        if self.total_elapsed <= 0:
            return 0.0
        return self.total_tokens / self.total_elapsed
