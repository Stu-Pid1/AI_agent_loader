from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class RunnerStatus(Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    RUNNING = "running"
    ERROR = "error"


@dataclass
class RunnerInfo:
    name: str
    supported_pipeline_tags: List[str]
    supported_libraries: List[str]
    description: str
    input_description: str
    output_description: str


@dataclass
class RunResult:
    success: bool
    output: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class BaseRunner(ABC):

    def __init__(self, device: str = "cuda"):
        self._model = None
        self._tokenizer = None
        self._pipeline = None
        self._model_id: Optional[str] = None
        self._device = device
        self._status = RunnerStatus.UNLOADED

    @property
    def status(self) -> RunnerStatus:
        return self._status

    @property
    def model_id(self) -> Optional[str]:
        return self._model_id

    @staticmethod
    @abstractmethod
    def get_info() -> RunnerInfo:
        ...

    @abstractmethod
    def load(self, model_id: str, **kwargs) -> None:
        ...

    @abstractmethod
    def run(self, inputs: Any, **kwargs) -> RunResult:
        ...

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
        self._model_id = None
        self._status = RunnerStatus.UNLOADED
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def is_loaded(self) -> bool:
        return self._status == RunnerStatus.READY

    @abstractmethod
    def get_default_parameters(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def validate_model(self, model_id: str) -> bool:
        ...
