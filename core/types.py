from enum import Enum
from typing import Dict, List, Optional

class GPUType(Enum):
    RTX4090 = "NVIDIA GeForce RTX 4090"
    RTXA6000 = "NVIDIA RTX A6000"
    
    @classmethod
    def list_all(cls) -> List[str]:
        return [gpu.value for gpu in cls]

class PodState(Enum):
    Creating = 0
    Starting = 1
    Processing = 2
    Free = 3
    Stopped = 4
    Terminated = 5

class PodInfo:
    def __init__(
        self,
        port_mappings: Dict[str, int],
        public_ip: str
    ):
        self.port_mappings = port_mappings
        self.public_ip = public_ip

class Prompt:
    def __init__(
        self,
        url: str,
        workflow_id: int,
    ):
        self.url = url
        self.workflow_id = workflow_id
        self.result: Optional[PromptResult] = None

class PromptResult:
    def __init__(
        self,
        status: str,
        data
    ):
        self.status = status
        self.data = data