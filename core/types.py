from enum import Enum
from typing import Dict, List, Optional, Tuple

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