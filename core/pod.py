import requests
import time
from threading import Thread, Lock
from typing import List, Optional

from .types import \
    GPUType, \
    PodState, \
    PodInfo
from .constants import \
    RUNPOD_API, \
    POD_RETRY_DELAY, \
    POD_CREATE_RETRY_MAX, \
    POD_START_RETRY_MAX, \
    POD_RUN_SERVER_RETRY_MAX, \
    POD_REQUEST_TIMEOUT_RETRY_MAX

class Pod:
    def __init__(
        self, 
        name: str, 
        template_id: str, 
        volume_id: str, 
        gpu_types: List[GPUType] = [GPUType.RTXA6000], 
        pod_id: Optional[str] = None
    ):
        self._name = name,
        self._template_id = template_id,
        self._volume_id = volume_id,
        self._gpu_types = gpu_types
        self._pod_id: Optional[str] = pod_id
        self._pod_info: Optional[PodInfo] = None
        self._lock = Lock()
        self._latest_updated_time: Optional[float] = None
        self._state = PodState.Creating
        self._api_key = RUNPOD_API
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        })

    @property
    def name(self) -> Optional[str]:
        with self._lock:
            return self._name
    
    @property
    def volume_id(self) -> Optional[str]:
        with self._lock:
            return self._volume_id

    @property
    def template_id(self) -> Optional[str]:
        with self._lock:
            return self._template_id

    @property
    def gpu_types(self) -> Optional[List[GPUType]]:
        with self._lock:
            return self._gpu_types

    @property
    def session(self) -> Optional[requests.Session]:
        with self._lock:
            return self._session

    @property
    def state(self) -> Optional[PodState]:
        with self._lock:
            return self._state
    
    @state.setter
    def state(self, value: Optional[PodState]) -> None:
        with self._lock:
            self._state = value

    @property
    def pod_id(self) -> Optional[str]:
        with self._lock:
            return self._pod_id
        
    @pod_id.setter
    def pod_id(self, value: Optional[str]) -> None:
        with self._lock:
            self._pod_id = value

    @property
    def pod_info(self) -> Optional[PodInfo]:
        with self._lock:
            return self._pod_info
        
    @pod_info.setter
    def pod_info(self, value: Optional[PodInfo]) -> None:
        with self._lock:
            self._pod_info = value

    @property
    def latest_updated_time(self) -> Optional[float]:
        with self._lock:
            return self._latest_updated_time
    
    @latest_updated_time.setter
    def latest_updated_time(self, value: Optional[float]) -> None:
        with self._lock:
            self._latest_updated_time = value

    def _initialize(
        self
    ):
        try:
            if self.pod_id is None:
                self.pod_id = self._create_pod()
            if self.pod_id is None:
                return

            self.pod_info = self._get_pod_info()
            if self.pod_info is None:
                return

            self._check_server()
            
        except:
            self.state = PodState.Terminated

    def _create_pod(
        self
    ) -> Optional[str]:
        payload = {
            "gpuTypeIds": [gpu_type.value for gpu_type in self.gpu_types],
            "name": self.name,
            "networkVolumeId": self.volume_id,
            "templateId": self.template_id,
            "ports": [
                "8188/tcp"
            ]
        }

        retries = 0
        while retries < POD_CREATE_RETRY_MAX:
            try:
                response = self.session.post(
                    "https://rest.runpod.io/v1/pods",
                    json=payload
                )
                response.raise_for_status()
                return response.json().get("id", "")
            except:
                retries += 1
                time.sleep(POD_RETRY_DELAY / 1000.)
                continue
        else:
            self.state = PodState.Terminated

    def _get_pod_info(
        self
    ) -> Optional[PodInfo]:
        retries = 0
        while retries < POD_START_RETRY_MAX:
            try:
                response = self.session.get(
                    f"https://rest.runpod.io/v1/pods/{self.pod_id}"
                )
                response.raise_for_status()
                data = response.json()

                if "portMappings" in data and "publicIp" in data:
                    self.state = PodState.Starting
                    return PodInfo(data["portMappings"], data["publicIp"])

            finally:
                retries += 1
                time.sleep(POD_RETRY_DELAY / 1000.)

        else:
            self.state = PodState.Terminated

    def _check_server(
        self
    ):
        public_ip, port = self.pod_info.public_ip, self.pod_info.port_mappings["8188"]
        
        retries = 0
        while retries < POD_RUN_SERVER_RETRY_MAX:
            try:
                response = requests.get(
                    f"http://{public_ip}:{port}/health"
                )
                response.raise_for_status()
                data = response.json()

                if data.get("status", None) == "ready":
                    self.latest_updated_time = time.time()
                    self.state = PodState.Free
                    return
            finally:
                retries += 1
                time.sleep(POD_RETRY_DELAY / 1000.)
        else:
            self.state = PodState.Terminated

    @staticmethod
    def check_pod(
        pod_id: str, 
        template_id, 
        gpu_types: List[GPUType] = [GPUType.RTXA6000]
    ):
        try:
            response = requests.get(
                f"https://rest.runpod.io/v1/pods/{pod_id}",
                headers={
                    "Authorization": f"Bearer {RUNPOD_API}",
                    "Content-Type": "application/json"
                }
            )
            response.raise_for_status()
            info = response.json()
            if "machine" in info:
                machine = info["machine"]
                gpuTypeId = machine.get("gpuTypeId", None)
                if gpuTypeId in [gpu_type.name for gpu_type in gpu_types]:
                    return True
        finally:
            return False        

    def queue(
        self, 
        url, 
        workflow_id
    ):
        self.latest_updated_time = time.time()
        while time.time() - self.latest_updated_time < POD_REQUEST_TIMEOUT_RETRY_MAX:
            if self.state == PodState.Free:
                break
            elif self.state == PodState.Creating or \
                self.state == PodState.Starting or \
                self.state == PodState.Processing:
                time.sleep(POD_RETRY_DELAY / 1000.)
            elif self.state == PodState.Terminated or PodState.Stopped:
                return {
                    "status": "error",
                    "data": "Pod is not working."
                }
        else:
            return {
                "status": "error",
                "data": "Processing timeout."
            }

        public_ip, port = self.pod_info.public_ip, self.pod_info.port_mappings["8188"]

        try:
            response = requests.post(
                f"http://{public_ip}:{port}/process",
                json={
                    "url": url,
                    "workflow_id": workflow_id
                },
                timeout=POD_REQUEST_TIMEOUT_RETRY_MAX
            )
            response.raise_for_status()
            return {
                "status": "success",
                "data": {
                    "content": response.content,
                    "media_type": response.headers.get("content-type", "image/jpeg")
                }
            }
        except:
            return {
                "status": "error",
                "data": "Unknown error occurred."
            }

    def _destroy(
        self
    ):
        while True:
            try:
                response = self.session.delete(
                    f"https://rest.runpod.io/v1/pods/{self.pod_id}"
                )
                response.raise_for_status()
                return
            except:
                time.sleep(POD_RETRY_DELAY / 1000.)

    