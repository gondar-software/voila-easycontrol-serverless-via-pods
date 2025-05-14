import requests
import time
from threading import \
    Thread, \
    Lock
from typing import \
    List, \
    Optional

from .utils import \
    terminate_thread
from .types import \
    GPUType, \
    PodState, \
    PodInfo, \
    Prompt, \
    PromptResult
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
        image_name: str,
        gpu_types: List[GPUType] = [GPUType.RTXA6000], 
        pod_id: Optional[str] = None
    ):
        self._name = name,
        self._template_id = template_id,
        self._volume_id = volume_id,
        self._gpu_types = gpu_types
        self._image_name = image_name
        self._pod_id: Optional[str] = pod_id
        self._pod_info: Optional[PodInfo] = None
        self._lock = Lock()
        self._latest_updated_time: float = 0
        self._state = PodState.Creating
        self._is_working = False
        self._api_key = RUNPOD_API
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        })

        self._init_thread = Thread(
            target=self._initialize
        )
        self._init_thread.daemon = True
        self._init_thread.start()

    @property
    def name(self) -> str:
        with self._lock:
            return self._name
    
    @property
    def volume_id(self) -> str:
        with self._lock:
            return self._volume_id
        
    @property
    def image_name(self) -> str:
        with self._lock:
            return self._image_name

    @property
    def template_id(self) -> str:
        with self._lock:
            return self._template_id

    @property
    def gpu_types(self) -> List[GPUType]:
        with self._lock:
            return self._gpu_types

    @property
    def session(self) -> requests.Session:
        with self._lock:
            return self._session

    @property
    def state(self) -> PodState:
        with self._lock:
            return self._state
    
    @state.setter
    def state(self, value: PodState) -> None:
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
    def latest_updated_time(self) -> float:
        with self._lock:
            return self._latest_updated_time
    
    @latest_updated_time.setter
    def latest_updated_time(self, value: float) -> None:
        with self._lock:
            self._latest_updated_time = value

    @property
    def init_thread(self) -> Optional[Thread]:
        with self._lock:
            return self._init_thread
    
    @init_thread.setter
    def init_thread(self, value: Optional[Thread]) -> None:
        with self._lock:
            self._init_thread = value

    @property
    def is_working(self) -> bool:
        with self._lock:
            return self._is_working
        
    @is_working.setter
    def is_working(self, value: bool) -> None:
        with self._lock:
            self._is_working = value

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

            else:
                self.pod_info = self._resume_pod_and_get_pod_info()
                if self.pod_info is None:
                    return

            self._check_server()
            
        except Exception as e:
            self.state = PodState.Terminated

    def _create_pod(
        self
    ) -> Optional[str]:
        payload = {
            "gpuTypeIds": [gpu_type.value for gpu_type in self.gpu_types],
            "name": self.name[0],
            "gpuCount": 1,
            "networkVolumeId": self.volume_id[0],
            "imageName": self.image_name,
            "templateId": self.template_id[0],
            "supportPublicIp": True,
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
            except Exception as e:
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
                port_mappings = data.get("portMappings", None)
                public_ip = data.get("publicIp", "")
                if port_mappings and public_ip != "":
                    self.state = PodState.Starting
                    return PodInfo(port_mappings, public_ip)
            except:
                pass
            
            retries += 1
            time.sleep(POD_RETRY_DELAY / 1000.)

        else:
            self.state = PodState.Terminated

    def _resume_pod_and_get_pod_info(
        self
    ) -> Optional[PodInfo]:
        retries = 0
        init = True
        while retries < POD_START_RETRY_MAX:
            try:
                response = self.session.get(
                    f"https://rest.runpod.io/v1/pods/{self.pod_id}"
                )
                response.raise_for_status()
                data = response.json()
                port_mappings = data.get("portMappings", None)
                public_ip = data.get("publicIp", "")
                if port_mappings and public_ip != "":
                    self.state = PodState.Starting
                    return PodInfo(port_mappings, public_ip)
                elif (port_mappings is None or public_ip == "") and init:
                    init = False
                    self.resume()
            except:
                pass
            
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
            except:
                pass
            
            retries += 1
            time.sleep(POD_RETRY_DELAY / 1000.)
        else:
            self.state = PodState.Terminated

    @staticmethod
    def check_pod(
        pod_id: str, 
        template_id: str, 
        volume_id: str,
        image_name: str
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
            templateId = info.get("templateId", '')
            networkVolumeId = info.get("networkVolumeId", '')
            imageName = info.get("imageName", '')
            if template_id == templateId and \
                volume_id == networkVolumeId and \
                image_name == imageName:
                return True
        except:
            pass
            
        return False    

    def queue(
        self, 
        prompt: Prompt
    ) -> PromptResult:
        self.is_working = True
        self.latest_updated_time = time.time()
        while time.time() - self.latest_updated_time < POD_REQUEST_TIMEOUT_RETRY_MAX:
            if self.state == PodState.Free and self.pod_info:
                break
            elif self.state == PodState.Creating or \
                self.state == PodState.Starting or \
                self.state == PodState.Processing:
                time.sleep(POD_RETRY_DELAY / 1000.)
            elif self.state == PodState.Terminated or PodState.Stopped:
                self._is_working = False
                self.latest_updated_time = time.time()
                return PromptResult(
                    "error",
                    "Pod is not working."
                )
        else:
            self.latest_updated_time = time.time()
            self._is_working = False
            return PromptResult(
                "error",
                "Processing timeout."
            )

        self.state = PodState.Processing
        public_ip, port = self.pod_info.public_ip, self.pod_info.port_mappings["8188"]

        try:
            response = requests.post(
                f"http://{public_ip}:{port}/process",
                json={
                    "url": prompt.url,
                    "workflow_id": prompt.workflow_id
                },
                timeout=POD_REQUEST_TIMEOUT_RETRY_MAX
            )
            response.raise_for_status()
            self._is_working = False
            self.state = PodState.Free
            self.latest_updated_time = time.time()
            return PromptResult(
                "success",
                {
                    "content": response.content,
                    "media_type": response.headers.get("content-type", "image/jpeg")
                }
            )
        except:
            self._is_working = False
            self.state = PodState.Free
            self.latest_updated_time = time.time()
            return PromptResult(
                "error",
                "Unknown error occurred."
            ) 

    def stop(
        self
    ):
        try:
            response = self.session.post(
                f"https://rest.runpod.io/v1/pods/{self.pod_id}/stop"
            )
            response.raise_for_status()
            try:
                terminate_thread(self.init_thread)
            except:
                pass
            self.pod_info = None
            self.state = PodState.Stopped
            return True
        except:
            return False

    def resume(
        self
    ) -> bool:
        try:
            response = self.session.post(
                f"https://rest.runpod.io/v1/pods/{self.pod_id}/start"
            )
            response.raise_for_status()
            self.pod_info = None
            self.state = PodState.Creating
            self.init_thread = Thread(
                target=self._initialize
            )
            self.init_thread.daemon = True
            self.init_thread.start()
            return True
        except:
            return False

    def destroy(
        self
    ) -> bool:
        try:
            response = self.session.delete(
                f"https://rest.runpod.io/v1/pods/{self.pod_id}"
            )
            response.raise_for_status()
            try:
                terminate_thread(self.init_thread)
            except:
                pass
            return True
        except:
            return False

    