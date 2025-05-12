import requests
import uuid
import time
from threading import \
    Thread, \
    Lock
from typing import \
    List, \
    Dict

from .types import \
    GPUType, \
    PodState, \
    Prompt, \
    PromptResult
from .pod import \
    Pod
from .constants import \
    POD_MAX_NUM, \
    RUNPOD_API, \
    POD_REQUEST_TIMEOUT_RETRY_MAX, \
    POD_RETRY_DELAY

class PodManager:
    def __init__(
        self, 
        pre_name: str, 
        template_id: str, 
        volume_id: str, 
        gpu_types: List[GPUType] = [GPUType.RTXA6000]
    ):
        self._lock = Lock()
        self._pre_name = pre_name
        self._template_id = template_id
        self._volume_id = volume_id
        self._gpu_types = gpu_types
        self._api_key = RUNPOD_API
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        })

        self._pods: List[Pod] = []
        self._queued_prompts: Dict[str, Prompt] = {}
        self._processing_prompts: Dict[str, Prompt] = {}
        self._completed_prompts: Dict[str, Prompt] = {}
        self._stopped = False

        self._initialize()

    @property
    def session(self) -> requests.Session:
        with self._lock:
            return self._session

    @property
    def template_id(self) -> str:
        with self._lock:
            return self._template_id
        
    @property
    def pre_name(self) -> str:
        with self._lock:
            return self._pre_name

    @property 
    def volume_id(self) -> str:
        with self._lock:
            return self._volume_id

    @property
    def gpu_types(self) -> List[GPUType]:
        with self._lock:
            return self._gpu_types
        
    @property
    def pods(self) -> List[Pod]:
        with self._lock:
            return self._pods

    @property
    def queued_prompts(self) -> Dict[str, Prompt]:
        with self._lock:
            return self._queued_prompts
        
    @property
    def processing_prompts(self) -> Dict[str, Prompt]:
        with self._lock:
            return self._processing_prompts
        
    @property
    def completed_prompts(self) -> Dict[str, Prompt]:
        with self._lock:
            return self._completed_prompts
        
    @property
    def stopped(self) -> bool:
        with self._lock:
            return self._stopped
        
    @property
    def stopped(self, value: bool):
        with self._lock:
            self._stopped = value
        
    @property
    def state(self):
        with self._lock:
            pods_by_state = {}
            for pod in self._pods:
                if pod.state in pods_by_state:
                    pods_by_state[pod.state] += 1
                else:
                    pods_by_state[pod.state] = 1

            prompts_by_state = {
                "queued": len(self._queued_prompts),
                "processing": len(self._processing_prompts),
                "completed": len(self._completed_prompts)
            }

            return pods_by_state, prompts_by_state

    def _initialize(self):
        self._check_existing_pods()
        background_thread = Thread(
            target=self._background_work
        )
        background_thread.daemon = True
        background_thread.start()

    def _check_existing_pods(
        self
    ):
        try:
            response = self.session.get(
                "https://rest.runpod.io/v1/pods"
            )
            response.raise_for_status()
            data = response.json()
            for pod in data:
                pod_id = pod.get("id", None)
                pod_name = pod.get("name", None)
                if pod_id is not None and \
                    pod_name is not None and \
                    Pod.check_pod(
                        pod_id, 
                        self.template_id, 
                        self.volume_id, 
                        self.gpu_types
                    ):
                    self.pods.append(Pod(
                        pod_name,
                        self.template_id,
                        self.volume_id,
                        self.gpu_types,
                        pod_id=pod_id
                    ))
        except:
            return

    def _background_work(
        self
    ):
        while not self.stopped:
            if len(self.pods) < POD_MAX_NUM:
                self.pods.append(Pod(
                    f"{self.pre_name}-{uuid.uuid4()}",
                    self.template_id,
                    self.volume_id,
                    gpu_types=self.gpu_types
                ))
            elif len(self.pods) > POD_MAX_NUM:
                extra_size = len(self.pods) - POD_MAX_NUM
                terminated_count = 0

                for pod in sorted(
                    self.pods, 
                    key=lambda pod: (
                        pod.state == PodState.Stopped,
                        pod.latest_updated_time
                    )
                ):
                    if terminated_count >= extra_size:
                        break
                    if pod.state == PodState.Processing or \
                        pod.state == PodState.Terminated or \
                        pod.is_working:
                        continue
                    
                    pod.state = PodState.Terminated
                    terminated_count += 1

            for _ in range(len(self.queued_prompts)):
                for pod in sorted(
                    self.pods,
                    key=lambda pod: (
                        pod.state != PodState.Free,
                        pod.latest_updated_time
                    ),
                    reverse=True
                ):
                    if not pod.is_working:
                        if pod.state == PodState.Terminated:
                            continue
                        if pod.state != PodState.Stopped or \
                            pod.resume():
                            key = self.queued_prompts.keys()[0]
                            thread = Thread(
                                target=self._process_request,
                                args=[pod, key, self.queued_prompts.pop(key)]
                            )
                            thread.daemon = True
                            thread.start()
                            break

            for pod in self.pods:
                if pod.latest_updated_time is not None and \
                    time.time() - pod.latest_updated_time > POD_REQUEST_TIMEOUT_RETRY_MAX:
                    pod.stop()
                
                if pod.state == PodState.Terminated:
                    if pod.destroy():
                        self.pods.remove(pod)

            time.sleep(POD_RETRY_DELAY / 1000.)

    def _process_request(
        self,
        pod: Pod,
        id: str,
        prompt: Prompt
    ):
        self.processing_prompts[id] = prompt
        response = pod.queue(prompt)
        res_prompt = self.processing_prompts.pop(id)
        res_prompt.result = response
        self.completed_prompts[id] = res_prompt

    def queue_prompt(
        self, 
        prompt: Prompt
    ) -> PromptResult:
        id = uuid.uuid4()
        key = str(id)
        self.queued_prompts[key] = prompt

        while True:
            res_prompt = self.completed_prompts.pop(key, None)
            if res_prompt is not None:
                return res_prompt.result
            else:
                time.sleep(POD_RETRY_DELAY, 1000.)

    def stop(
        self
    ):
        self.stopped = True
        for pod in self.pods:
            while not pod.destroy():
                continue

