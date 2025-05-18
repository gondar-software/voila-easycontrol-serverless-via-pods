import requests
import uuid
import time
import numpy as np
from collections import \
    deque
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
    POD_RETRY_DELAY, \
    POD_MIN_NUM, \
    POD_SCALING_SENSIVITY

class PodManager:
    def __init__(
        self, 
        pre_name: str, 
        template_id: str, 
        volume_id: str, 
        image_name: str,
        gpu_types: List[GPUType] = [GPUType.RTXA6000]
    ):
        self._lock = Lock()
        self._pre_name = pre_name
        self._template_id = template_id
        self._volume_id = volume_id
        self._image_name = image_name
        self._gpu_types = gpu_types
        self._api_key = RUNPOD_API
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        })

        self._prompts_histories = deque([], maxlen=300)
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
    def image_name(self) -> str:
        with self._lock:
            return self._image_name

    @property
    def gpu_types(self) -> List[GPUType]:
        with self._lock:
            return self._gpu_types
        
    @property
    def pods(self) -> List[Pod]:
        with self._lock:
            return self._pods

    @property
    def prompts_histories(self) -> deque:
        with self._lock:
            return self._prompts_histories

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
        
    @stopped.setter
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
        background_thread1 = Thread(
            target=self._background_work
        )
        background_thread1.daemon = True
        background_thread1.start()
        background_thread2 = Thread(
            target=self._clear_prompts
        )
        background_thread2.daemon = True
        background_thread2.start()

    def _calc_num_pods(self) -> int:
        num_prompts = len(self.queued_prompts) + len(self.processing_prompts)
        self.prompts_histories.append(num_prompts)
        
        avg_load = np.average(self.prompts_histories)
        peak_load = max(self.prompts_histories)
        
        weighted_load = (avg_load * (100. - POD_SCALING_SENSIVITY) / 100. + 
                       peak_load * (POD_SCALING_SENSIVITY / 100.))
        
        return min(POD_MAX_NUM, POD_MIN_NUM + round(weighted_load * 1.2))

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
                if pod_id and \
                    str(pod_name).startswith(self.pre_name) and \
                    Pod.check_pod(
                        pod_id, 
                        self.template_id, 
                        self.volume_id, 
                        self.image_name
                    ):
                    self.pods.append(Pod(
                        pod_name,
                        self.template_id,
                        self.volume_id,
                        self.image_name,
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
                    self.image_name,
                    gpu_types=self.gpu_types
                ))
            elif len(self.pods) > POD_MAX_NUM:
                extra_size = len(self.pods) - POD_MAX_NUM
                terminated_count = 0

                for pod in sorted(
                    self.pods, 
                    key=lambda pod: (
                        pod.state != PodState.Stopped,
                        pod.state != PodState.Creating,
                        pod.state != PodState.Starting,
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
                        not pod.is_working,
                        pod.state == PodState.Free,
                        pod.state == PodState.Starting,
                        pod.state == PodState.Creating,
                        pod.latest_updated_time
                    ),
                    reverse=True
                ):
                    if pod.is_working:
                        continue
                        
                    if pod.state == PodState.Terminated:
                        continue
                        
                    if pod.state == PodState.Stopped and not pod.resume():
                        continue
                        
                    key = next(iter(self.queued_prompts))
                    prompt = self.queued_prompts.pop(key, None)
                    if prompt is None:
                        continue
                    
                    thread = Thread(
                        target=self._process_request,
                        args=(pod, key, prompt),
                        daemon=True
                    )
                    thread.start()
                    break

            num_pods = self._calc_num_pods()
            cur_num_pods = len([pod for pod in self.pods 
                    if pod.state == PodState.Creating or 
                    pod.state == PodState.Starting or
                    pod.state == PodState.Processing or
                    pod.state == PodState.Free])
            if cur_num_pods > num_pods:
                extra_num = cur_num_pods - num_pods
                extra_count = 0
                for _ in range(extra_num):
                    for pod in sorted(
                            self.pods, 
                            key=lambda pod: (
                                pod.latest_updated_time
                            )
                        ):
                        if pod.state != PodState.Stopped and \
                            pod.state != PodState.Terminated and \
                            not pod.is_working and \
                            time.time() - pod.latest_updated_time > POD_REQUEST_TIMEOUT_RETRY_MAX:
                            if pod.stop():
                                extra_count += 1
                        if extra_count >= extra_num:
                            break
            else:
                extra_num = num_pods - cur_num_pods
                extra_count = 0
                for _ in range(extra_num):
                    for pod in self.pods:
                        if pod.state == PodState.Stopped:
                            if pod.resume():
                                extra_count += 1
                        if extra_count >= extra_num:
                            break
            
            for pod in self.pods:
                if pod.state == PodState.Terminated:
                    if pod.destroy():
                        self.pods.remove(pod)

            time.sleep(50 / 1000.)

    def _process_request(
        self,
        pod: Pod,
        id: str,
        prompt: Prompt
    ):
        self.processing_prompts[id] = prompt
        pod.is_working = True
        try:
            response = pod.queue(prompt)
            res_prompt = self.processing_prompts.pop(id, None)
            pod.is_working = False
            if res_prompt is None:
                return
            res_prompt.result = response
            self.completed_prompts[id] = res_prompt
        except:
            pod.is_working = False
            res_prompt = self.processing_prompts.pop(id, None)
            if res_prompt is None:
                return
            res_prompt.result = PromptResult(
                "error",
                "unknown error occurred."
            )
            self.completed_prompts[id] = res_prompt

    def _clear_prompts(
        self
    ):
        while not self.stopped:
            current_time = time.time()
            timeout = POD_REQUEST_TIMEOUT_RETRY_MAX

            expired_queued = [key for key, prompt in self.queued_prompts.items() 
                            if current_time - prompt.start_time > timeout]
                            
            expired_processing = [key for key, prompt in self.processing_prompts.items() 
                                if current_time - prompt.start_time > timeout]
                                
            expired_completed = [key for key, prompt in self.completed_prompts.items() 
                                if current_time - prompt.start_time > timeout]

            for key in expired_queued:
                self.queued_prompts.pop(key, None)
                
            for key in expired_processing:
                self.processing_prompts.pop(key, None)
                
            for key in expired_completed:
                self.completed_prompts.pop(key, None)

            time.sleep(POD_RETRY_DELAY / 1000.)

    def queue_prompt(
        self, 
        prompt: Prompt
    ) -> PromptResult:
        id = uuid.uuid4()
        key = str(id)
        self.queued_prompts[key] = prompt

        start_time = time.time()

        while time.time() - start_time < POD_REQUEST_TIMEOUT_RETRY_MAX:
            res_prompt = self.completed_prompts.pop(key, None)
            if res_prompt:
                return res_prompt.result
            else:
                time.sleep(POD_RETRY_DELAY / 1000.)

        else:
            prompt.result = PromptResult(
                "error",
                "request timeout."
            )
            return prompt.result

    def stop(
        self
    ):
        self.stopped = True
        for pod in self.pods:
            while not pod.destroy():
                continue

