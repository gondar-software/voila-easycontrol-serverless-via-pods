import base64
import asyncio
import aiohttp
import runpod
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from runpod import AsyncioEndpoint, AsyncioJob
from dotenv import load_dotenv
import os
import time
from typing import Dict
from threading import Thread, Lock

from core.pod_manager import PodManager
from core.constants import \
    POD_EASYCONTROL_TEMPLATE_ID, \
    POD_EASYCONTROL_STORAGE_ID, \
    SERVERLESS_EASYCONTROL_ENDPOINT_ID, \
    ORIGIN_IMAGE_URL
from core.types import PodState, Prompt

load_dotenv()

class RequestCounter:
    def __init__(self):
        self._count = 0
        self._lock = Lock()
    
    def increment(self):
        with self._lock:
            self._count += 1
            return self._count
    
    def get(self):
        with self._lock:
            return self._count

class AppState:
    def __init__(self):
        self.counter = RequestCounter()
        self.managers: Dict[str, PodManager] = {}
        self.logging_thread = None
        self.initialized = False

app_state = AppState()

async def log_state():
    while True:
        if app_state.managers.get("easycontrol"):
            pod_state, prompt_state = app_state.managers["easycontrol"].state
            print(
                f"{pod_state[PodState.Creating]}  {pod_state[PodState.Starting]}  "
                f"{pod_state[PodState.Processing]}  {pod_state[PodState.Free]}  "
                f"{pod_state[PodState.Stopped]}  {pod_state[PodState.Terminated]}  "
                f"{prompt_state["queued"]}  {prompt_state["processing"]}  "
                f"{prompt_state["completed"]}  ",
                end="\r"
            )
        await asyncio.sleep(1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    set_max_threads()

    app_state.managers["easycontrol"] = PodManager(
        "pod-easycontrol",
        POD_EASYCONTROL_TEMPLATE_ID,
        POD_EASYCONTROL_STORAGE_ID
    )
    
    app_state.logging_thread = Thread(
        target=lambda: asyncio.run(log_state()),
        daemon=True
    )
    app_state.logging_thread.start()
    app_state.initialized = True
    
    yield
    
    for manager in app_state.managers.values():
        manager.stop()
    if app_state.logging_thread:
        app_state.logging_thread.join(timeout=1)

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

runpod.api_key = os.getenv("RUNPOD_API")

async def run_remote_job(url: str, workflow_id: int):
    """Generic function to run jobs on remote endpoints"""
    try:
        async with aiohttp.ClientSession() as session:
            endpoint = AsyncioEndpoint(os.getenv(f"ENDPOINT_ID{SERVERLESS_EASYCONTROL_ENDPOINT_ID}"), session)
            job: AsyncioJob = await endpoint.run({
                "url": url,
                "workflow_id": workflow_id
            })

            while True:
                status = await job.status()
                
                if status == "COMPLETED":
                    output = await job.output()
                    return output
                elif status in ["FAILED", "CANCELLED"]:
                    raise RuntimeError(f"Job failed with status: {status}")
                
                await asyncio.sleep(3)
    except Exception as e:
        raise RuntimeError(f"Remote job error: {str(e)}")

@app.post('/api/v2/prompt')
async def process_prompt(query: dict):
    start_time = time.time()
    current_count = app_state.counter.increment()
    
    try:
        url = query.get("url", ORIGIN_IMAGE_URL)
        workflow_id = query.get("workflow_id", 1)
        
        if current_count % 2 == 0:
            if workflow_id in {1, 2, 4, 5, 6}:
                result = await asyncio.to_thread(
                    app_state.managers["easycontrol"].queue_prompt,
                    Prompt(
                        url,
                        workflow_id
                    )
                )
                
                if result.status == "success":
                    print(f"mode1: {(time.time() - start_time):.4f} seconds")
                    return Response(
                        content=result.data["content"],
                        media_type=result.data["media_type"]
                    )
                raise HTTPException(500, detail=f"Processing error: {result.data}")
        else:
            output = await run_remote_job(
                url,
                workflow_id
            )
            
            print(f"mode2: {(time.time() - start_time):.4f} seconds")
            base64_image = output["message"]
            return Response(
                content=base64.b64decode(base64_image),
                media_type="image/jpeg"
            )
            
    except Exception as e:
        raise HTTPException(500, detail=f"Error processing request: {str(e)}")

@app.post('/api/v2/stop')
async def stop_service():
    for manager in app_state.managers.values():
        manager.stop()
    if app_state.logging_thread:
        app_state.logging_thread.join(timeout=1)
    return {"status": "stopped"}

@app.post('/api/v2/restart')
async def restart_service():
    await stop_service()
    
    app_state.managers["easycontrol"] = PodManager(
        "pod-easycontrol",
        POD_EASYCONTROL_TEMPLATE_ID,
        POD_EASYCONTROL_STORAGE_ID
    )
    app_state.logging_thread = Thread(
        target=lambda: asyncio.run(log_state()),
        daemon=True
    )
    app_state.logging_thread.start()
    return {"status": "restarted"}

def set_max_threads():
    new_max_workers = 150
    executor = ThreadPoolExecutor(max_workers=new_max_workers)
    loop = asyncio.get_event_loop()
    loop.set_default_executor(executor)

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app="server_ab_test:app",
        host="localhost",
        reload=False,
        port=8088
    )