import threading
import ctypes

def terminate_thread(thread: threading.Thread, timeout: float = 5.0) -> bool:
    """
    Safely terminate a thread with proper cleanup
    
    Args:
        thread: The thread to terminate
        timeout: How long to wait for thread to exit (seconds)
        
    Returns:
        bool: True if thread was terminated successfully, False otherwise
    """
    if not thread.is_alive():
        return True
    
    if hasattr(thread, 'stop'):
        thread.stop()
        thread.join(timeout)
        if not thread.is_alive():
            return True
    
    thread_id = thread.ident
    if thread_id is None:
        raise ValueError("Thread has no ident (not started?)")
    
    exc = ctypes.py_object(SystemExit)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), exc)
    
    if res == 0:
        raise ValueError(f"Invalid thread ID: {thread_id}")
    elif res != 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")
    
    thread.join(timeout)
    
    if thread.is_alive():
        if hasattr(thread, '_Thread__stop'):
            thread._Thread__stop()
        return False
    
    return True