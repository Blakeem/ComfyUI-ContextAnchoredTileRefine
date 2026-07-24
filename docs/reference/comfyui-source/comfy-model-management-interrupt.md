# comfy/model_management.py — interrupt handling

**Source:** `comfy/model_management.py`, lines 1318–1345; `nodes.py`, lines 41–45
**ComfyUI version:** 0.3.45 (local source tree — `C:\Users\Blake\AppData\Local\Programs\@comfyorgcomfyui-electron\resources\ComfyUI`)
**Retrieved:** 2026-07-22

Extracted for: `InterruptProcessingException` and `throw_exception_if_processing_interrupted`
(how core loops check for/raise on user-requested interrupt).

## InterruptProcessingException / interrupt_current_processing / processing_interrupted / throw_exception_if_processing_interrupted (comfy/model_management.py, lines 1318–1345)

```python
#TODO: might be cleaner to put this somewhere else
import threading

class InterruptProcessingException(Exception):
    pass

interrupt_processing_mutex = threading.RLock()

interrupt_processing = False
def interrupt_current_processing(value=True):
    global interrupt_processing
    global interrupt_processing_mutex
    with interrupt_processing_mutex:
        interrupt_processing = value

def processing_interrupted():
    global interrupt_processing
    global interrupt_processing_mutex
    with interrupt_processing_mutex:
        return interrupt_processing

def throw_exception_if_processing_interrupted():
    global interrupt_processing
    global interrupt_processing_mutex
    with interrupt_processing_mutex:
        if interrupt_processing:
            interrupt_processing = False
            raise InterruptProcessingException()
```

## Call site: before_node_execution / interrupt_processing wrappers (nodes.py, lines 41–45)

```python
def before_node_execution():
    comfy.model_management.throw_exception_if_processing_interrupted()

def interrupt_processing(value=True):
    comfy.model_management.interrupt_current_processing(value)
```

Call sites of `throw_exception_if_processing_interrupted` in the 0.3.45 tree: `main.py:243`,
`nodes.py:42` (`before_node_execution`, shown above — invoked from `execution.py`, not itself
part of this doc set), and `comfy/sd.py:216` (inside CLIP token-weight encoding, not otherwise
part of this doc set).
