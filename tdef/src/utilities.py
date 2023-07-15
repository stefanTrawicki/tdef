import logging
import ctypes
import sys
import threading

_cudart = ctypes.CDLL('libcudart.so')

def ConfigureLog():
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.FileHandler("logs/debug.log"),
        logging.StreamHandler()
        ]
    )

# https://stackoverflow.com/a/56810619
def DefaultLog():
    manager = logging.root.manager
    manager.disabled = logging.NOTSET
    for logger in manager.loggerDict.values():
        if isinstance(logger, logging.Logger):
            logger.setLevel(logging.NOTSET)
            logger.propagate = True
            logger.disabled = False
            logger.filters.clear()
            handlers = logger.handlers.copy()
            for handler in handlers:
                try:
                    handler.acquire()
                    handler.flush()
                    handler.close()
                except (OSError, ValueError):
                    pass
                finally:
                    handler.release()
                logger.removeHandler(handler)

class Profiler():
    def __init__(self, profile:bool = False):
        self.profile = profile

    def __enter__(self):
        if self.profile:
            logging.info("Profiling started")
            ret = _cudart.cudaProfilerStart()
            if ret != 0:
                raise Exception("cudaProfilerStart() returned %d" % ret)
        else:
            logging.info("Profiling disabled")
        return self

    def __exit__(self, type, value, traceback):
        if self.profile:
            logging.info("Profiling stopped")
            ret = _cudart.cudaProfilerStop()
            if ret != 0:
                raise Exception("cudaProfilerStop() returned %d" % ret)
        else:
            logging.info("Profiling disabled")