import ctypes, logging
_cudart = ctypes.CDLL('libcudart.so')

def start():
    logging.info("Starting profiling")
    ret = _cudart.cudaProfilerStart()
    if ret != 0:
        raise Exception("cudaProfilerStart() returned %d" % ret)

def stop():
    logging.info("Stopping profiling")
    ret = _cudart.cudaProfilerStop()
    if ret != 0:
        raise Exception("cudaProfilerStop() returned %d" % ret)