import ctypes
_cudart = ctypes.CDLL('libcudart.so')

def start(profile:bool = False):
    if profile:
        print("Starting profiling")
        ret = _cudart.cudaProfilerStart()
        if ret != 0:
            raise Exception("cudaProfilerStart() returned %d" % ret)

def stop(profile:bool = False):
    if profile:
        print("Stopping profiling")
        ret = _cudart.cudaProfilerStop()
        if ret != 0:
            raise Exception("cudaProfilerStop() returned %d" % ret)