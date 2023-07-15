from .src.tdef import TDEF
from .src.dump_model import dumpONNX, dumpPyTorch

job_0 = {
    "trials": 1000,
    "gpus": [0],
    "enable_autoscheduler": False
}

job_1 = {
    "trials": 1000,
    "gpus": [0],
    "enable_autoscheduler": True
}

TDEF(
    job=job_1,
    model_path="/home/s/cupti/learning/tuning_as_a_defence/tvm/scripts/dumps/onnx/vision/classification/resnet/model/4e8f8653e7a2222b3904cc3fe8e304cd8b339ce1d05fd24688162f86fb6df52c_resnet18-v1-7.onnx",
    dry_run=False
)

TDEF(
    job=job_0,
    model_path="/home/s/cupti/learning/tuning_as_a_defence/tvm/scripts/dumps/onnx/vision/classification/resnet/model/4e8f8653e7a2222b3904cc3fe8e304cd8b339ce1d05fd24688162f86fb6df52c_resnet18-v1-7.onnx",
    dry_run=False
)