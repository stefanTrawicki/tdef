from .src.tdef import TDEF
from .src.dump_model import dumpONNX

# There is an issue at the moment where importing TVM and torch causes a free pointer
# error as they both have overlapping symbols, didn't know it was possible.
# from .src.dump_model import getPyTorchModels
# getPyTorchModels()
# dumpONNX(model="resnet18")

job_1 = {
    "trials": 25,
    "gpus": [0],
    "parallel": 4,
    "number": 3,
    "timeout": 10,
}

job_2 = {
    "trials": 50,
    "gpus": [0],
    "parallel": 4,
    "number": 3,
    "timeout": 10,
}

job_3 = {
    "trials": 100,
    "gpus": [1],
    "parallel": 4,
    "number": 3,
    "timeout": 10,
}

TDEF(
    job=job_3,
    model_path="/home/s/cupti/learning/tuning_as_a_defence/tvm/scripts/dumps/onnx/vision/classification/resnet/model/4e8f8653e7a2222b3904cc3fe8e304cd8b339ce1d05fd24688162f86fb6df52c_resnet18-v1-7.onnx"
)

# TDEF(
#     job=job_2,
#     model_path="/home/s/cupti/learning/tuning_as_a_defence/tvm/scripts/dumps/onnx/vision/classification/resnet/model/4e8f8653e7a2222b3904cc3fe8e304cd8b339ce1d05fd24688162f86fb6df52c_resnet18-v1-7.onnx"
# )