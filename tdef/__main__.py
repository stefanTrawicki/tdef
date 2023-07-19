from .src.tdef import TDEF
from .src.dump_model import dumpONNX, dumpPyTorch

tuning_variants = {
    "trials": [
        None,
        25,
        50,
        150,
        500,
        1000
    ],
    "autoschedule": [
        True,
        False
    ],
    "tuner": [
        "xgb",
        "xgb_rank",
        "ga",
        "random",
        "gridsearch"
    ],
    "model": {
        "resnet18": {
            "path": "dumps/onnx/frozen/resnet18-v1-7.onnx",
            "input_layer_name": "data",
            "input_size": [1, 3, 224, 224]
        },
        "resnet152": {
            "path": "dumps/onnx/frozen/resnet152-v1-7.onnx",
            "input_layer_name": "data",
            "input_size": [1, 3, 224, 224]
        },
        "densenet121": {
            "path": "dumps/onnx/frozen/densenet-9.onnx",
            "input_layer_name": "data",
            "input_size": [1, 3, 224, 224]
        },
        "yolov4": {
            "path": "dumps/onnx/frozen/yolov4.onnx",
            "input_layer_name": "input_1:0",
            "input_size": [1, 416, 416, 3]
        },
        "robertabase": {
            "path": "dumps/onnx/frozen/roberta-base-11.onnx",
            "input_layer_name": "input_ids",
            "input_size": [1, 512]
        },
    }
}

# dumpONNX("dumps/onnx")

# for trial in tuning_variants["trials"]:
#     for autoschedule in tuning_variants["autoschedule"]:
#         if not autoschedule:
#             for tuner in tuning_variants["tuner"]:
#                 TDEF(job={
#                     "trials": trial, "enable_autoscheduler": autoschedule, "tuner": tuner
#                 },
#                 model_path=)

model = "yolov4"

test_job = {
    "trials": 100,
    "gpus": [0, 1],
    "enable_autoscheduler": True,
    "path": tuning_variants["model"][model]["path"],
    "input_name": tuning_variants["model"][model]["input_layer_name"],
    "input_shape": tuning_variants["model"][model]["input_size"]
}

TDEF(
    job=test_job,
    dry_run=False
)