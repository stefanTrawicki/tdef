from .src.tdef import TDEF
from .src.dump_model import dumpONNX, dumpPyTorch

tuning_variants = {
    "trials": [
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
        # "xgb",
        "xgb_rank",
        # "ga",
        "random",
        # "gridsearch"
    ],
    "model": {
        "resnet18": {
            "path": "dumps/onnx/frozen/resnet18-v1-7.onnx",
            "input_name": "data",
            "input_shape": [1, 3, 224, 224]
        },
        "resnet152": {
            "path": "dumps/onnx/frozen/resnet152-v1-7.onnx",
            "input_name": "data",
            "input_shape": [1, 3, 224, 224]
        },
        "densenet121": {
            "path": "dumps/onnx/frozen/densenet-9.onnx",
            "input_name": "data_0",
            "input_shape": [1, 3, 224, 224]
        },
        "yolov4": {
            "path": "dumps/onnx/frozen/yolov4.onnx",
            "input_name": "input_1:0",
            "input_shape": [1, 416, 416, 3]
        },
        "robertabase": {
            "path": "dumps/onnx/frozen/roberta-base-11.onnx",
            "input_name": "input_ids",
            "input_shape": [1, 512]
        },
    }
}

# dumpONNX("dumps/onnx")

for trial in tuning_variants["trials"]:
    for model, model_data in tuning_variants["model"].items():
        for autoschedule in tuning_variants["autoschedule"]:
            job = {
                "trials": trial,
                "gpus": [0, 1],
                "enable_autoscheduler": autoschedule,
                "path": model_data["path"],
                "input_name": model_data["input_name"],
                "input_shape": model_data["input_shape"]
            }
            if not autoschedule:
                for tuner in tuning_variants["tuner"]:
                    job["tuner"] = tuner
                    print(job)
                    TDEF(job=job, dry_run=False)
            else:
                print(job)
                TDEF(job=job, dry_run=False)

# model = "yolov4"

# test_job = {
#     "trials": 100,
#     "gpus": [0, 1],
#     "enable_autoscheduler": True,
#     "path": tuning_variants["model"][model]["path"],
#     "input_name": tuning_variants["model"][model]["input_layer_name"],
#     "input_shape": tuning_variants["model"][model]["input_size"]
# }

# TDEF(
#     job=test_job,
#     dry_run=False
# )