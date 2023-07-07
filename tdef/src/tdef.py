import os
import json
import ast
import time
import ctypes
import numpy as np

import hashlib

# ============== TDEF imports ==============
from .tvm_model import Model

def get_hash(data:dict) -> str:
    s = []
    for _, value in data.items():
        s.append(str(value))
    s.sort()
    s = "_".join(s)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


class TDEF():
    def records_and_data(self, job:dict) -> str:
        hash_code = get_hash(job)
        records = f"records/{hash_code}_records.json"
        data = f"records/{hash_code}_data.json"
        job["hash"] = hash_code
        job["records"] = records
        records_exist = os.path.exists(records)
        if not records_exist:
            with open(data, "w+") as f:
                f.write(json.dumps(job, indent=4))
        return records_exist, records

    def __init__(self, job:dict = {}, model_path="/home/s/cupti/learning/tuning_as_a_defence/tvm/scripts/dumps/pytorch/resnet18.pth.zip"):
        file_name = model_path
        # shape_dict = None
        # with open(file_name + ".json") as f:
        #     shape_dict = json.load(f)
        
        dry_run = False

        job["path"] = file_name
        # job["shape_dict"] = shape_dict
        job["enable_autoscheduler"] = job["enable_autoscheduler"] if "enable_autoscheduler" in job else False
        job["parallel"] = job["parallel"] if "parallel" in job else 4
        job["trials"] = job["trials"] if "trials" in job else 10000
        job["timeout"] = job["timeout"] if "timeout" in job else 100
        job["repeat"] = job["repeat"] if "repeat" in job else 1
        job["number"] = job["number"] if "number" in job else 10
        job["mixed_precision"] = job["mixed_precision"] if "mixed_precision" in job else False
        job["opt_level"] = job["opt_level"] if "opt_level" in job else 3
        job["gpus"] = job["gpus"] if "gpus" in job else [0]
        # job["target"] = job["target"] if "target" in job else "cuda -arch=sm_80"
        job["target"] = job["target"] if "target" in job else "cuda"
        job["min_repeat_ms"] = job["min_repeat_ms"] if "min_repeat_ms" in job else 1000
        job["tuner"] = job["tuner"] if "tuner" in job else "xgb"
        job["early_stopping"] = job["early_stopping"] if "early_stopping" in job else 1000

        records_exist, records_path = self.records_and_data(job)
        job["records"] = records_path

        print(json.dumps(job, indent=4))


        # job["records"] = "/home/s/cupti/learning/tuning_as_a_defence/tvm/scripts/dumps/onnx/vision/classification/resnet/model/test_records.json"
        model = Model(job)
        if not records_exist:
            print(f"Records {records_path!a} not found, will tune")
            if not dry_run:
                model.tune(job)
        else:
            print(f"Existing records {records_path!a} found, tuning not required")

        if not dry_run:
            model.compile(job)
            model.inferRandom(profile=True)