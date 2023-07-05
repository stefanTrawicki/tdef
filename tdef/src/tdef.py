import os
import json
import ast
import time
import logging
import ctypes
import numpy as np

# ============== TDEF imports ==============
from .tvm_model import Model

def get_hash(data):
    return int(abs(hash(json.dumps(data)) / 50000000000))


class TDEF():
    def get_record(self, file_name, job):
        data = job
        data["file"] = file_name
        h = get_hash(data)
        file = f"records/{h}_records.json"
        if os.path.exists(file):
            return f"records/{h}_records.json"
        return None

    def store_record(self, file_name, job):
        data = job
        data["file"] = file_name
        h = get_hash(data)
        data["hash"] = h
        with open(f"records/{h}_data.json", "w+") as f:
            f.write(json.dumps(data, indent=4))
        return f"records/{h}_records.json"

    def __init__(self):
        file_name = "/home/s/cupti/learning/tuning_as_a_defence/tvm/scripts/dumps/pytorch/resnet18.pth.zip"
        records = file_name + ".records.json"
        shape_dict = None
        with open(file_name + ".json") as f:
            shape_dict = json.load(f)
        
        job = {}

        job["path"] = file_name
        job["shape_dict"] = shape_dict
        job["enable_autoscheduler"] = job["enable_autoscheduler"] if "enable_autoscheduler" in job else False
        job["parallel"] = job["parallel"] if "parallel" in job else True
        job["trials"] = job["trials"] if "trials" in job else 10000
        job["timeout"] = job["timeout"] if "timeout" in job else 10
        job["repeat"] = job["repeat"] if "repeat" in job else 1
        job["number"] = job["number"] if "number" in job else 1
        job["mixed_precision"] = job["mixed_precision"] if "mixed_precision" in job else False
        job["opt_level"] = job["opt_level"] if "opt_level" in job else 4
        job["gpus"] = job["gpus"] if "gpus" in job else [1]
        job["target"] = job["target"] if "target" in job else "cuda -arch=sm_80"
        job["min_repeat_ms"] = job["min_repeat_ms"] if "min_repeat_ms" in job else 1000
        job["tuner"] = job["tuner"] if "tuner" in job else "random"
        job["early_stopping"] = job["early_stopping"] if "early_stopping" in job else None

        records = self.get_record(file_name, job)
        if records == None:
            records = self.store_record(file_name, job)
            logging.info(f"No existing records found in {records}, generated new")
        else:
            logging.info(f"Records found in {records}")
        
        job["records"] = records

        model = Model(job)

        if not os.path.exists(records):
            model.tune(job)

        model.compile(job)
        model.inferRandom(True)