import os
import json
import numpy as np
import hashlib

# ============== TDEF imports ==============
from .tvm_model import Model

from .utilities import DefaultLog, ConfigureLog

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

    def __init__(self, job:dict = {}, model_path="/home/s/cupti/learning/tuning_as_a_defence/tvm/scripts/dumps/pytorch/resnet18.pth.zip", dry_run=False):
        file_name = model_path
        # shape_dict = None
        # with open(file_name + ".json") as f:
        #     shape_dict = json.load(f)

        job["path"] = file_name
        # job["shape_dict"] = shape_dict
        job["enable_autoscheduler"] = job.get("enable_autoscheduler", False)
        job["parallel"] = job.get("parallel", 4)
        job["trials"] = job.get("trials", 10000)
        job["timeout"] = job.get("timeout", 100)
        job["repeat"] = job.get("repeat", 1)
        job["number"] = job.get("number", 10)
        job["mixed_precision"] = job.get("mixed_precision", False)
        job["opt_level"] = job.get("opt_level", 3) 
        job["gpus"] = job.get("gpus", [0])
        job["target"] = job.get("target", "cuda -arch=sm_80")
        job["min_repeat_ms"] = job.get("min_repeat_ms", 1000)
        job["tuner"] = job.get("tuner", "xgb")
        job["early_stopping"] = job.get("early_stopping", 1000)
        job["include_simple_tasks"] = job.get("include_simple_tasks", False)

        job["hardware_params_num_cores"] = job.get("hardware_params_num_cores", None)
        job["hardware_params_vector_unit_bytes"] = job.get("hardware_params_vector_unit_bytes", None)
        job["hardware_params_cache_line_bytes"] = job.get("hardware_params_cache_line_bytes", None)
        job["hardware_params_max_shared_memory_per_block"] = job.get("hardware_params_max_shared_memory_per_block", None)
        job["hardware_params_max_local_memory_per_block"] = job.get("hardware_params_max_local_memory_per_block", None)
        job["hardware_params_max_threads_per_block"] = job.get("hardware_params_max_threads_per_block", None)
        job["hardware_params_max_vthread_extent"] = job.get("hardware_params_max_vthread_extent", None)
        job["hardware_params_warp_size"] = job.get("hardware_params_warp_size", None)
        job["hardware_params_target"] = job.get("hardware_params_target", job["target"])
        job["hardware_params_target_host"] = job.get("hardware_params_target_host", None)

        records_exist, records_path = self.records_and_data(job)
        job["records"] = records_path

        print(json.dumps(job, indent=4))

        model = Model(job)
        if not records_exist:
            print(f"Records {records_path!a} not found, will tune")
            model.tune(job, dry_run=dry_run)
        else:
            print(f"Existing records {records_path!a} found, tuning not required")

        if not dry_run:
            model.compile(job)
            times = []
            for i in range(0, 10000):
                times.append(model.inferRandom(profile=True))
            avg = sum(times) / len(times)
            times.sort()
            min_t = times[0]
            max_t = times[-1]
            med_t = times[int(len(times)/2)]
            var = np.var(times)
            print(f"avg {avg} ms \n" +
                  f"min {min_t} ms \n" +
                  f"med {med_t} ms \n" +
                  f"max {max_t} ms \n" +
                  f"variance {var}")