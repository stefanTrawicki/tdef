import os
import json
import numpy as np
import hashlib
from tqdm import trange
import jsonschema
import onnx
import onnxruntime
from .utilities import Profiler

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

job_schema = {
    "type": "object",
    "properties": {
        "model": {"type": "string", "default": None},
        "path": {"type": "string", "default": None},
        "records": {"type": "string", "default": None},
        "input_shape": {
            "type": "array",
            "items": {"type": "integer"},
            "default": [1, 3, 224, 224],
        },
        "input_name": {"type": "string", "default": "data"},
        "enable_autoscheduler": {"type": "boolean", "default": False},
        "parallel": {"type": "integer", "default": 4},
        "trials": {"type": "integer", "default": None},
        "timeout": {"type": "integer", "default": 100},
        "repeat": {"type": "integer", "default": 1},
        "number": {"type": "integer", "default": 10},
        "mixed_precision": {"type": "boolean", "default": False},
        "opt_level": {"type": "integer", "default": 3},
        "gpus": {
            "type": "array",
            "items": {"type": "integer"},
            "default": [0, 1],
        },
        "target": {"type": "string", "default": "cuda -arch=sm_80"},
        "min_repeat_ms": {"type": "integer", "default": 1000},
        "tuner": {"type": "string", "default": "xgb_rank"},
        "early_stopping": {"type": "integer", "default": 1000},
        "include_simple_tasks": {"type": "boolean", "default": False},
        "hardware_params_num_cores": {"type": "integer", "default": None},
        "hardware_params_vector_unit_bytes": {"type": "integer", "default": None},
        "hardware_params_cache_line_bytes": {"type": "integer", "default": None},
        "hardware_params_max_shared_memory_per_block": {"type": "integer", "default": None},
        "hardware_params_max_local_memory_per_block": {"type": "integer", "default": None},
        "hardware_params_max_threads_per_block": {"type": "integer", "default": None},
        "hardware_params_max_vthread_extent": {"type": "integer", "default": None},
        "hardware_params_warp_size": {"type": "integer", "default": None},
        "hardware_params_target": {"type": "string", "default": "cuda -arch=sm_80"},
        "hardware_params_target_host": {"type": "string", "default": None},
    },
}

def pretty_print_schema(schema):
    schema_json_str = json.dumps(schema, indent=2, sort_keys=True)
    schema_dict = json.loads(schema_json_str)
    max_key_length = max(len(key) for key in schema_dict.keys())
    for key, value in schema_dict.items():
        padding = max_key_length - len(key) + 2
        print(f"{key}{' ' * padding}: {value}")

def assign_defaults(data, schema):
    merged_data = data.copy()
    for key, field in schema.get("properties", {}).items():
        if key not in merged_data:
            merged_data[key] = field.get("default")
    merged_data["hardware_params_target"] = merged_data.get("target")
    return merged_data

def create_hashed_dir(data_dict):
    hashable_dict = data_dict.copy()
    json_string = json.dumps(hashable_dict, sort_keys=True)
    del(hashable_dict["gpus"])
    hash_value = hashlib.sha256(json_string.encode()).hexdigest()
    model_dir = f"records/{data_dict.get('model', 'default_model')}/{hash_value}/"
    metadata_filename = os.path.join(model_dir, f"metadata.json")
    records_filename = os.path.join(model_dir, f"records.json")
    already_existed = os.path.exists(records_filename) and os.path.isfile(records_filename)
    if already_existed:
        file_size_ok = os.path.getsize(records_filename) > 10
        already_existed = file_size_ok
        if not file_size_ok:
            print("Found records but file size was < 10 bytes, so probably failed previously. Will rerun")
    os.makedirs(model_dir, exist_ok=True)
    with open(metadata_filename, 'w') as file:
        file.write(json.dumps(data_dict, indent=2))
    open(records_filename, "a").close()
    return model_dir, already_existed

def run_model_x_times(model, job, x, profile=False):
    times = []
    for i in trange(x):
        input_data = np.random.rand(*job["input_shape"])
        model.module.set_input(job["input_name"], input_data)
        t, o = model.inferRandom(profile=profile)
        times.append(t)
    avg = sum(times) / len(times)
    times.sort()
    min_t = times[0]
    max_t = times[-1]
    med_t = times[int(len(times)/2)]
    var = np.var(times)
    info = {
        "timing": {
            "min": min_t,
            "max": max_t,
            "med": med_t,
            "var": var,
            "avg": avg,
            "times": times
        }
    }
    return info

class TDEF():
    def __init__(self, job_json, force_no_tuning=False, collecting_baselines=False):
        try:
            jsonschema.validate(instance=job_json, schema=job_schema)
            job = assign_defaults(job_json, job_schema)
        except jsonschema.exceptions.ValidationError as e:
            print("Job was invalid:")
            print(e)
            return
        
        x = 10
        if collecting_baselines:
            print("Collecting baselines, nothing else...")
            job["records"] = None
            model = Model(job)
            model.compile(job)
            print("Running baseline measured inference...")
            run_model_x_times(model, job, 1, profile=True)
            print("Finished collecting baselines!")
            return

        pretty_print_schema(job)
        dir, test_ran_previously = create_hashed_dir(job)
        records = os.path.join(dir, "records.json")
        job["records"] = records

        timings = os.path.join(dir, "timings.json")
        if os.path.exists(timings) and not force_no_tuning:
            print("Already ran combination and gathered timings, skipping!")
            return

        model = None
        if not force_no_tuning:
            if not test_ran_previously:
                print(f"First time combination ({dir!a}) ran, beginning tuning")
                model = Model(job)
                model.tune(job)
            else:
                print(f"Combination ({dir!a}) ran previously, no tuning required")

        if force_no_tuning:
            print("Will not attempt to tune the model, only execute!")

        model = Model(job)
        model.compile(job)

        # see if this has a bearing on perf, the earliest runs can be a bit slow
        print(f"Running {x} shakedown inferences...")
        timing = run_model_x_times(model, job, x, profile=False)
        print("Running measured inference...")
        run_model_x_times(model, job, 1, profile=True)

        if not force_no_tuning:
            with open(timings, "w+") as f:
                f.write(json.dumps(timing, indent=2))