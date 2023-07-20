from .src.tdef import TDEF
from .src.dump_model import dumpONNX, dumpPyTorch

import argparse
import json

def process(json_data):
    for trial in json_data["trials"]:
        for model, model_data in json_data["model"].items():
            for autoschedule in json_data["autoschedule"]:
                job = {
                    "trials": trial,
                    "gpus": [0, 1],
                    "enable_autoscheduler": autoschedule,
                    "path": model_data["path"],
                    "input_name": model_data["input_name"],
                    "input_shape": model_data["input_shape"],
                    "override_records": json_data.get("override_records", None)
                }
                if not autoschedule:
                    for tuner in json_data["tuner"]:
                        job["tuner"] = tuner
                        print(job)
                        TDEF(job=job, dry_run=False, sample=True)
                else:
                    print(job)
                    TDEF(job=job, dry_run=False, sample=True)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Input job')
    parser.add_argument('json_file', help='Path to the JSON file')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    with open(args.json_file) as f:
        json_data = json.load(f)

    process(json_data)