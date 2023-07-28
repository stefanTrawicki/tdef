import argparse
import json
import time
import os
import sys
import subprocess
from tdef.src.converter import convert

BASE = os.getenv("PWD")
TVM_LOC, _ = os.path.split(BASE)
PYTHON_PATH = os.path.join(TVM_LOC, "python")
keys_to_keep = ["model", "trials", "enable_autoscheduler", "path", "input_name", "input_shape", "gpus", "tuner", "opt_level"]

def run_tests(notune):
    get_subs = lambda directory : [ f.path for f in os.scandir(directory) if f.is_dir() ]
    subfolders = get_subs("records")
    checked = []
    for s in subfolders:
        for exp in get_subs(s):
            with open(os.path.join(exp, "metadata.json"), "r") as f:
                info = json.load(f)
                timings_path = os.path.join(exp, "timings.json")
                if not os.path.exists(timings_path):
                    print("Timing info not present, probably hasn't been tuned yet, skipping!")
                    continue

                logs_path = os.path.join(exp, "ncu_logs.log")
                if os.path.exists(logs_path):
                    _, m = os.path.split(exp)
                    print(f"Already assessed this model {info['model']}/{m!a}, skipping!")
                    continue

                checked.append(exp)
                info = {k: info[k] for k in keys_to_keep}
                json_str = json.dumps(info).replace("'", '"')
                wrapped_json_str = f"\'{json_str}\'"
                ncu_command = "sudo CUDA_VISIBLE_DEVICES=1 PATH=/usr/local/cuda-11.7/bin:$PATH TVM_PATH={} PYTHONPATH={} LD_LIBRARY_PATH=/home/s/cupti/learning/samples/activity_trace_async/venv/lib/python3.8/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH /usr/local/cuda-11.7/bin/ncu --csv --log-file {} --metric lts__t_sectors_op_write.sum,lts__t_sectors_op_read.sum --target-processes all --profile-from-start no {}/venv/bin/python -m tdef {} --json_string {}".format(
                    TVM_LOC,
                    PYTHON_PATH,
                    logs_path,
                    BASE,
                    "--notune" if notune else "",
                    wrapped_json_str
                )
                print(ncu_command)
                process = subprocess.Popen([ncu_command], stdout=subprocess.PIPE, shell=True)
                for c in iter(lambda: process.stdout.read(1), b""):
                    sys.stdout.buffer.write(c)

                mem_output_name = os.path.join(exp, "memory_logs.log")
                time_output_name = os.path.join(exp, "time_logs.log")
                try:
                    convert(logs_path, mem_output_name, time_output_name)
                except Exception as e:
                    print(e)

    if len(checked):
        print("Assessed: ")
        for c in checked:
            print(f"\t{c}")
    else:
        print("Did not assess any models!")

def run_tuning(json_obj):
    from tdef.src.tdef import TDEF
    TDEF(json_obj)

def main():
    parser = argparse.ArgumentParser(description='Process JSON data and pass it to a subprocess.')
    parser.add_argument('--json_file', type=str, help='Path to the JSON file containing an array of JSON objects.', required=False)
    parser.add_argument('--run_tests', action='store_true')
    parser.add_argument('--no_run_tests', dest='run_tests', action='store_false')
    parser.add_argument('--override_opt_level', type=int, help='If compiling without records, will override opt level.', default=3, required=False)
    parser.add_argument("--notune", action="store_true", help="Whether to force disable tuning (run baseline)")
    parser.set_defaults(run_tests=False, notune=False)
    args = parser.parse_args()


    if args.run_tests:
        run_tests(args.notune)
    else:
        with open(args.json_file, 'r') as file:
            data = json.load(file)
            for i, json_obj in enumerate(data):
                print(f"Job Progress [{i}/{len(data)}]")
                run_tuning(json_obj)

main()