import argparse
import json
import time
import subprocess
from tdef.src.tdef import TDEF
    
def run_subprocess(json_obj):
    # log_filename = f"{str(time.time()).replace('.', '')}.log"
    # json_str = json.dumps(json_obj).replace("'", '"')
    # wrapped = f"\'{json_str}\'"
    # command = "sudo PATH=/usr/local/cuda-11.7/bin:$PATH TVM_PATH=/home/s/cupti/learning/tuning_as_a_defence/tvm PYTHONPATH=/home/s/cupti/learning/tuning_as_a_defence/tvm/python LD_LIBRARY_PATH=/home/s/cupti/learning/samples/activity_trace_async/venv/lib/python3.8/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH /usr/local/cuda-11.7/bin/ncu --csv --log-file {} --metric lts__t_sectors_op_write.sum,lts__t_sectors_op_read.sum --target-processes all --profile-from-start no  $PWD/venv/bin/python -m tdef --json_string {}".format(
    #     log_filename,
    #     wrapped
    # )
    # command = f"$PWD/venv/bin/python -m tdef --json_string {wrapped}"
    # proc = subprocess.run([command], input=json.dumps(json_obj), text=True, capture_output=True, shell=True)
    # print(proc)
    # print(proc.stdout[-len("8e286d60b336f50013077b465d7fa654336d8b418405e0c873a57d3b616fef54/\n"):-1])
    # try:
    TDEF(json_obj)
    # except Exception as e:
    #     print("error")
    #     print(e.messages)
    
def main():
    parser = argparse.ArgumentParser(description='Process JSON data and pass it to a subprocess.')
    parser.add_argument('--json_file', type=str, help='Path to the JSON file containing an array of JSON objects.')
    args = parser.parse_args()

    with open(args.json_file, 'r') as file:
        data = json.load(file)
        for _, json_obj in enumerate(data):
            run_subprocess(json_obj)
            print(f"Job Progress [{_}/{len(data)}]")

main()