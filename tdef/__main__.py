from .src.tdef import TDEF
from .src.dump_model import dumpONNX, dumpPyTorch

import argparse
import json

example = """
    {
        "trials": 50,
        "tuner": "random",
        "model": "resnet18",
        "path": "dumps/onnx/frozen/resnet18-v1-7.onnx",
        "input_name": "data",
        "input_shape": [1, 3, 224, 224]
    }
    """

# python -m tdef --json_string <this>
example_as_str = '{"trials": 50,"tuner": "random","model": "resnet18","path": "dumps/onnx/frozen/resnet18-v1-7.onnx","input_name": "data","input_shape": [1, 3, 224, 224]}'

def main(data:dict, force_notune=False, baselines=False):
    return TDEF(job_json=data, force_no_tuning=force_notune, collecting_baselines=baselines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse JSON string from command line.")
    parser.add_argument("--json_string", type=str, help="JSON string to be parsed", required=True)
    parser.add_argument("--notune", action="store_true", help="Whether to force disable tuning", default=False)
    parser.add_argument("--baselines", action="store_true", help="Whether to run baseline", default=False)
    args = parser.parse_args()
    
    json_string = args.json_string
    
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as e:
        print("Error: Invalid JSON format.")
        print(e)
        exit

    main(data, args.notune, args.baselines)