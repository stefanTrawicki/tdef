import json
    
content = []
other_keys = {"gpus": [0]}

with open("job_description.json", 'r') as file:
    json_data = json.load(file)
    
    for trial in json_data["trials"]:
        for model, model_data in json_data["model"].items():
            for autoschedule in json_data["autoschedule"]:
                job = {
                    "model": model,
                    "trials": trial,
                    "enable_autoscheduler": autoschedule,
                    "path": model_data["path"],
                    "input_name": model_data["input_name"],
                    "input_shape": model_data["input_shape"]
                }
                job.update(other_keys)
                if not autoschedule:
                    for tuner in json_data["tuner"]:
                        temp = job.copy()
                        temp["tuner"] = tuner
                        content.append(temp)
                else:
                    content.append(job)

with open("job.json", "w+") as file:
    json.dump(content, file, indent=2)