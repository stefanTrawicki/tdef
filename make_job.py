import json
    
with open("job_description.json", 'r') as file:
    json_data = json.load(file)
    
with open("job.json", "w+") as file:
    content = []
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
                if not autoschedule:
                    for tuner in json_data["tuner"]:
                        job["tuner"] = tuner
                        content.append(job)
                else:
                    content.append(job)
    json.dump(content, file, indent=2)