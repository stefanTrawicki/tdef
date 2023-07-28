# TVM Harness

1. Write a job description JSON containing the evaluation matrix.

2. Use `make_job.py` to turn this into a job list.

3. Use `CUDA_VISIBLE_DEVICES=0,1 driver.py --json_file <job.json>` to run the jobs. Use whatever GPUs are appropriate.

4. The dir `records` will be created, each model ran makes a folder, i.e `resnet18`. Each job ran creates a dir with a hash name containing the metadata, tuning records and information about execution time.