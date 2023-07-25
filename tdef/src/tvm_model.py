import tvm
import enum
import time
import sys
import json

from tvm.driver import tvmc
from tvm.target import cuda
from tvm.driver.tvmc.autotuner import autoscheduler_get_tuning_tasks, schedule_tasks
from tvm.driver.tvmc.compiler import parse_configs
from tvm.contrib import graph_executor as executor
import tvm.relay as relay
import tvm.auto_scheduler as auto_scheduler
from tvm.autotvm.tuner import XGBTuner
from tvm.autotvm.tuner import GATuner
from tvm.autotvm.tuner import RandomTuner
from tvm.autotvm.tuner import GridSearchTuner
from tvm import autotvm
from tvm.driver.tvmc.transform import parse_graph_transform_args, apply_graph_transforms
import numpy as np

# ============== TDEF imports ==============
from .utilities import ConfigureLog, DefaultLog, Profiler

def pick_tuner(tuner:str, task):
    tuner_obj = None
    if tuner == "xgb":
        tuner_obj = XGBTuner(task, loss_type="reg")
    elif tuner == "xgb_knob":
        tuner_obj = XGBTuner(
            task, loss_type="reg", feature_type="knob")
    elif tuner == "xgb_itervar":
        tuner_obj = XGBTuner(task, loss_type="reg",
                            feature_type="itervar")
    elif tuner == "xgb_curve":
        tuner_obj = XGBTuner(task, loss_type="reg",
                            feature_type="curve")
    elif tuner == "xgb_rank":
        tuner_obj = XGBTuner(task, loss_type="rank")
    elif tuner == "xgb_rank_knob":
        tuner_obj = XGBTuner(
            task, loss_type="rank", feature_type="knob")
    elif tuner == "xgb_rank_itervar":
        tuner_obj = XGBTuner(task, loss_type="rank",
                            feature_type="itervar")
    elif tuner == "xgb_rank_curve":
        tuner_obj = XGBTuner(
            task, loss_type="rank", feature_type="curve")
    elif tuner == "xgb_rank_binary":
        tuner_obj = XGBTuner(task, loss_type="rank-binary")
    elif tuner == "xgb_rank_binary_knob":
        tuner_obj = XGBTuner(
            task, loss_type="rank-binary", feature_type="knob")
    elif tuner == "xgb_rank_binary_itervar":
        tuner_obj = XGBTuner(
            task, loss_type="rank-binary", feature_type="itervar")
    elif tuner == "xgb_rank_binary_curve":
        tuner_obj = XGBTuner(
            task, loss_type="rank-binary", feature_type="curve")
    elif tuner == "ga":
        tuner_obj = GATuner(task, pop_size=50)
    elif tuner == "random":
        tuner_obj = RandomTuner(task)
    elif tuner == "gridsearch":
        tuner_obj = GridSearchTuner(task)
    else:
        raise ValueError("Invalid tuner: " + tuner)
    return tuner_obj

class Model():
    def __init__(self, job: dict):
        # self.shape = job["shape_dict"]
        frontend = tvmc.frontends.guess_frontend(job["path"])
        self.mod, self.params = frontend.load(job["path"], shape_dict={job["input_name"]: job["input_shape"]})
        # self.mod, self.params = frontend.load(job["path"])

    def compile(self, job: dict):
        config=parse_configs(None)
        with tvm.transform.PassContext(opt_level=job["opt_level"],
                                       config=config):
            transformed_mod = tvm.driver.tvmc.transform.apply_graph_transforms(self.mod, parse_graph_transform_args(locals()))
            if job["enable_autoscheduler"]:
                with auto_scheduler.ApplyHistoryBest(job["records"]):
                    config["relay_backend.use_auto_scheduler"] = True
                    self.lib = relay.build(
                        ir_mod=transformed_mod,
                        target=job["target"],
                        executor=relay.backend.Executor("graph"),
                        params=self.params
                    )
                    self.device = tvm.device(str(job["target"]), job["gpus"][0])
                    self.module = executor.GraphModule(self.lib["default"](self.device))
            else:
                with autotvm.apply_history_best(job["records"]):
                    self.lib = relay.build(transformed_mod, target=job["target"], params=self.params)
                    self.device = tvm.device(str(job["target"]), job["gpus"][0])
                    self.module = executor.GraphModule(self.lib["default"](self.device))

    def tune(self, job: dict) -> str:
        with tvm.transform.PassContext(opt_level=int(job["opt_level"])):

            hardware_params = None
            if job["enable_autoscheduler"]:
                hardware_params = auto_scheduler.HardwareParams(
                    num_cores=job["hardware_params_num_cores"],
                    vector_unit_bytes=job["hardware_params_vector_unit_bytes"],
                    cache_line_bytes=job["hardware_params_cache_line_bytes"],
                    max_shared_memory_per_block=job["hardware_params_max_shared_memory_per_block"],
                    max_local_memory_per_block=job["hardware_params_max_local_memory_per_block"],
                    max_threads_per_block=job["hardware_params_max_threads_per_block"],
                    max_vthread_extent=job["hardware_params_max_vthread_extent"],
                    warp_size=job["hardware_params_warp_size"],
                    target=job["hardware_params_target"],
                    target_host=job["hardware_params_target_host"]
                )
                print(hardware_params)

            runner_func = auto_scheduler.LocalRunner if job["enable_autoscheduler"] else autotvm.LocalRunner
            runner = runner_func(
                number=job["number"],
                repeat=job["repeat"],
                timeout=job["timeout"],
                min_repeat_ms=job["min_repeat_ms"]
            )

            transform_arguments = parse_graph_transform_args(locals())
            transformed_mod:tvm.IRModule = apply_graph_transforms(self.mod, transform_arguments)

            tuning_options = {}
            if job["enable_autoscheduler"]:
                tuning_options = auto_scheduler.TuningOptions(
                    num_measure_trials=job["trials"],
                    measure_callbacks=[auto_scheduler.RecordToFile(job["records"])],
                    runner=runner,
                    early_stopping=job["early_stopping"]
                )

                tasks, weights = autoscheduler_get_tuning_tasks(
                    mod=transformed_mod,
                    params=self.params,
                    target=job["target"],
                    transform_args=transform_arguments,
                    hardware_params=hardware_params,
                    include_simple_tasks=job["include_simple_tasks"]
                )
                try:
                    schedule_tasks(tasks, weights, tuning_options)
                except ValueError as e:
                    print(e)
            else:
                tuning_options = {
                    "tuner": job["tuner"],
                    "trials": job["trials"],
                    "early_stopping": job["early_stopping"],
                    "measure_option": autotvm.measure_option(
                        builder=autotvm.LocalBuilder(build_func="default"),
                        runner=runner
                    ),
                    "tuning_records": job["records"]
                }

                tasks = autotvm.task.extract_from_program(
                    mod = transformed_mod,
                    target=job["target"],
                    params=self.params
                )

                for i, task in enumerate(tasks):
                    prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
                    tuner_obj = pick_tuner(tuner = job["tuner"], task=task)
                    tuner_obj.tune(
                        n_trial=min(tuning_options["trials"], len(task.config_space)),
                        early_stopping=tuning_options["early_stopping"],
                        measure_option=tuning_options["measure_option"],
                        callbacks=[autotvm.callback.progress_bar(tuning_options["trials"], prefix=prefix),
                                autotvm.callback.log_to_file(job["records"])]
                    )

    def inferRandom(self, profile: bool = False):
        s = None
        out = []
        with Profiler(profile):
            s = time.time_ns()
            self.module.run()
            s = (time.time_ns() - s) / 1000000.0
        for o in range(0, self.module.get_num_outputs()):
            out.append(self.module.get_output(o))
        return s, out