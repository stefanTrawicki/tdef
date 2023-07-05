import tvm
import logging
import enum
import time

from tvm.driver import tvmc
from tvm.target import cuda
from tvm.contrib import graph_executor as executor
import tvm.relay as relay
import tvm.auto_scheduler as auto_scheduler
from tvm.autotvm.tuner import XGBTuner
from tvm.autotvm.tuner import GATuner
from tvm.autotvm.tuner import RandomTuner
from tvm.autotvm.tuner import GridSearchTuner
from tvm import autotvm
import numpy as np

# ============== TDEF imports ==============
from .cuda import start, stop


class Model():
    # path:str, shape_dict:dict, n_classes:dict
    def __init__(self, job: dict, profile: bool = False):
        logging.info(f'Loading model {job["path"]}, {job["shape_dict"]}')
        self.shape = job["shape_dict"]
        frontend = tvmc.frontends.guess_frontend(job["path"])
        self.mod, self.params = frontend.load(job["path"], job["shape_dict"])
        logging.info(f'Loaded {frontend} model')

    # records:str, target:str="cuda -arch=sm_80", opt_level:int=4
    def compile(self, job: dict, profile: bool = False):
        with autotvm.apply_history_best(job["records"]):
            with tvm.transform.PassContext(opt_level=job["opt_level"]):
                # with self.profiler as profiler:
                self.lib = relay.build(self.mod, target=job["target"], params=self.params)

        self.device = tvm.device(str(job["target"]), 0)
        self.module = executor.GraphModule(self.lib["default"](self.device))

    # number:int, repeat:int, timeout:int, min_repeat_ms:int, enable_cpu_cache_flush:True
    # tuner:str, trials: int, early_stopping:True, records:str
    def tune(self, job: dict, profile: bool = False) -> str:
        runner = autotvm.LocalRunner(
            number=job["number"],
            repeat=job["repeat"],
            timeout=job["timeout"],
            min_repeat_ms=job["min_repeat_ms"],
            enable_cpu_cache_flush=True
        )

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
            self.mod["main"], target=job["target"], params=self.params)

        for i, task in enumerate(tasks):
            prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
            tuner = job["tuner"]
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

            tuner_obj.tune(
                    n_trial=min(tuning_options["trials"], len(task.config_space)),
                    early_stopping=tuning_options["early_stopping"],
                    measure_option=tuning_options["measure_option"],
                    callbacks=[autotvm.callback.progress_bar(tuning_options["trials"], prefix=prefix),
                               autotvm.callback.log_to_file(tuning_options["tuning_records"])]
                )

    def inferRandom(self, profile: bool = False):
        input = np.random.rand(1, 3, 224, 224)
        self.module.set_input("", input)

        start()
        self.module.run()
        stop()