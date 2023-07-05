import os
import logging
import json
import torch
# from onnx import hub
# from enum import Enum

# class Frameworks(Enum):
#     PYTORCH = 0,
#     TENSORFLOW = 1,
#     ONNX = 3


# for TF models: (don't open in safari)
# https://tfhub.dev/s?module-type=image-classification&tf-version=tf2
# for onnx models:
# https://github.com/onnx/models/tree/main/vision/classification


# def dumpTensorFlow(file_path: str, model: str, input_shape: list = None):
#     print("manually download TF models")
#     return {}


# def dumpONNX(file_path: str, model: str, input_shape: list = None):
#     hub.set_dir("dumps")
#     model = hub.load(model)


# handlers = {
#     Frameworks.PYTORCH: dumpPyTorch,
#     Frameworks.ONNX: dumpONNX
# }


# def dumpModel(model: str, framework: Frameworks):
#     f = handlers[framework]
#     file_path = "dumps/" + f"{model}.pth.zip"
#     if not os.path.isfile(file_path):
#         f(file_path=file_path, model=model, input_shape=None)
#     return file_path


def getPyTorchModels():
    logging.warning(
        "Fetching PyTorch models from TorchHub. Importing TVM & PyTorch WILL cause a segfault, be aware.")
    models = [
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        "vgg16",
        "vgg13",
        "vgg19",
        "alexnet",
        "mobilenet_v2"
    ]
    for m in models:
        file_name = "dumps/pytorch/" + m + ".pth.zip"
        if not os.path.exists(file_name):
            dumpPyTorch(file_name, m)
            logging.info(f"Downloaded {m} to disk")


def dumpPyTorch(file_path: str, model: str):
    input_shape = [1, 3, 224, 224]
    model = torch.hub.load("pytorch/vision:v0.10.0", model, pretrained=True)
    model.eval()
    shape_dict = {[name for name, _ in model.named_modules()][0]: input_shape}
    input_data = torch.randn(input_shape)
    model = model.to("cpu")
    input_data = input_data.to("cpu")
    # https://discuss.tvm.apache.org/t/import-scripted-instead-of-traced-pytorch-model/9511
    scripted_model = torch.jit.trace(model, input_data).eval()
    scripted_model.save(file_path)
    with open(f"{file_path}.json", "w+") as f:
        json.dump(shape_dict, f)