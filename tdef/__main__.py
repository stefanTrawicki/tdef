from .src.utilities import LoggingSetup
from .src.tdef import TDEF

# There is an issue at the moment where importing TVM and torch causes a free pointer
# error as they both have overlapping symbols, didn't know it was possible.
# from .src.dump_model import getPyTorchModels
# getPyTorchModels()

TDEF()