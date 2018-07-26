from ID_Optimizer import ParameterBatch
from ID_CNN_V01 import setup_thread_environment
from _utils.ID_utils import check_available_gpus

"""
This runs the network testing, just using the default hyperparameters in ID_Optimizer.ParameterBatch
"""
args = ParameterBatch()
args.gpu_id = check_available_gpus()[0]
args.training = False
args.n_max_epochs = 30
acc = setup_thread_environment(args)
print(acc)