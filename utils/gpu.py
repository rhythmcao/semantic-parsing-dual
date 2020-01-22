#coding=utf8
import os, sys, math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# special case in our remote server, just ignore
if '/cm/local/apps/cuda/libs/current/pynvml' in sys.path:
    sys.path.remove('/cm/local/apps/cuda/libs/current/pynvml')
import gpustat
import torch

def set_torch_device(deviceId):
    # Simplified version of gpu selection
    if deviceId < 0:
        device = torch.device("cpu")
        print('Use CPU ...')
    else:
        assert torch.cuda.device_count() >= deviceId + 1
        device = torch.device("cuda:%d" % (deviceId))
        print('Use GPU with index %d' % (deviceId))
        # os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # used when debug
        ## These two sentences are used to ensure reproducibility with cudnnbacken
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    return device

if __name__ == '__main__':

    set_torch_device(0)