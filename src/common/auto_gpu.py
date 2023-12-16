import pynvml, re, os
def auto_gpu():
    pynvml.nvmlInit()
    gpu_utils = []
    gpu_list = [i for i in range(pynvml.nvmlDeviceGetCount())]
    for i in gpu_list:
        gpu_state = os.popen(f'nvidia-smi -i {i}').read()
        gpu_util = int(re.findall('(\d+)(?=\s*%)', gpu_state)[0])
        gpu_utils.append(gpu_util)
    gpu_id = gpu_list[gpu_utils.index(min(gpu_utils))]
    return gpu_id