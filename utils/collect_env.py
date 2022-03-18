import torch
from detectron2.utils.collect_env import collect_env_info, test_nccl_ops

if __name__ == "__main__":
    
    print(collect_env_info())

    if torch.cuda.is_available():
        num_gpu = torch.cuda.device_count()
        for k in range(num_gpu):
            device = f"cuda:{k}"
            try:
                x = torch.tensor([1, 2.0], dtype=torch.float32)
                x = x.to(device)
            except Exception as e:
                print(
                    f"Unable to copy tensor to device={device}: {e}. "
                    "Your CUDA environment is broken."
                )
        if num_gpu > 1:
            test_nccl_ops()