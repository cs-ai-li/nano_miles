import time
from typing import Any, Dict
import requests
import torch
import torch.nn as nn
from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket


def sync_weights_to_sglang(
    model: nn.Module,
    sglang_url: str = "http://127.0.0.1:30000",
    flush_cache: bool = True,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """
    Zero-disk CUDA IPC weight hot-update protocol.
    Streams weights from PyTorch model to colocated SGLang server in memory via CUDA IPC.

    Safety contracts:
      1. Open session with begin_weight_update.
      2. Stream sync with torch.cuda.synchronize().
      3. Tied weights: explicitly include lm_head.weight if tied.
      4. Keep bucket references alive until update_weights_from_tensor returns 200 OK.
      5. flush_cache=True to invalidate RadixCache.
      6. Close session with end_weight_update.
    """
    t0 = time.time()
    session = requests.Session()

    # 1. Open weight-update session
    resp = session.post(
        f"{sglang_url}/begin_weight_update",
        json={"selector": "all", "sync_base": True},
        timeout=timeout,
    )
    if resp.status_code != 200 or not resp.json().get("success", False):
        raise RuntimeError(f"begin_weight_update failed: HTTP {resp.status_code} {resp.text}")

    # 2. Collect model weights & handle tied weights
    named_tensors = [(name, p.detach().contiguous()) for name, p in model.named_parameters()]
    param_names = {n for n, _ in named_tensors}
    if hasattr(model, "lm_head") and "lm_head.weight" not in param_names:
        named_tensors.append(("lm_head.weight", model.lm_head.weight.detach().contiguous()))

    # 3. Create FlattenedTensorBucket and serialize to CUDA IPC handles
    bucket = FlattenedTensorBucket(named_tensors=named_tensors)
    bucket_data = {
        "flattened_tensor": bucket.get_flattened_tensor(),
        "metadata": bucket.get_metadata(),
    }

    # Ensure GPU writes are fully visible in device memory
    torch.cuda.synchronize()
    serialized_str = MultiprocessingSerializer.serialize(bucket_data, output_str=True)

    # 4. Dispatch IPC handle via HTTP while keeping bucket_data alive in GPU memory
    try:
        resp = session.post(
            f"{sglang_url}/update_weights_from_tensor",
            json={
                "serialized_named_tensors": [serialized_str],
                "load_format": "flattened_bucket",
                "flush_cache": flush_cache,
            },
            timeout=timeout,
        )
        if resp.status_code != 200 or not resp.json().get("success", False):
            raise RuntimeError(f"update_weights_from_tensor failed: HTTP {resp.status_code} {resp.text}")
    finally:
        # 5. Close session before releasing tensor references
        session.post(f"{sglang_url}/end_weight_update", json={}, timeout=timeout)
        del bucket_data
        del bucket
        del named_tensors

    torch.cuda.synchronize()
    latency = time.time() - t0
    return {"success": True, "latency_seconds": latency}
