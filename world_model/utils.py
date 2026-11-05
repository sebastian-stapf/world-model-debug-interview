import hashlib
import json
from pathlib import Path
import platform
import random

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def load_config(path):
    return json.loads(Path(path).read_text())


def to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def provenance(data, config, device):
    sources = {}
    for path in sorted(Path(__file__).parent.glob("*.py")):
        sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "config": config, "device": device,
        "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        "python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
        "data_sha256": hashlib.sha256(Path(data).read_bytes()).hexdigest(), "source_sha256": sources,
    }


def load_checkpoint(path, device):
    from .model import WorldModel
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = WorldModel(checkpoint["config"]["width"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["config"]
