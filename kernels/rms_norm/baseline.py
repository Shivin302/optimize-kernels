import numpy as np
import json
from typing import Dict, Any
from pathlib import Path


def load_inputs(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        input_configs = json.load(f)
    input_arrays = [{}] * len(input_configs)
    for i, input_config in enumerate(input_configs):
        for key, value in input_config.items():
            input_arrays[i][key] = np.random.randn(*value["shape"]).astype(value["dtype"])
    return input_arrays

def numpy_kernel(x: np.ndarray, gamma: np.ndarray, epsilon: float) -> np.ndarray:
    return x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + epsilon) * gamma

if __name__ == "__main__":
    input_arrays = load_inputs(Path(__file__).parent / "inputs.json")
    for input_array in input_arrays:
        output = numpy_kernel(**input_array)
        print(output.shape, output.dtype)