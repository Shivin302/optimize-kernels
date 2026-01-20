from pathlib import Path
from rms_norm.baseline import load_inputs, numpy_kernel

if __name__ == "__main__":
    input_arrays = load_inputs(Path(__file__).parent / "rms_norm" / "inputs.json")
    for input_array in input_arrays:
        output = numpy_kernel(**input_array)
        print(output.shape, output.dtype)