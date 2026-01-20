from pathlib import Path
import sys
import os



if __name__ == "__main__":
    # add folder kernels to path
    sys.path.append(Path(__file__).parent / "kernels")