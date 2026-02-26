import os, yaml
import pandas as pd
import subprocess

def main():
    # Reuse src/02_build_state_vector.py by swapping config temporarily is annoying,
    # so do the clean approach: copy your state_vector.yaml block into a new file.
    # This script assumes you create configs/state_vector_kepler_field.yaml.
    subprocess.check_call(["python", "src/02_build_state_vector.py"], env={**os.environ, "SHV_CFG":"configs/state_vector_kepler_field.yaml"})

if __name__ == "__main__":
    main()
