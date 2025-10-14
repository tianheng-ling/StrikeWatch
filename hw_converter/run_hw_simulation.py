import os

from hw_converter import (
    vivado_runner,
    radiant_runner,
)


def run_hw_simulation(
    model_save_dir: str,
    hw_config: dict,
):
    target_hw = hw_config["target_hw"]
    runner_kwargs = {
        "top_module": hw_config["top_module"],
        "base_dir": os.path.abspath(os.path.join(model_save_dir, "hw", target_hw)),
        "fpga_type": hw_config["fpga_type"],
        "clk_freq_mhz": 100 if target_hw == "amd" else (1 / 50) * 1000,
    }
    hw_runners = {
        "amd": vivado_runner,
        "lattice": radiant_runner,
    }
    if target_hw not in hw_runners:
        raise ValueError(f"Unsupported target_hw: {target_hw}")
    return hw_runners[target_hw](**runner_kwargs)
