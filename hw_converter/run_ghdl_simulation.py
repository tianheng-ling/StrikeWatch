import os
import shutil
import subprocess
from pathlib import Path

from hw_converter.utils import copy_to_dir
from .utils import remove_tmp_dir


def run_ghdl_simulation(
    base_dir: str,
    source_dir: str,
    data_dir: str,
    makefile_path: str,
):

    tmp_ghdl_dir = os.path.join(base_dir, "tmp_ghdl_proj")
    ghdl_report_dir = os.path.join(base_dir, "ghdl_report")
    for path in [tmp_ghdl_dir, ghdl_report_dir]:
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)

    copy_to_dir(src=source_dir, dest=tmp_ghdl_dir)
    copy_to_dir(src=data_dir, dest=tmp_ghdl_dir)
    copy_to_dir(src=makefile_path, dest=tmp_ghdl_dir)

    tmp_ghdl_source_dir = os.path.join(tmp_ghdl_dir, "source")
    tb_files = list(Path(tmp_ghdl_source_dir).rglob("*_tb.vhd"))

    for tb_file in tb_files:
        module = tb_file.stem
        if module.endswith("_tb"):
            module = module[:-3]

        subprocess.run(["make", f"TESTBENCH={module}"], cwd=tmp_ghdl_dir)

        sim_output_path = os.path.join(tmp_ghdl_dir, ".simulation", "make_output.txt")
        report_output_path = os.path.join(ghdl_report_dir, f"ghdl_{module}_output.txt")
        if os.path.exists(sim_output_path):
            shutil.copy(sim_output_path, report_output_path)

    remove_tmp_dir(tmp_ghdl_dir)
