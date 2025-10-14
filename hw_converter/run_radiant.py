import os
import shutil
import subprocess

from hw_converter.run_ghdl_simulation import run_ghdl_simulation
from hw_converter.analyze_ghdl_report import (
    get_ghdl_simulation_time,
)
from hw_converter.analyze_radiant_report import (
    analyze_resource_utilization,
    analyze_power_consumption,
)
from hw_converter.utils import copy_to_dir, get_radiant_env


def run_resource_estimation(
    tmp_dir: str,
    firmware_dir: str,
    report_dir: str,
    source_dir: str,
    const_dir: str,
    data_dir: str,
    tcl_path: str,
    env: dict,
):
    try:
        # copy necessary files to tmp_dir
        copy_to_dir(src=source_dir, dest=tmp_dir)
        copy_to_dir(src=data_dir, dest=tmp_dir)
        copy_to_dir(src=const_dir, dest=tmp_dir)

        # replacing the testbed data path
        subprocess.run(f"cp -r {firmware_dir}/* {tmp_dir}/source", shell=True)
        subprocess.run(["cp", "-r", tcl_path, tmp_dir])
        subprocess.run(
            ["pnmainc", "resource_estimation.tcl", tmp_dir, "env5se_top_reconfig"],
            env=env,
            cwd=tmp_dir,
        )

        report_priority = [
            ("par", "radiant_project_impl_1.par"),
            ("mrp", "radiant_project_impl_1.mrp"),
            ("srp", "radiant_project_impl_1.srp"),
        ]

        for ext, filename in report_priority:
            report_path = os.path.join(tmp_dir, "impl_1", filename)
            if os.path.exists(report_path):
                dst_path = os.path.join(report_dir, f"utilization_report.{ext}")
                subprocess.run(["cp", report_path, dst_path])
                break
            else:
                raise FileNotFoundError(f"{filename} not found in {tmp_dir}/impl_1")
    except Exception as e:
        print(f"[Resource Estimation] Failed for top module: {e}")


def run_power_simulation(
    tmp_dir: str,
    report_dir: str,
    time_fs: int,
    env: dict,
    power_sim_tcl: str,
    power_est_tcl: str,
):
    try:
        power_sim_dir = os.path.join(tmp_dir, "power_simulation")
        os.makedirs(power_sim_dir, exist_ok=True)

        subprocess.run(["cp", "-r", power_sim_tcl, tmp_dir])
        subprocess.run(["cp", "-r", power_est_tcl, tmp_dir])

        tcl_file_path = os.path.join(tmp_dir, "power_simulation.tcl")
        vo_path = os.path.join(tmp_dir, "impl_1/radiant_project_impl_1_vo.vo")
        if os.path.exists(vo_path):
            subprocess.run(
                [
                    "sed",
                    "-i",
                    f"s/run 2989125 ns/run {time_fs} fs/g",
                    tcl_file_path,
                ]
            )
            radiant_root = f"/home/tianhengling/lscc/radiant/2023.2"
            vsim_bin = os.path.join(radiant_root, "modeltech/linuxloem/vsim")
            subprocess.run(
                [vsim_bin, "-c", "-do", "power_simulation.tcl"], env=env, cwd=tmp_dir
            )
            subprocess.run(["pnmainc", "power_estimation.tcl"], env=env, cwd=tmp_dir)
            subprocess.run(
                ["cp", os.path.join(tmp_dir, "power_report.txt"), report_dir]
            )
        else:
            raise FileNotFoundError(".vo file missing — cannot run power simulation")

    except Exception as e:
        print(f"[Power Simulation] Failed for top module: {e}")


def radiant_runner(
    base_dir: str, top_module: str, fpga_type: str, clk_freq_mhz: int = 16
):

    # Setup dirs
    data_dir = os.path.join(base_dir, "data")
    source_dir = os.path.join(base_dir, "source")
    makefile_path = os.path.join(base_dir, "makefile")
    const_dir = os.path.join(base_dir, "constraints")
    firmware_dir = os.path.join(base_dir, "firmware")
    lattice_tcl_dir = "hw_converter/tcl_files/lattice/"
    resource_tcl_path = f"{lattice_tcl_dir}resource_estimation.tcl"
    power_sim_tcl = f"{lattice_tcl_dir}power_simulation.tcl"
    power_est_tcl = f"{lattice_tcl_dir}power_estimation.tcl"

    # Initialize hw_metrics
    hw_metrics = {
        "res_info": {},
        "time(ms)": None,
        "power_info": {},
        "energy(muJ)": None,
    }

    # [1] Run GHDL simulation and get simulation time
    run_ghdl_simulation(
        base_dir=base_dir,
        source_dir=source_dir,
        data_dir=data_dir,
        makefile_path=makefile_path,
    )
    latency_100Mhz_fs, real_latency_ns = get_ghdl_simulation_time(
        ghdl_report_dir=os.path.join(base_dir, "ghdl_report"),
        target_module=top_module,
        clk_freq_mhz=clk_freq_mhz,
    )
    if latency_100Mhz_fs is None:
        return hw_metrics
    else:
        hw_metrics["time(ms)"] = real_latency_ns / 1e6  # ns → ms

        # [2] Estimate resource utilization and parse resource utilization report
        tmp_dir = os.path.join(base_dir, "tmp_radiant_proj")
        radiant_report_dir = os.path.join(base_dir, "radiant_report")
        for path in [tmp_dir, radiant_report_dir]:
            shutil.rmtree(path, ignore_errors=True)
            os.makedirs(path, exist_ok=True)
        run_resource_estimation(
            tmp_dir=tmp_dir,
            firmware_dir=firmware_dir,
            report_dir=radiant_report_dir,
            source_dir=source_dir,
            const_dir=const_dir,
            data_dir=data_dir,
            tcl_path=resource_tcl_path,
            env=get_radiant_env(),
        )
        hw_metrics["res_info"] = analyze_resource_utilization(
            report_dir=radiant_report_dir
        )
        if hw_metrics["res_info"].get("is_deployable") == False:
            return hw_metrics
        else:
            # [3] Power estimation
            run_power_simulation(
                tmp_dir=tmp_dir,
                report_dir=radiant_report_dir,
                time_fs=latency_100Mhz_fs,
                env=get_radiant_env(),
                power_sim_tcl=power_sim_tcl,
                power_est_tcl=power_est_tcl,
            )
            power_info = analyze_power_consumption(report_dir=radiant_report_dir)
            if power_info is None:
                return hw_metrics
            else:
                hw_metrics["power_info"] = power_info

                # [4] Calculate energy consumption -> muJ
                hw_metrics["energy(muJ)"] = power_info["total_power(mW)"] * (
                    real_latency_ns / 1e6
                )
                return hw_metrics
