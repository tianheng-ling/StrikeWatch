import os


def get_ghdl_simulation_time(
    ghdl_report_dir: str, target_module: str, clk_freq_mhz: int
):
    report_path = os.path.join(ghdl_report_dir, f"ghdl_{target_module}_output.txt")
    if not os.path.exists(report_path):
        raise FileNotFoundError(f"Report file not found: {report_path}")
    with open(report_path, "r") as f:
        lines = f.readlines()

    found_time = False

    # Step 1: check difference
    for line in lines:
        if "Differece" in line:
            parts = line.split("Differece =")
            if len(parts) > 1:
                diff_val = int(parts[1].strip())
                print(
                    f"Difference for {target_module} is {diff_val}. Check the simulation settings."
                )

    # Step 2: extract time
    for line in lines:
        if "Time taken for processing" in line:
            parts = line.split("=")
            latency_100Mhz_fs = (
                parts[1].split("fs")[0].strip()
            )  # assumed in 10 ns (100 MHz)

            # convert to the real lantency fololowing the set clk_freq_mhz
            clock_cycles = float(latency_100Mhz_fs) / 1e6 / 10
            cycle_period_ns = 1000 / clk_freq_mhz
            real_latency_ns = clock_cycles * cycle_period_ns

            found_time = True

            return latency_100Mhz_fs, real_latency_ns

        if "simulation stopped by --stop-time" in line:
            print(
                f"Simulation stopped by --stop-time in {target_module}. Check the simulation settings."
            )
            return None, None

    # Step 3: check for only warning/assertion lines
    if not found_time:
        if all(
            "assertion" in line or "warning" in line or "NUMERIC_STD" in line
            for line in lines
        ):
            print(
                f"No valid simulation result for {target_module}. Only warnings/assertions found."
            )
            return None, None

    return None, None
