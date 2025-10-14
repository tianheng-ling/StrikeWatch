import re
import os
from pathlib import Path

from hw_converter.utils import deployability_check


def analyze_resource_utilization(report_dir: str):
    report_dir = Path(report_dir)
    report_info = {}
    try:
        for ext in [".par", ".mrp", ".srp"]:
            candidate = report_dir / f"utilization_report{ext}"
            if candidate.exists():
                report_path = candidate
                break
            else:
                print(
                    f"[Resource Analysis] No supported report file found in: {report_dir}"
                )
                return {"is_deployable": False}

        # ----- Handle .par -----
        if report_path.suffix == ".par":
            with open(report_path, "r") as f:
                for line in f:
                    if m := re.match(r"\s+(\S*)\s+(\d+)/(\d+)\s+\d+% used", line):
                        key, used, total = m.groups()
                        used_util = int(used) / int(total)
                        if key in ["LUT", "DSP", "EBR"]:
                            key = key.lower() + "s"
                            report_info[f"{key}_used"] = int(used)
                            report_info[f"{key}_used_util"] = round(used_util * 100, 2)
                            report_info["dsps_total"] = total
            report_info["is_deployable"] = deployability_check(report_info)
            return report_info
        # ----- Handle .mrp -----
        elif report_path.suffix == ".mrp":
            with report_path.open("rt") as f:
                for line in f:
                    for keyword, mapped_key in [
                        ("Number of LUT4s:", "luts"),
                        ("Number of DSPs:", "dsps"),
                        ("Number of EBRs:", "ebrs"),
                    ]:
                        if keyword in line:
                            try:
                                parts = line.split(":")
                                used_value = int(parts[1].split("out of")[0].strip())
                                total_value = int(
                                    parts[1].split("out of")[1].split("(")[0].strip()
                                )
                                report_info[f"{mapped_key}_used"] = used_value
                                report_info[f"{mapped_key}_used_util"] = round(
                                    used_value / total_value * 100, 2
                                )
                                report_info[f"{mapped_key}_total"] = total_value
                            except Exception as e:
                                print(f"[MRP Parse] Failed to parse '{keyword}': {e}")
            report_info["is_deployable"] = deployability_check(report_info)
            return report_info
        # ----- Handle .srp -----
        elif report_path.suffix == ".srp":
            with report_path.open("rt") as f:
                content = f.read()

            tmp = {}
            for line in content.splitlines():
                for keyword, key in [
                    ("Device Register Count .........:", "luts_total"),
                    ("Number of registers needed ....:", "luts_used"),
                    ("Device EBR Count ..............:", "ebrs_total"),
                    ("Used EBR Count ................:", "ebrs_used"),
                    ("Number of EBR Blocks Needed ...:", "ebrs_missing"),
                    ("Number of DSP Blocks:", "dsps_used"),
                ]:
                    if keyword in line:
                        try:
                            tmp[key] = int(line.split(":")[1].strip())
                        except Exception as e:
                            print(f"[SRP Parse] Failed to parse {key}: {e}")

            report_info["luts_total"] = tmp["luts_total"]
            report_info["luts_used"] = tmp["luts_used"]
            report_info["luts_used_util"] = round(
                tmp["luts_used"] / tmp["luts_total"] * 100, 2
            )

            report_info["ebrs_used"] = tmp["ebrs_used"]
            report_info["ebrs_total"] = tmp["ebrs_used"] + tmp["ebrs_missing"]
            report_info["ebrs_used_util"] = round(
                report_info["ebrs_used"] / tmp["ebrs_total"] * 100, 2
            )

            report_info["dsps_used"] = tmp["dsps_used"]
            report_info["dsps_total"] = 8  # only for ICE40UP5K
            report_info["dsps_used_util"] = round(
                tmp["dsps_used"] / tmp["dsps_total"] * 100, 2
            )
            report_info["is_deployable"] = deployability_check(report_info)
            return report_info

        # ----- Unsupported file type -----
        else:
            print(f"[Resource Analysis] Unsupported file type: {report_path.suffix}")
            return {"is_deployable": False}

    except Exception as e:
        print(f"[Resource Analysis] Failed to parse report '{report_path}': {e}")
        return {"is_deployable": False}


def analyze_power_consumption(report_dir: str):
    report_path = os.path.join(report_dir, "power_report.txt")

    try:
        if not Path(report_path).exists():
            print(f"[Power Analysis] File not found: {report_path}")
            return None

        power_values = {}
        with open(report_path, "rt") as f:
            power_report = f.read()

            match = re.search(
                r"Total Power Est.\ Design  : (.+) W, (.+) W, (.+) W", power_report
            )
            static, dynamic, total = match.groups()
            power_values["static_power(mW)"] = round(float(static) * 1000, 4)
            power_values["dynamic_power(mW)"] = round(float(dynamic) * 1000, 4)
            power_values["total_power(mW)"] = round(float(total) * 1000, 4)
        return power_values  # in mW
    except Exception as e:
        print(f"[Power Analysis] Failed to parse power report: {e}")
        return None
