import os
from pathlib import Path

from hw_converter.utils import deployability_check


def clean_resource_key(key):
    key_mapping = {
        "Slice_LUTs": "luts",
        "LUT_as_Memory": "luts_mem",
        "Block_RAM_Tile": "brams",
        "DSPs": "dsps",
    }
    cleaned_key = key.strip("| ").replace(" ", "_")
    return key_mapping.get(cleaned_key, cleaned_key)


def clean_power_key(key):
    key_mapping = {
        "Total_On-Chip_Power_(W)": "total_power(mW)",
        "Dynamic_(W)": "dynamic_power(mW)",
        "Device_Static_(W)": "static_power(mW)",
    }
    cleaned_key = key.strip("| ").replace(" ", "_")
    return key_mapping.get(cleaned_key, cleaned_key)


def analyze_resource_utilization(report_dir: str):
    report_path = os.path.join(report_dir, "utilization_report.txt")
    keywords = ["| Slice LUTs", "|   LUT as Memory", "| Block RAM Tile", "| DSPs"]

    try:
        if not Path(report_path).exists():
            print(f"[Resource Analysis] File not found: {report_path}")
            return {"is_deployable": False}

        report_info = {}
        with open(report_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                for keyword in keywords:
                    if keyword in line:
                        parts = line.split("|")
                        used_value = (
                            float(parts[2].strip()) if parts[2].strip() else 0.0
                        )
                        total_value = (
                            float(parts[4].strip()) if parts[4].strip() else 0.0
                        )
                        utils_value = (
                            float(parts[5].strip()) if parts[5].strip() else 0.0
                        )

                        cleaned_keyword = clean_resource_key(keyword)
                        report_info[cleaned_keyword + "_used"] = used_value
                        report_info[cleaned_keyword + "_total"] = total_value
                        report_info[cleaned_keyword + "_used_util"] = utils_value

        report_info["is_deployable"] = deployability_check(report_info)
        return report_info
    except Exception as e:
        print(f"[Resource Analysis] Failed to parse resource report: {e}")
        return {"is_deployable": False}


def analyze_power_consumption(report_dir: str):
    report_path = os.path.join(report_dir, "power_report.txt")
    keywords = ["Total On-Chip Power (W)", "Dynamic (W)", "Device Static (W)"]

    try:
        if not Path(report_path).exists():
            print(msg=f"[Power Analysis] File not found: {report_path}")
            return None

        power_values = {}
        with open(report_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                for keyword in keywords:
                    if keyword in line:
                        parts = line.split("|")
                        value = float(parts[2].strip()) * 1000  # W → mW
                        cleaned_keyword = clean_power_key(keyword)
                        power_values[cleaned_keyword] = value
        return power_values
    except Exception as e:
        print(f"[Power Analysis] Failed to parse power report: {e}")
        return None
