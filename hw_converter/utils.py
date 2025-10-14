import os
import shutil


def copy_to_dir(src: str, dest: str):
    if os.path.isdir(src):
        shutil.copytree(
            src, os.path.join(dest, os.path.basename(src)), dirs_exist_ok=True
        )
    else:
        shutil.copy(src, dest)


def deployability_check(res_info: dict) -> bool:
    for key, val in res_info.items():
        if key.endswith("_used_util") and val > 100.0:
            return False
    return True


def get_radiant_env():
    radiant_home = (
        "/home/tianhengling/lscc/radiant/2023.2"  # TODO: make it configurable
    )
    env = os.environ.copy()
    env["RADIANT_HOME"] = radiant_home
    env["bindir"] = f"{radiant_home}/bin/lin64"
    env["PATH"] = f"{radiant_home}/bin/lin64:" + env.get("PATH", "")
    env["LM_LICENSE_FILE"] = f"{radiant_home}/license/license.dat"
    env["LD_LIBRARY_PATH"] = (
        f"{radiant_home}/bin/lin64:"
        f"{radiant_home}/ispfpga/bin/lin64:" + env.get("LD_LIBRARY_PATH", "")
    )
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["QT_DEBUG_PLUGINS"] = "1"
    return env


def remove_tmp_dir(dir_path: str):
    try:
        shutil.rmtree(dir_path)
        print(f"[Clean up] Removed temporary directory: {dir_path}")
    except Exception as e:
        print(f"[Warning] Failed to remove {dir_path}: {e}")
