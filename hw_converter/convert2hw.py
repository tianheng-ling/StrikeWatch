import os
import torch
import shutil
from pathlib import Path
from torch.utils.data import Subset

from config import DEVICE, STOP_TIME
from models.build_model import build_model
from torch.utils.data import DataLoader, Dataset
from hw_converter.firmware import GetFirmware
from elasticai.creator.file_generation.on_disk_path import OnDiskPath
from elasticai.creator.nn.integer.vhdl_test_automation import create_makefile


def convert2hw(
    model_type: str,
    test_dataset: Dataset,
    subset_size: int,
    model_config: dict,
    model_save_dir: str,
    target_hw: str,
) -> None:

    # get test dataloader slice
    test_dataset = Subset(test_dataset, list(range(subset_size)))
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=subset_size,
        shuffle=False,
        drop_last=True,
    )

    # set up directories
    hw_dir = Path(model_save_dir) / "hw" / target_hw
    if hw_dir.exists():
        shutil.rmtree(hw_dir)
    hw_dir.mkdir(parents=True, exist_ok=True)
    quant_data_dir = os.path.join(hw_dir, "data")
    os.makedirs(quant_data_dir, exist_ok=True)
    model_config.update({"quant_data_dir": quant_data_dir})

    # build model and load weights
    if model_type == "rnn":
        model_config.update({"batch_size": subset_size})
    model = build_model(model_config, enable_qat=True).to(DEVICE)
    checkpoint = torch.load(model_save_dir / "best_model.pth", weights_only=True)
    model.load_state_dict(checkpoint, strict=False)

    # get and save quantized data for HW simulation
    model.eval()
    with torch.no_grad():
        for _, (features, target) in enumerate(test_dataloader):
            features = features.to(DEVICE)
            target = target.to(DEVICE)
            _ = model(inputs=features)

    # transform model to HW network_design
    if model_type != "transformer":
        network_design = model.sequential.create_design(model_config["name"])
    else:
        network_design = model.create_design(model_config["name"])
    network_design.save_to(OnDiskPath(name=f"source", parent=str(hw_dir)))

    # get makefile
    create_makefile(hw_dir, stop_time=STOP_TIME)

    # get firmware for HW simulation
    firmware = GetFirmware(
        hw_version={
            "amd": "env5",
            "lattice": "env5se",
        }[target_hw],
        network_design=network_design,
    )
    firmware.save_to(destination_dir=hw_dir)
