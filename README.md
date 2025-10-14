# StrikeWatch
![Gait Recognition](https://img.shields.io/badge/Gait--Recognition-Forefoot%20%7C%20Heel-critical) ![FPGA](https://img.shields.io/badge/FPGA-Low--Power-blue) ![Quantization](https://img.shields.io/badge/Quantization-Integer--Only--Inference-green)

#### Overview

**StrikeWatch** is a wrist-worn gait recognition system that distinguishes **forefoot** vs. **heel strikes** using quantized deep learning models on **low-power embedded FPGAs**. This repository contains:
- 📊 A self-collected **IMU dataset** from 12 participants during real-world outdoor running
- 🧠 Lightweight **1D-CNN**,**1D-SepCNN**, **LSTM** and **Transformer** models for on-device classification  
- 🧱 Automated **RTL code generation and FPGA synthesis** for deployment
- ⚙️ Integrated **hardware-aware hyperparameter optimization** using Optuna

> ⚠️ This repository works in tandem with our [ElasticAI.Creator](https://github.com/es-ude/elastic-ai.creator/tree/add-linear-quantization) library, which provides the core VHDL templates and quantization modules for hardware generation. Please make sure to install it as part of the setup process.

---

#### Corresponding Paper
*StrikeWatch: Wrist-worn Gait Recognition with Compact Time-series Models on Low-power FPGAs*，which was accepted at IEEE Annual Congress on Artificial Intelligence of Things, Osaka, Japan, Dec 3–5, 2025. 

> **Abstract**  Running offers substantial health benefits, but improper gait patterns can lead to injuries, particularly without expert feedback. While prior gait analysis systems based on cameras, insoles, or body-mounted sensors have demonstrated effectiveness, they are often bulky and limited to offline, post-run analysis. Wrist-worn wearables offer a more practical and non-intrusive alternative, yet enabling real-time gait recognition on such devices remains challenging due to noisy Inertial Measurement Unit (IMU) signals, limited computing resources, and dependence on cloud connectivity.
This paper introduces StrikeWatch, a compact wrist-worn system that performs entirely on-device, real-time gait recognition using IMU signals. As a case study, we target the detection of heel versus forefoot strikes to enable runners to self-correct harmful gait patterns through visual and auditory feedback during running. We propose four compact DL architectures (1D-CNN, 1D-SepCNN, LSTM, and Transformer) and optimize them for energy-efficient inference on two representative embedded Field-Programmable Gate Arrays (FPGAs): the AMD Spartan-7 XC7S15 and the Lattice iCE40UP5K.
Using our custom-built hardware prototype, we collect a labeled dataset from outdoor running sessions and evaluate all models via a fully automated deployment pipeline. Our results reveal clear trade-offs between model complexity and hardware efficiency. Evaluated across 12 participants, 6-bit quantized 1D-SepCNN achieves the highest average F1 score of 0.847 while consuming just 0.350 $\mu$J per inference with a latency of 0.140 ms on the iCE40UP5K running at 20 MHz. This configuration supports up to 13.6 days of continuous inference on a 320 mAh battery.

If you use the collected data or code, please consider citing our work:
```bibtex
@inproceedings{ling2025strikewatch,
  title     = {StrikeWatch: Wrist-worn Gait Recognition with Compact Time-series Models on Low-power FPGAs},
  author    = {Ling, Tianheng and Qian, Chao and Zdankin, Peter and Weis, Torben and Schiele, Gregor},
  booktitle = {Proceedings of the IEEE Annual Congress on Artificial Intelligence of Things (IEEE AIoT)},
  year      = {2025},
  location  = {Osaka, Japan},
  note      = {To appear},
  url       = {https://arxiv.org/abs/YYYY}
}
```
---

#### Dataset

The **StrikeWatch dataset** was collected for running gait classification (forefoot vs. heel strike) using 3-axis IMU data from a custom wrist-worn device. Data was recorded from **12 participants** during real-world outdoor running sessions. In addition to raw accelerometer readings, the magnitude signal (computed as \( a = \sqrt{a_x^2 + a_y^2 + a_z^2} \)) is also included. The dataset is released under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/). You may use, share, and adapt the data for **academic and non-commercial purposes**, with proper attribution. Commercial use is not permitted. More details are available in ```data``` folder.

---

#### Getting Started
```
# Clone and enter repo
git clone https://github.com/tianheng-ling/StrikeWatch
cd StrikeWatch

# Set up virtual environment (Python 3.11)
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# Install requirements
pip install -r requirements.txt
```
---

#### Usage
All runnable scripts are organized in the **`scripts/`** folder for convenience: You can **run scripts directly** from their folders.  
For example:

```bash
# Find optimal configuration of certain model on certain participant on AMD FPGA
bash scripts/exp1/quant_train.sh

# Find optimal configuration of certain model on certain participant on Lattice FPGA
bash scripts/exp2/quant_train.sh

# Verfiy the generalibility of chosen configuration of CNN model on all participants
bash scripts/exp3/quant_train_cnn.sh

# if you don't want to execute optuna search, you can 
python main.py
```
---

### Supported FPGA Platforms
This framework targets ultra-small, low-power FPGAs, making it ideal for on-device edge AI deployment:

| FPGA Platform	Model  | Used in Papers | Frequency | Resource Budges               |
| -------------------- | -------------- | --------- | ----------------------------- |
| AMD Spartan-7 Series | XC7S15         | 100 MHz   | 8,000 LUTs, 10 BRAMs, 20 DSPs |
| Lattice iCE40 Series | UP5K           | 16 MHz    | 5,280 LUTs, 30 EBRs, 8 DSPs   |

Deployment scripts and bitstreams for both platforms are included. 
> **Radiant Path Configuration:** If you plan to deploy models to the Lattice iCE40 platform, please **make sure to update the toolchain paths** used in the automation scripts: In `optuna_utils/radiant_runner.py`, replace the hardcoded path `/home/tianhengling/lscc/radiant/2023.2/bin/lin64` with the path to your own Radiant installation directory. You can typically find your Radiant install path by checking the environment variable `RADIANT_PATH` or by locating the executable manually.

---

#### Related Repositories
This project is part of a broader family of FPGA-optimized time-series models. You may also be interested in:

- **OnDeviceSoftSensorMLPs** → [GitHub Repository](https://github.com/tianheng-ling/OnDeviceSoftSensorMLP)  
- **OnDeviceLSTM** → [GitHub Repository](https://github.com/tianheng-ling/EdgeOverflowForecast)
- **OnDeviceTransformer** → [GitHub Repository](https://github.com/tianheng-ling/TinyTransformer4TS)
- **OnDevice1D-(Sep)CNN** → coming soon

---

#### Acknowledgement
This work is supported by the German Federal Ministry for Economic Affairs and Climate Action under the RIWWER project (01MD22007C). 

---

#### Contact
This repository is maintained by researchers from the Intelligent Embedded Systems Chair and Distributed Systems Chair at University of Duisburg-Essen, Germany.

For questions or feedback, please feel free to open an issue or contact us at tianheng.ling@uni-due.de. If you are interested in our customized hardware, contact us at chao.qian@uni-due.de or peter.zdankin@uni-due.de.