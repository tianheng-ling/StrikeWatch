## StrikeWatch Dataset

The **StrikeWatch dataset** was collected for running gait classification (forefoot vs. heel strike) using 3-axis IMU data from a custom wrist-worn device. Data was recorded from **twelve participants** during real-world outdoor running sessions. In addition to raw accelerometer readings, the magnitude signal (computed as \( a = \sqrt{a_x^2 + a_y^2 + a_z^2} \)) is also included.


##### Citation
For details on data collection, preprocessing, and model evaluation, please refer to the following paper:
```bibtex
@inproceedings{ling2025strikewatch,
  author={Ling, Tianheng and Qian, Chao and Zdankin, Peter and Weis, Torben and Schiele, Gregor},
  booktitle={2025 IEEE Annual Congress on Artificial Intelligence of Things (AIoT)}, 
  title={StrikeWatch: Wrist-worn Gait Recognition with Compact Time-series Models on Low-power FPGAs}, 
  year={2025},
  pages={66-74},
  keywords={Visualization;Transformers;Real-time systems;Hardware;Internet of Things;Wearable devices;Artificial intelligence;Gait recognition;Field programmable gate arrays;Software development management;Wrist-Worn Wearables;Running Gait Recognition;Time-Series Models;Model Quantization;On-Device Inference;Low-Power FPGA},
  doi={10.1109/AIoT66900.2025.00021}}
```

##### License
The dataset is released under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/). You may use, share, and adapt the data for **academic and non-commercial purposes**, with proper attribution. Commercial use is not permitted.