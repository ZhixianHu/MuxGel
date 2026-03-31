
# MuxGel: Simultaneous Dual-Modal Visuo-Tactile Sensing via Spatially Multiplexing and Deep Reconstruction

[website](https://zhixianhu.github.io/muxgel/)   
[dataset](https://huggingface.co/datasets/zhixianhu/muxgel)

MuxGel is a novel plug-and-play visuo-tactile gel-pad design that spatially multiplexes visual and tactile sensing, enabling the simultaneous recovery of vision and tactile signals for robotic manipulation.



## 🛠 Installation

### 1\. Clone the Repository

Clone the repo along with its submodules:

```bash
git clone --recursive https://github.com/zhixian/MuxGel.git
cd MuxGel
```

### 2\. Environment Setup

We recommend using miniforge for dependency management:

```bash
# Create the environment from the provided file
conda env create -f conda_environment.yaml

# Activate the environment
conda activate muxgel
```

### 3\. Simulation Assets (Optional)

To generate your own datasets using the `mujoco_scanned_objects` library, fetch the 3D assets (\~2 GB):

```bash
git submodule update --init external/mujoco_scanned_objects
```

## 📊 Dataset Preparation

### For Simulation Training

To train in simulation, you must first set up the **Scene Backgrounds**, then obtain the **Object Patches** via one of two methods.

#### 1\. Scene Backgrounds (Required)

For scene backgrounds, you will need to download the following from [Hugging Face](https://huggingface.co/datasets/zhixianhu/muxgel) and extract them into the `data/` folder:

  * **File**: `indoorCVPRBlur_320_240.tar.xz`
  * *Note: This dataset was processed using a disk defocus blur, originally from A. Quattoni and A. Torralba, “Recognizing indoor scenes,” CVPR 2009.*

#### 2\. Object Patches (Choose Option A or B)

You can either generate these patches manually or download our pre-processed version.

  * **Option A: Manual Generation**

    Run the following scripts in order:

    1.  **Generate Data**: `python scripts/datasetGeneration/mujoco_imageGenerate.py`
    2.  **Clean Folders**: `python scripts/datasetGeneration/mujoco_folder_clean.py`
    3.  **Resize Data**: `python scripts/datasetGeneration/dataResize.py` (Downsamples to 320x240)
        *The results will be stored in `data/mujoco_patch_output_320_240`.*

  * **Option B: Direct Download**

    Download and extract the following into the `data/` folder:

      * **File**: `mujoco_patch_output_320_240.tar.xz` from [Hugging Face](https://huggingface.co/datasets/zhixianhu/muxgel).

-----

### For Real-World Training

To train with real-world data, you will need the calibration assets:

  * Download `calibration_data.zip` from [Hugging Face](https://huggingface.co/datasets/zhixianhu/muxgel) and unzip it into the `data/` folder.

-----


## ⚖️ Pre-trained Weights

The trained model weights for all six architectures (SI, DI-AbsT, DI-ResT for both Simulation and Real-world) are hosted on **Hugging Face**.

| Repository | Link |
| :--- | :--- |
| **MuxGel Weights** | [huggingface.co/datasets/zhixianhu/muxgel](https://www.google.com/search?q=https://huggingface.co/datasets/zhixianhu/muxgel) |

### 📥 Automatic Download (Recommended)

We provide a helper script to fetch the necessary checkpoints (approx. 636 MB total) directly into your project root.

1.  **Install requirements:**

    ```bash
    pip install huggingface_hub
    ```

2.  **Run the download script:**

    ```bash
    python scripts/download_weights.py
    ```

## 🚀 Training & Testing

### Training Scripts

Training scripts are located in `scripts/train/`. We use the following acronyms for configurations:

| Acronym | Meaning | Acronym | Meaning |
| :--- | :--- | :--- | :--- |
| **si** | Single-Input | **abst** | Absolute Tactile |
| **di** | Dual-Input | **rest** | Residual Tactile |

**Example run (Dual-Input Residual-Tactile model):**

```bash
python scripts/train/train_real_di_rest.py --wandb
```

### Real-time Test

To run the real-time inference and visualization:

```bash
python scripts/test/realtime_vis.py
```

-----

## 🙏 Acknowledgments

This project adapts and modifies several excellent open-source repositories:

  * **[GelSight Mini](https://github.com/duyipai/gsmini)** & **[GS Robotics](https://github.com/gelsightinc/gsrobotics)**: Basis for sensor drivers and integration.
  * **[Taxim](https://github.com/Robo-Touch/Taxim)**: Our tactile simulation is built upon Taxim for example-based rendering.
  * **[TacEx](https://github.com/DH-Ng/TacEx)**: Source of calibration files for tactile simulation.
  * **[MuJoCo Scanned Objects](https://github.com/kevinzakka/mujoco_scanned_objects)**: Simulation 3D model assets, originally from the [Google Scanned Objects collection](https://app.gazebosim.org/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research).
  * **[Indoor CVPR Dataset](https://web.mit.edu/torralba/www/indoor.html)**: Visual backgrounds, based on *Quattoni & Torralba, "Recognizing indoor scenes," CVPR 2009*.
-----

## 📜 Citation

If you find this work helpful, welcome to cite our paper:

```bibtex
@article{hu2026muxgel,
  title={MuxGel: Simultaneous Dual-Modal Visuo-Tactile Sensing via Spatially Multiplexing and Deep Reconstruction},
  author={Hu, Zhixian and Xu, Zhengtong and Athar, Sheeraz and Wachs, Juan and She, Yu},
  journal={arXiv preprint arXiv:2603.09761},
  year={2026}
}
```

