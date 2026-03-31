
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
conda env create -f environment.yml

# Activate the environment
conda activate muxgel
```

### 3\. Simulation Assets (Optional)

To generate your own datasets using the `mujoco_scanned_objects` library, fetch the 3D assets (\~2 GB):

```bash
git submodule update --init external/mujoco_scanned_objects
```

-----

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































## 📊 Dataset Preparation

### For Simulation Training

To prepare the simulation data, you can either generate it manually or download our pre-processed version.

#### Option A: Manual Generation

Run the following scripts in order to generate, clean, and resize the data:

1.  **Generate Data**: `python scripts/datasetGeneration/mujoco_imageGenerate.py`
2.  **Clean Folders**: `python scripts/datasetGeneration/mujoco_folder_clean.py` (Removes empty or insufficient data folders)
3.  **Resize Data**: `python scripts/datasetGeneration/dataResize.py` (Downsamples to 320x240 for optimized training)

*The final dataset will be located in `data/mujoco_patch_output_320_240`.*

#### Option B: Direct Download

Download the following from [Hugging Face](https://huggingface.co/datasets/zhixianhu/muxgel) and extract them into the `data/` folder:

  * `mujoco_patch_output_320_240.tar.xz`


  * `indoorCVPRBlur_320_240.tar.xz`
      * *Note: The indoor scene dataset is processed with a disk defocus blur based on Quattoni & Torralba (CVPR 2009).*

-----

### \#\#\# For Real-World Training

To train with real-world data, download the calibration assets from [Hugging Face](https://huggingface.co/datasets/zhixianhu/muxgel):

  * Download `calibration_data.zip` and unzip it into the `data/` folder.

-----

## 🚀 Training & Testing

### Training Scripts

Training scripts are located in `scripts/train/`. We use the following acronyms for configurations:

| Acronym | Meaning | Acronym | Meaning |
| :--- | :--- | :--- | :--- |
| **si** | Single-Input | **abst** | Absolute Tactile |
| **di** | Dual-Input | **rest** | Residual Tactile |

**Example run (Dual-Input Residual model):**

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
  * **[TacEx](https://github.com/DH-Ng/TacEx)**: Source of calibration files.
  * **[MuJoCo Scanned Objects](https://github.com/kevinzakka/mujoco_scanned_objects)**: Simulation 3D model assets.
  * **Indoor CVPR Dataset**: Originally from *Quattoni & Torralba, "Recognizing indoor scenes," CVPR 2009*.

-----

## 📜 Citation

If you find this work helpful, please cite our paper:

```bibtex
@article{hu2026muxgel,
  title={MuxGel: Simultaneous Dual-Modal Visuo-Tactile Sensing via Spatially Multiplexing and Deep Reconstruction},
  author={Hu, Zhixian and [Other Authors]},
  year={2026},
  journal={Selected Journal/Conference Name}
}
```

-----

### What I changed:

1.  **Corrected Hierarchy**: Grouped "Option A" and "Option B" clearly under the "For Simulation Training" header.
2.  **Explicit File Lists**: Listed the `.tar.xz` and `.zip` files clearly so users know exactly what to look for on Hugging Face.
3.  **Refined Language**: Used "Manual Generation" and "Direct Download" as clearer sub-headings.
4.  **Consistency**: Ensured all paths and script names match your provided source.

Does this version look ready for your repo?





### For Simulation Training
Option A: Manual Generation (Simulation)

Run the following scripts in order to generate, clean, and resize the data:

1.  **Generate Data**: `python scripts/datasetGeneration/mujoco_imageGenerate.py`
2.  **Clean Data**: `python scripts/datasetGeneration/mujoco_folder_clean.py` (Removes empty or insufficient folders)
3.  **Resize Data**: `python scripts/datasetGeneration/dataResize.py` (Downsamples to 320x240 for efficiency)

The final dataset will be located in `data/mujoco_patch_output_320_240`.

Option B: Direct Download

Download the following from [Hugging Face](https://huggingface.co/datasets/zhixianhu/muxgel) and extract them into the `data/` folder:

Download `mujoco_patch_output_320_240.tar.xz` and `indoorCVPRBlur_320_240.tar.xz`.
      * *Note: The indoor scene dataset is processed with a disk defocus blur based on Quattoni & Torralba (CVPR 2009).

### For Simulation Training
For Real-World Training**: Download `calibration_data.zip`.

-----

## 🚀 Training & Testing

### Training Scripts

Training scripts are located under `scripts/train/`. We use the following naming conventions for model configurations:

| Prefix | Meaning | Prefix | Meaning |
| :--- | :--- | :--- | :--- |
| **si** | Single-Input | **abst** | Absolute Tactile |
| **di** | Dual-Input | **rest** | Residual Tactile |

**Example: Training a Dual-Input Residual model**

```bash
python scripts/train/train_real_di_rest.py --wandb
```

### Real-time Visualization

To run the real-time inference and visualization test:

```bash
python scripts/test/realtime_vis.py
```

-----

## 🙏 Acknowledgments

This project builds upon and adapts several excellent open-source repositories:

  * **[GelSight Mini SDK](https://github.com/duyipai/gsmini)**: Basic driver and robotics interface integration.
  * **[Taxim](https://github.com/Robo-Touch/Taxim)**: Our tactile simulation is built upon Taxim for example-based rendering.
  * **[TacEx](https://github.com/DH-Ng/TacEx)**: Calibration files were obtained from the TacEx project.
  * **[MuJoCo Scanned Objects](https://github.com/kevinzakka/mujoco_scanned_objects)**: 3D model assets for simulation.
  * **[Indoor CVPR Dataset](https://web.mit.edu/torralba/www/indoor.html)**: Used for visual background data with defocus blur.

-----

## 📜 Citation

If you find this work useful for your research, please cite:

```bibtex
@article{hu2026muxgel,
  title={MuxGel: Simultaneous Dual-Modal Visuo-Tactile Sensing via Spatially Multiplexing and Deep Reconstruction},
  author={Hu, Zhixian and [Other Authors]},
  year={2026},
  journal={Selected Journal/Conference Name}
}
```

-----

### Key Improvements in this version:

1.  **Professional Terminology**: Replaced phrases like "clean folder that is not enough data" with "Removes empty or insufficient folders."
2.  **Clearer Workflow**: Grouped the multi-step dataset generation process into a numbered list.
3.  **Table for Acronyms**: Made the `si`, `di`, `rest` meanings much easier to read at a glance.
4.  **Markdown Best Practices**: Used standard header levels, bold text for emphasis, and organized the Acknowledgment links.

How does this look to you? Feel free to ask if you'd like any specific section expanded\!

Coming soon...

### 1. Clone the repository
git clone https://github.com/zhixian/MuxGel.git
cd MuxGel

## Installation

Bash
conda env create -f environment.yml
conda activate muxgel


### 3. (Optional) Simulation Assets
If you want to generate dataset yourself using mujoco_scanned_objects dataset.
fetch the assets using:

```bash
git submodule update --init external/mujoco_scanned_objects
Note: This will download 2 GBs of 3D models.


Acknowledge
We integrate Taxim for example-based tactile rendering. While calibs files we obtain from TacEx calib files.

