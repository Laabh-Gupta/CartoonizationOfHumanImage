# Cartoonization Project using CycleGAN and Pix2Pix

This project focuses on the cartoonization of human images using CycleGAN and Pix2Pix architectures. It utilizes a structured dataset and visualizes training progress with **Visdom**.

---

## 📁 Dataset Structure

The dataset is located at:

```
cartoonization_project/pytorch-CycleGAN-and-pix2pix/datasets/photo2/
```

It contains four folders:

| Folder | Description |
|--------|-------------|
| `trainA` | Human images used for training |
| `trainB` | Cartoonized images corresponding to trainA |
| `testA`  | Human images used for testing |
| `testB`  | Cartoonized images corresponding to testA |

> Folder A contains real human images, and Folder B contains the corresponding cartoonized versions. This structure is compatible with both CycleGAN and Pix2Pix.

---

## 📊 Visualizing Training with Visdom

**Visdom** is used to monitor training metrics such as loss curves and sample outputs.

### 🔧 Installation

Install Visdom using pip:

```
pip install visdom
```

### 🚀 Starting the Visdom Server

Launch the Visdom server in a separate terminal:

```
visdom
```

This will start a local server at:

```
http://localhost:8097
```

Keep this terminal open during training.

### ⚙️ Enabling Visdom in Training

Add the following flags to your training command:

```
--display_id 1 --display_port 8097 --display_server http://localhost
```

Example command:

```
python train.py --dataroot ./datasets/photo2 --name photo2cartoon --model cycle_gan --display_id 1
```

---

## 🧪 Repository and Modifications

This project is based on the official CycleGAN and Pix2Pix implementation:

```
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
```

### 🚀 Setup Instructions

Clone the repository:

```
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
cd pytorch-CycleGAN-and-pix2pix
```

### ⚙️ Enhancements

Several modifications have been made to improve cartoonization quality:

- Fine-tuning models for better stylistic transformation.
- Improved training pipeline and visualizations.
- Optimized hyperparameters and preprocessing steps.

---

You can now experiment with CycleGAN and Pix2Pix for high-quality cartoonization of human images. Feel free to tweak and extend the setup for your own creative applications!
"""

