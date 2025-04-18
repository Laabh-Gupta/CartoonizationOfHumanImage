# CartoonizationOfHumanImage

## 📌 Project Overview

This project aims to cartoonize human images using deep learning techniques. Traditional methods such as bilateral filtering and edge detection are limited in preserving texture and structural details. Our solution leverages **Generative Adversarial Networks (GANs)**—specifically **CycleGAN** and **Pix2Pix**—to perform high-quality, stylized transformations from real images to cartoon versions.

---

## 🚀 Technologies Used

- **Python 3.x**
- **TensorFlow / PyTorch**
- **OpenCV**
- **Jupyter Notebook via Anaconda**
- **NVIDIA GeForce RTX 3050 GPU**
- **Matplotlib, NumPy, PIL, etc.**

---

## 🔧 System Setup

### Hardware Configuration
| Component | Specification |
|----------|---------------|
| **Platform** | Jupyter Notebook (Anaconda) |
| **GPU** | NVIDIA GeForce RTX 3050 |
| **CPU** | Intel i5-11350H |
| **RAM** | 16 GB |
| **Storage** | Local Drive |

### Software Requirements
Install required libraries using:

```bash
pip install -r requirements.txt
```

---

## 📂 Dataset Structure

The dataset is stored in the following directory:

```
cartoonization_project/pytorch-CycleGAN-and-pix2pix/datasets/photo2/
```

It contains four folders:

| Folder  | Description                              |
|---------|------------------------------------------|
| `trainA` | Human images used for training           |
| `trainB` | Cartoonized images corresponding to `trainA` |
| `testA`  | Human images used for testing            |
| `testB`  | Cartoonized images corresponding to `testA` |

- Folder **A** contains real human images.
- Folder **B** contains the corresponding cartoonized images.
- This structure is compatible with both **CycleGAN** and **Pix2Pix** models.

---

## 🖼️ Visual Examples

Below are sample outputs generated from the models:

| **Input (Human Image)** | **CycleGAN Output** | **Pix2Pix Output** |
|-------------------------|---------------------|--------------------|
| ![input](assets/sample_input.jpg) | ![cyclegan](assets/sample_cyclegan.jpg) | ![pix2pix](assets/sample_pix2pix.jpg) |

> 💡 Place your sample images in an `assets/` folder inside your project directory to render them correctly in the README.

---