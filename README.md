# Impact Location and Force Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)

This repository houses the complete source code and documentation for the undergraduate thesis project by **Enej Podlipnik** at the University of Ljubljana, Faculty of Mechanical Engineering.

The project details a comprehensive system for predicting the location and force of impacts on a mechanical structure using data-driven approaches and deep learning. It integrates the design of a custom experimental setup, data acquisition with National Instruments hardware, advanced signal processing, and the training of a Convolutional Neural Network (CNN) for accurate, real-time impact analysis.

## Project Overview

Modern Structural Health Monitoring (SHM) systems require automated and precise methods for detecting and characterizing impacts. This project addresses this challenge by developing a complete workflow that combines experimental work with state-of-the-art machine learning.

**Key Features:**

-   **End-to-End System:** Covers the entire development pipeline: from the 3D design and printing of a test plate and custom sensors to data acquisition, model training, and performance evaluation.
-   **Custom Piezoresistive Sensors:** To measure the dynamic response, custom-designed sensors were fabricated using multi-material 3D printing, combining flexible, conductive TPU with rigid PLA for optimal mechanical vibration transfer.
-   **Dynamic System Analysis & Preprocessing:** Experimental modal analysis was performed to determine the system's natural frequencies. This informed the selection of optimal parameters for low-pass filtering (900 Hz cutoff) and down-sampling (factor of 60), significantly improving model performance.
-   **Multi-Task Convolutional Neural Network:** A custom `ImpactPredictor` model was developed in PyTorch to simultaneously predict both the location and force of an impact from the same input signals, framed as classification tasks.
-   **Automated Data Acquisition:** The open-source LDAQ library was used for data acquisition with National Instruments hardware, enabling easy integration and automation of the measurement process.
-   **Diverse Sampling Strategies:** Data was collected using both a systematic grid and Latin Hypercube Sampling (LHC) to ensure a diverse and well-distributed training dataset, enhancing the model's ability to generalize.

## Repository Structure

-   `functions.py`: A modular library containing all core functions, including the `ImpactPredictor` model definition, data handling classes, training/evaluation scripts, signal processing functions, and visualization tools.
-   `notebooks/`: A collection of Jupyter notebooks demonstrating the entire workflow.
    -   `measuring_script.ipynb`: For data acquisition using grid, LHC, and live inference modes.
    -   `data_edit.ipynb`: For preprocessing raw sensor data, including filtering, down-sampling, and label remapping.
    -   `model.ipynb`: For training, hyperparameter tuning, and evaluating the final CNN model.
    -   `lastna_frekvenca.ipynb`: For performing experimental modal analysis to determine the plate's natural frequencies.
-   `requirements.txt`: A list of all Python dependencies.
-   `LICENSE`: The MIT License governing this project.

## Installation and Requirements

All necessary Python packages are listed in `requirements.txt`.

To install the dependencies, run the following command in your terminal:
```bash
pip install -r requirements.txt
```

## How to Cite

If you use this project or its code in your academic work, please cite it as follows:
```
Podlipnik, Enej. "Impact Location and Force Prediction." Undergraduate Diploma Thesis, University of Ljubljana, Faculty of Mechanical Engineering, 2025. https://github.com/podlen/Diploma_SourceCode.git
```

Or in BibTeX format:

```bibtex
@misc{podlipnik2025impact,
  author = {Enej Podlipnik},
  title = {Določanje lokacije in sile udarca na plošči z uporabo nevronskih mrež - Impact Location and Force Prediction},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/podlen/Diploma_SourceCode.git}}
}
```

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
