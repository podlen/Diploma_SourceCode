# Impact Location and Force Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)

This repository contains the complete source code and documentation for the undergraduate thesis project by **Enej Podlipnik** at the University of Ljubljana, Faculty of Mechanical Engineering.

The project focuses on developing a comprehensive system for predicting the location and force of impacts on a mechanical structure using data-driven approaches and deep learning. It integrates the design of a custom experimental setup, data acquisition with National Instruments hardware, advanced signal processing, and the training of a Convolutional Neural Network (CNN) to enable accurate, real-time impact analysis.

## Project Overview & Key Features

Modern Structural Health Monitoring (SHM) systems require automated and precise methods for detecting and characterizing impacts. This project addresses this challenge by developing a complete workflow that combines experimental work with state-of-the-art machine learning.

**Key Features:**

- **End-to-End System:** The project covers the entire development pipeline: from the 3D design and printing of a test plate and custom sensors to data acquisition, model training, and performance evaluation.
- **Custom Piezoresistive Sensors:** To measure the dynamic response, custom-designed sensors were fabricated using multi-material 3D printing, combining flexible, conductive TPU with rigid PLA for optimal mechanical vibration transfer.
- **Data-Driven Preprocessing:** An experimental modal analysis was performed to determine the system's natural frequencies. This informed the selection of optimal parameters for low-pass filtering (900 Hz cutoff) and down-sampling (factor of 60), which significantly improved model performance.
- **Multi-Task Convolutional Neural Network:** An `ImpactPredictor` model was developed in PyTorch to simultaneously predict both the location and force of an impact from the same input signals, framed as classification tasks.
- **Data Acquisition with LDAQ:** The open-source LDAQ library was used for data acquisition with National Instruments hardware, enabling easy integration and automation of the measurement process.
- **Diverse Sampling Strategies:** Data was collected using both a systematic grid and Latin Hypercube Sampling (LHC) to ensure a diverse and well-distributed training dataset, enhancing the model's ability to generalize.

## Repository Structure

- `functions.py`: A modular library containing all core functions used in the project. This includes the `ImpactPredictor` model definition, data handling classes, training and evaluation scripts, signal processing functions, and visualization tools.
- `notebooks/`: Jupyter notebooks demonstrating the entire workflow.
  - `measuring_script.ipynb`: For data acquisition using grid, LHC, and live inference modes.
  - `data_edit.ipynb`: For preprocessing raw sensor data, including filtering, down-sampling, and label remapping.
  - `model.ipynb`: For training, hyperparameter tuning, and evaluating the final CNN model.
  - `lastna_frekvenca.ipynb`: For performing experimental modal analysis to determine the plate's natural frequencies.
- `requirements.txt`: A list of Python dependencies.
- `LICENSE`: The MIT License for this project.

## Requirements

All necessary Python packages are listed in `requirements.txt`. Key dependencies include:

- torch
- torchmetrics
- torchinfo
- numpy
- scipy
- matplotlib
- LDAQ
- nidaqmx
- ipykernel
- jupyter

*Note: The `LDAQ` and `nidaqmx` libraries require National Instruments hardware and the corresponding NI-DAQmx drivers to be installed on your system.*

## Workflow

1.  **System Setup:** Fabricate the mechanical components (frame, plate, sensors) using the provided 3D models and set up the measurement chain.
2.  **System Dynamics Analysis:** Use the `lastna_frekvenca.ipynb` notebook to perform a modal analysis and determine the system's natural frequencies.
3.  **Data Acquisition:** Use the `measuring_script.ipynb` notebook to acquire dynamic response data from impacts on the plate.
4.  **Data Preprocessing:** Use the `data_edit.ipynb` notebook to filter, down-sample, and prepare the final dataset.
5.  **Model Training:** Use the `model.ipynb` notebook to train the `ImpactPredictor` model on the prepared data.
6.  **Evaluation and Application:** Evaluate the trained model's performance using the metrics and visualization tools also included in the `model.ipynb` notebook.

## How to Cite

If you use this project or its code in your academic work, please cite it as follows:

> Podlipnik, Enej. "Impact Location and Force Prediction." Undergraduate Diploma Thesis, University of Ljubljana, Faculty of Mechanical Engineering, 2025. https://github.com/yourusername/your-repo-name

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