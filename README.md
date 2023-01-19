# ECG Diagnostic Multi-label Classification 🫀
![ECG Image](https://www.heart.org/-/media/Images/Health-Topics/Arrhythmia/ECG-normal.jpg)

# Setup 📰

## Install [Python](https://www.python.org/downloads/release/python-3109/)
## Create [venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment)
```
python -m venv /path/to/new/virtual/environment
```
```
.\env\Scripts\activate
```
## Install requirements inside activated venv
```
pip install -r /path/to/requirements.txt
```

## Download dataset 💽
* [Description page](https://physionet.org/content/ptb-xl/1.0.3/)
* [Download the ZIP file](https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip) (1.7GB)
* Download the files using your terminal:
```
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/
```

## To make TSFEL work 🚩
  * [Problem with SciPy](https://github.com/fraunhoferportugal/tsfel/issues/123)
  * [Problem with MLkNN](https://github.com/scikit-multilearn/scikit-multilearn/issues/230)

# Directories descriptions ❔
## [fonts](fonts)
* Fonts used
## [models_comparison](models_comparison)
* images and .csv files summing up models performance
## [networks](networks)
* neural networks models architectures
## [patient_data_analysis_images](patient_data_analysis_images)
* distributions of patients features
## [saved_data](saved_data)
* data evaluation for each model
## [saved_images](saved_images)
* confusion matrix for each model
## [scripts](scripts)
* useful functions 

# Notebooks descriptions ❔
## [ecg_visualization.ipynb](ecg_visualization.ipynb)
* visualizations of each diagnostic for each lead for sample patients
## [main.ipynb](main.ipynb)
* process of training, saving and evaluating neural networks
## [patient_data_analysis.ipynb](patient_data_analysis.ipynb)
* visualizations of distributions of patients features
## [results_analysis.ipynb](results_analysis.ipynb)
* analysis of models performance
## [sample_patient_analysis.ipynb](sample_patient_analysis.ipynb)
* classification of sample patients
## [tsfel_adapted.ipynb](tsfel_adapted.ipynb)
* process of training, saving and evaluating classic machine learning models adapted to multi-label
## [tsfel_binary_relevance.ipynb](tsfel_binary_relevance.ipynb)
* process of training, saving and evaluating classic machine learning models with problem transformation
