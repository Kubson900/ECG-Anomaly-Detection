# ECG Diagnostic Multi-label Classification 🫀
![ECG Image](https://www.heart.org/-/media/Images/Health-Topics/Arrhythmia/ECG-normal.jpg)

# Setup 📰
* ## Install [Python](https://www.python.org/downloads/release/python-3109/)
* ## Create [venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment)
```
python -m venv /path/to/new/virtual/environment
```
```
.\env\Scripts\activate
```
* ## Install requirements inside activated venv
```
pip install -r /path/to/requirements.txt
```

## To make TSFEL work 🚩
* https://github.com/fraunhoferportugal/tsfel/issues/123
* https://github.com/scikit-multilearn/scikit-multilearn/issues/230

# Directories descriptions ❔
## [networks](networks)
* neural networks models architectures
## [patient_data_analysis_images](patient_data_analysis_images)
* distributions of patients features
## [saved_data](saved_data)
* data evaluation for each model
## [scripts](scripts)
* useful functions 

# Notebooks descriptions ❔
## [ecg_visualization.ipynb](ecg_visualization.ipynb)
* visualizations of each diagnostic for each lead for sample patients
## [main.ipynb](main.ipynb)
* process of training, saving and evaluating neural networks
## [patient_data_analysis.ipynb](patient_data_analysis.ipynb)
* visualizations of distributions of patients features
## [tsfel_adapted.ipynb](tsfel_adapted.ipynb)
* process of training, saving and evaluating classic machine learning models adapted to multi-label
## [tsfel_binary_relevance.ipynb](tsfel_binary_relevance.ipynb)
* process of training, saving and evaluating classic machine learning models with problem transformation
