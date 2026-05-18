# Real vs. Fake Face Detection Project

This repository contains the code for a deep learning project that distinguishes between real and AI generated faces. It uses a multi-stream neural network architecture that combines RGB, FFT (Fast Fourier Transform), and SRM (Spatial Rich Model) data to achieve high-accuracy classification.

The project includes:
- A Jupyter Notebook with the complete pipeline: data preprocessing, model definition, training, evaluation, and visualization.
- Grad-CAM visualizations to interpret and compare the model's decision-making process against a standard ResNet-18 model.
- A modular and well-documented codebase that is easy to follow and adapt.

## How to Run This Project

### 1. Clone the Repository
First, clone this repository to your local machine:
```bash
git clone https://github.com/aashray325/DeepFake-Detection-with-Grad-CAM.git
cd DeepFake-Detection-with-Grad-CAM
```

### 2. Download the Dataset
This project uses the "Real and Fake Face Detection" dataset, which is available on Kaggle.

- **Download the data from:** [https://www.kaggle.com/datasets/aashraygupta325/subset-50k](https://www.kaggle.com/datasets/aashraygupta325/subset-50k)
- **Note-** The dataset is actually a subset version of the original _140k Real and Fake Faces_ from kaggle (https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) for running on my machine. 
- After downloading, make sure your project directory is structured correctly. The notebook expects the data to be in a specific folder. Your final structure should look like this:

```
Project Directory/
├── subset_50k/
│   └── real_vs_fake/
│       ├── train/
│       ├── valid/
│       └── test/
├── project.ipynb
├── README.md
└── requirements.txt
```

### 3. Set Up the Environment
It is highly recommended to use a virtual environment to manage dependencies.

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  

# Install the required packages
pip install -r requirements.txt
```

### 4. Run the Notebook
Once the setup is complete, you can run the Jupyter Notebook:
```bash
jupyter notebook project.ipynb
```
Inside the notebook, you can execute the cells sequentially to train the models, view the results, and see the Grad-CAM visualizations.

## Project Structure
- `project.ipynb`: The main notebook containing all the code.
- `requirements.txt`: A list of all Python packages required to run the project.
- `README.md`: This file, providing an overview and setup instructions.
- `.gitignore`: Specifies which files and folders to exclude from version control.
