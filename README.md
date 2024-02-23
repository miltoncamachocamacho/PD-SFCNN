# Exploiting macro- and micro-structural brain changes for improved Parkinson's disease classification from MRI data

Milton Camacho1;2, Matthias Wilms2;3;4;5, Hannes Almgren5;6, Kimberly Amador1;2, Richard Camicioli7, Zahinoor Ismail5;6;8;9, Oury Monchi5;6;10;11, Nils D. Forkert2;3;4;5;6

For the Alzheimer’s Disease Neuroimaging Initiative+

1 Biomedical Engineering Graduate Program, University of Calgary, Canada.
2 Department of Radiology, University of Calgary, Canada.
3 Alberta Children’s Hospital Research Institute, University of Calgary, Canada.
4 Departments of Pediatrics and Community Health Sciences, University of Calgary, Canada.
5 Hotchkiss Brain Institute, University of Calgary, Canada.
6 Department of Clinical Neurosciences, University of Calgary, Canada.
7 Neuroscience and Mental Health Institute and Department of Medicine (Neurology), University of Alberta, Edmonton, Alberta, Canada.
8 Department of Psychiatry, University of Calgary, Canada.
9 College of Medicine and Health, University of Exeter, Exeter, UK.
10 Department of Radiology, Radio-oncology and Nuclear Medicine, Université de Montréal, Quebec, Canada.
11 Centre de Recherche, Institut Universitaire de Gériatrie de Montréal, Québec, Canada.

## Abstract

Parkinson’s disease (PD) is the second most common neurodegenerative disease. Accurate PD diagnosis is crucial for effective treatment and prognosis but can be challenging, especially at early disease stages. This study aimed to develop and evaluate an explainable deep learning model for PD classification from multimodal neuroimaging data. The model was trained using one of the largest collections of T1-weighted and diffusion-tensor magnetic resonance imaging (MRI) datasets. A total of 1264 datasets from eight different studies were collected, including 611 PD patients and 653 healthy controls (HC). These datasets were pre-processed and non-linearly registered to the MNI PD25 atlas. Six imaging maps describing the macro- and micro-structural integrity of brain tissues complemented with age and sex parameters were used to train a convolutional neural network (CNN) to classify PD/HC subjects. Explainability of the model’s decision-making was achieved using SmoothGrad saliency maps, highlighting important brain regions. The CNN was trained using a 75%/10%/15% train/validation/test split stratified by diagnosis, sex, age, and study, achieving a ROC-AUC of 0.89, accuracy of 80.8%, specificity of 82.4%, and sensitivity of 79.1% on the test set. Saliency maps revealed that diffusion tensor imaging data, especially fractional anisotropy, was more important for the classification than T1-weighted data, highlighting subcortical regions such as the brainstem, thalamus, amygdala, hippocampus, and cortical areas. The proposed model, trained on a large multimodal MRI database, can classify PD patients and HC subjects with high accuracy and clinically reasonable explanations, suggesting that micro-structural brain changes play an essential role in the disease course.

Before you begin, ensure you have the following installed:
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (See the installation instructions below)

### Step 1: Install Miniconda

Miniconda is a minimal installer for Conda. It is a smaller alternative to Anaconda, suitable for environments where space is a premium. Follow the instructions below to install Miniconda:

1. Download Miniconda for your operating system from the [official Miniconda page](https://docs.conda.io/en/latest/miniconda.html).
2. Follow the installation instructions for your OS:
    - Windows: Run the .exe installer and follow the on-screen instructions.
    - macOS and Linux: Open a terminal, navigate to the directory containing the downloaded file, and run the following command:
        ```bash
        bash Miniconda3-latest-Linux-x86_64.sh # Adjust the filename as necessary
        ```
    - Follow the prompts on the installer screens.
    - If you are unsure about any setting, accept the defaults. You can change them later.

3. To make the changes take effect, close and reopen your terminal window.

### Step 2: Create the Conda Environment

1. To create the Conda environment using the environment.yml file, follow these steps:

    - Open a terminal.
    - Navigate to the directory containing the environment.yml file.
    - Run the following command:
        ```bash
        conda env create -f environment.yml
        ```

Wait for the process to complete. This might take a few minutes depending on the number of packages to be installed.

### Step 3: Run job_example.sh

1. To run job_example.sh, follow these steps:
    ```bash
    bash job_example.sh
    ```
