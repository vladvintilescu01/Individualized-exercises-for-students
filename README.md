# Individualized exercises for students
This repository contains the software architecture for Individualized exercises for students(based on FLAN-T5, FLAN-T5-LARGE, BART)


---

## Prerequsits

Follow these steps if you want to train inside the Anaconda ecosystem. Also, if you want to see the demo of this app you will need to do these steps. You can use any type of environment to train these models, the steps will look almost the same.

### 1. Install Anaconda

Download and install Anaconda from the official page:  
<https://www.anaconda.com/products/distribution>

### 2. Create a New Environment

Open **Anaconda Prompt** (or your terminal) and create an isolated environment:

```bash
conda create -n DL-NLM python=3.8
conda activate DL-NLM
```

### 3. Install Required Libraries

With the environment active, install all necessary packages:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install pandas
conda install -c conda-forge transformers datasets
```

### 4. Install & Launch Spyder

If Spyder IDE is not already present:

```bash
conda install spyder
```

Then start it:

```bash
spyder
```
### 5. Get the dataset

The dataset is available via this link:https://mega.nz/file/C0JilJ4C#AZ7hgaole-2vFDox_6Ean_mdV2zpV_S112lIG-rjlc. After the download of this dataset you will need to put it in 'dataset' folder. This dataset is created by me and it is created in collaboration with a teacher for a better result.

## How to train a model (mandatory, before demo app)

**Is mandatory**, because the weights created are not available on Git, there are too large in order to be on Git.

1. In **Spyder**, select one of the models (BART is recommened, it is the best).  
2. Hit **Run** in Spyder to start training.  
3. Monitor the console for metrics, losses, and any early‑stopping callbacks.

---

## How to use the demo app

1. In **Spyder**, select the predict app (predict/predict-BART.py).  
2. Hit **Run** in Spyder to start the demo app.  
3. Right now the app can generate in console personalized exercises also you can modify inside the code, the next variable test_example which contain the profile of student and the exercise that you want to modify.