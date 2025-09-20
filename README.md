# TKG-LLM

title: "TKG-LLM: Temporal Knowledge Graph as Enhanced Prompt Learning with LLM for Time Series Forecasting"



## 🛠 Prerequisites

Ensure you have installed the necessary dependencies by first building environment:

```
conda create -n "myenv" python=3.10.0
conda activate myenv
```
Inside the folder, run:
```
pip install -r requirements.txt
```

## 📊 Prepare Datasets

M4 and ETT datasets are conveniently available at [Autoformer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy).
Create a separate folder named `./data` 



## 💻 Training

All scripts are located in `./scripts`. Example:

```shell
cd Long-term_Forecasting 
sh scripts/etth1.sh
```



