# DVB-S2X-Modulation-Classification

## Problem Statement: 
DVB-S2X waveforms is have various types of modulation scheme like QPSK, APSK etc. The selection of modulation scheme is based upon channel. A novel AMR based soft algorithm is need to develop to detect the DVB S2X modulation waveform on ground in order to synchronized the ground receiver from satellite. 

Challenge: 
1. To develop the algorithm compliant to existing DVB S2X waveform. 
2. To ensure the develop algorithm should not complex to implement and not taking much hardware resources. 
3. To integrate all DVB S2X waveform together and having a single detection algorithm. 

## Proposed Solution: 
Our workflow is as follows: 
1. Generate the various modulation signals with various levels of noise and number of symbols. 
2. Generate the symbols from the signals. 
3. Extract features such as magnitude, phase, as well as few cyclostationary features such as autocorrelation and spectral correlation density (SCD). 
4. Store the train and test datasets into separate csv files. 
5. Run a Random Forest Model and evaluate the model in terms of size and latency, as well as the model's accuracy. 


## Project Structure: 
```
├── data
│   ├── test.csv
│   └── train.csv
├── notebooks
│   ├── DVB-S2X Wave Generator.ipynb
│   └── rf_modelling_v2.ipynb
├── README.md
├── requirements.txt
├── DVBS2X.py
├── script.py
├── main.py
└── utils.py
```

* **data/**: contains the `train.csv` and `test.csv` file. 
* **notebooks/**: contains the initial dry runs before modularity. 
* **requirements.txt**: contains the dependencies to be installed. 
* **utils.py**: important functions used such as feature extraction, dataset creation, etc. 
* **script.py**: the draft script with very little modularity. 
* **main.py**: the final execution file. 

## To Run: 

### Step 1: Create a conda environment with the 
One can do so by running the command: 
```bash
conda create -n <env_name>
```

### Step 2: Install the dependencies. 
```bash
pip install -r requirements.txt
```

### Step 3: Running main.py 

The `main.py` file contains the script that creates the train and test data from the features specified, as well as the noise and the symbols which are passed as variables. Simply running: 
```bash
python main.py
```

You could play around with `notebooks/rf_modelling_v2.ipynb` wiht the various features or add anything additional for experimentation. 

## Licensing
This is not to be used publicly and is not commerically available. 

## Credits: Silicon Spatula, SIH 2024. 