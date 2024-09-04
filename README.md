# TM_PV_ADC

## Related Thesis
You can download the report of this Master Thesis -> [Master's Thesis - Maxime Charriere - 2024.pdf](./Report/Master's Thesis - Maxime Charriere - 2024.pdf).

## Code

The code is available on the GitHub repository -> [maximecharriere/TM_PV_ADC](https://github.com/maximecharriere/TM_PV_ADC).

### Setup 

1. Clone the repo
1. Install Python
1. Setup and install the virtual environment in a Powershell with: 
    ```python
    py -m venv .venv
    .\.venv\Scripts\Activate.ps1
    py -m pip install --upgrade pip
    py -m pip install -r requirements.txt
    ```
1. Add a `.env` file in the root directory with the variable `PARENT_DATA_DIR="path/to/database/directory"` where `path/to/database/directory` is the path to the directory where the database is stored (with the json of the metadats, and all csv with data of each system).

### Usage

> [!WARNING]  
> TODO: Add usage instructions

