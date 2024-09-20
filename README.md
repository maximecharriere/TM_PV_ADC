# Anomaly Detection and Classification for Photovoltaic Systems
> Master's Thesis by **Maxime Charriere**  
> Supervised by Prof. **Dr. Christof Bucher** and **Prof. Dr. Horst Heck**  

## Useful resources 

- **Poster**: The Poster can be downloaded here -> [Master's Thesis Poster - Maxime Charriere - 2024.pdf](Report/Master's%20Thesis%20Poster%20-%20Maxime%20Charriere%20-%202024.pdf)
- **Report**: The Report can be downloaded here -> [Master's Thesis Report - Maxime Charriere - 2024.pdf](Report/Master's%20Thesis%20Report%20-%20Maxime%20Charriere%20-%202024.pdf)  
- **Code**: The Code is available on this GitHub repository -> [maximecharriere/TM_PV_ADC](https://github.com/maximecharriere/TM_PV_ADC)

## Abstract

The primary objective of this thesis is to enhance the monitoring of residential photovoltaic (PV) systems by developing a novel methodology for anomaly detection and classification. This approach is designed specifically for companies managing small-scale PV systems, typically ranging from 3 to 100 kWp, which often lack advanced monitoring sensors and professional management, such as utility-scale farms. Our method requires only the energy production measurements from the monitored system and nearby PV systems, making it a cost-effective and universal solution that can be implemented across a wide range of PV installations without significant additional investment.  

As Switzerland aims to increase its solar energy production from 4 TWh today to 34 TWh by 2050 to meet its energy transition goals, there is a pressing need for effective monitoring solutions for PV systems. The planned expansion will involve a substantial increase in the number of small-scale, decentralized PV systems, which will require consistent monitoring and maintenance. Given the diversity of equipment, installer companies, and collected data, a universal anomaly detection system is critical. Such a system will help minimize Mean Down Time (MDT) and Mean Time To Repair (MTTR), ensuring rapid and efficient identification of potential issues to maximize solar production, reduce economic losses, and helping Switzerland to achieve its objectives.  

The methodology developed in this thesis utilizes a combination of data analysis techniques, machine learning, and statistical methods. By employing the Half Sibling Regressor principle alongside a physics-based normalizer, the method accurately estimates the expected daily production of a PV system based on the energy production data of its neighbouring systems. The system then monitors for deviations between expected and actual production, effectively detecting anomalies when significant underperformance is identified. This capability is crucial for minimizing the Mean Down Time and Mean Time To Repair of PV systems, thereby enhancing operational efficiency and reliability.  

The performance of this novel approach was evaluated using real-world data from 326 PV systems, each with an average of 400 days of historical data. The results demonstrated an average Mean Absolute Percentage Error (MAPE) of 4.38% in estimating the expected production, with a standard deviation of 4.60% between the performance of each tested system. The anomaly detection algorithm successfully identified 97.4% of simulated anomalies. Moreover, the model's ability to handle missing data ensures continuous monitoring and anomaly detection, even when some neighbouring systems fail to provide data due to technical issues. Our study's MAPE of 4.38% is in line with similar studies: SolarClique reported a MAPE of 7.81% in a study involving 88 systems in Austin, Texas, while SunDown achieved a MAPE of 2.98% by comparing each module performance within a single PV system.  

Despite these promising results, several areas for further improvement have been identified. Future work should focus on incorporating higher-frequency data, such as hourly or minute-level measurements, to enhance the precision of anomaly detection and allow for more frequent monitoring updates. Additionally, developing an advanced anomaly classification system would provide more detailed insights to the company into the nature of the detected issues, enabling quicker and more targeted maintenance actions. Real-world field testing and iterative refinement based on industry feedback will also be essential to optimize the model further and develop more sophisticated detection and classification rules tailored to the specific needs of PV system operators.

## Setup 

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

