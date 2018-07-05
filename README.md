# cotton-prediction (Data Explore)

A command line tool to explore the cotton data.

### Tech

I am using following platform and packages:

* [Python 3] - Main programing language
* [Jupyter Notebook] - Good presentation UI
* [Pandas, Numpy ...] - Deal with data easily
* [Matplotlib, Seaborn] - Good data viz packages



### Prerequisites
Please put one datasets - Data_28_F29.txt


### Installation

This project requires Conda or PIP to install.

Install all dependencies.

Use below command to generate the charts.
You can find type_of_chart in the next sections.
```sh
$ python data_explore.py file_name type_of_chart
$ python data_explore.py Data_28_F29.txt yield-stripplot
```

### Metadata

You can find type_of_chart in the next sections here.

| Type Of Chart | Description |
| ------ | ------ |
| ndvi-boxplot | create boxplot based on ndvi |
| ndvi-time | create timeseries based on ndvi |
| band-bar | create band bar based on band |
| yield-scatter | create yield scatter based on yield |
| yield-stripplot | create yield stripplot based on yield |
| yield-time | create yield timeseries based on yield |
| heatmap-corr  | create heatmap correlation based on all useful features |


### measures for performance

#### dimension: Band Data
| Model | mae | mse | rmse | 
| ------ | ------ | ------ | ------ | 
| Decision Tree | 160.59 | 39497.38 | 198.739478 | 
| SVM | 152.13 | 35730.77 | 189.025845 | 
| Random Forest | 163.33 | 39257.95 | 198.136191 |
| XGBoost | 157.42 | 37623.34 | 193.967368 |
| Linear Regression | 163.21 | 39662.45 | 199.154337 |

#### dimension: Soil Data
| Model | mae | mse | rmse | 
| ------ | ------ | ------ | ------ | 
| Decision Tree | 154.96 | 37585.53 | 193.869879 | 
| SVM | 131.15 | 26678.18 | 163.334565 | 
| Random Forest | 151.37 | 34650.81 | 186.147280 |
| XGBoost | 147.10 | 33261.91 | 182.378480 |
| Linear Regression | 201.11 | 64582.77 | 254.131403 |

#### dimension: ndvi Data
| Model | mae | mse | rmse | 
| ------ | ------ | ------ | ------ | 
| Decision Tree | 124.34 | 23983.26 | 154.865296 | 
| SVM | 111.04 | 19822.83 | 140.793572 | 
| Random Forest | 113.57 | 20401.07 | 142.832314 |
| XGBoost | 113.02 | 20522.64 | 143.257251 |
| Linear Regression | 123.85 | 24221.64 | 155.633030 |

#### dimension: Combined Data
| Model | mae | mse | rmse | 
| ------ | ------ | ------ | ------ | 
| Decision Tree | 123.24 | 24416.46 | 156.257672 | 
| SVM | 131.90 | 27923.23 | 167.102454 | 
| Random Forest | 104.54 | 17266.72 | 131.402892 |
| XGBoost | 120.21 | 22645.94 | 150.485680 |
| Linear Regression | 123.85 | 24221.64 | 155.633030 |


















## old - ingore/achive
| Model | accuracy | f1_weighted | precision_weighted | recall_weighted |
| ------ | ------ | ------ | ------ | ------ |
| Decision_Tree_00 | 0.85741 | 0.85348 | 0.87030 | 0.85741 | 
| Decision_Tree_01 | 0.82783 | 0.80221 | 0.80247 | 0.82783 |
| Decision_Tree_02 | 0.86165 | 0.86258 | 0.88478 | 0.86165 |
| Decision_Tree_03 | 0.86771 | 0.86284 | 0.88183 | 0.86771 |
| Support_Vector_Machines_00 | 0.84194 | 0.83528 | 0.88119 | 0.84194 | 
| Support_Vector_Machines_01 | 0.85976 | 0.85363 | 0.87554 | 0.85976 | 
| Support_Vector_Machines_02 | 0.96691 | 0.96652 | 0.97015 | 0.96691 | 
| Support_Vector_Machines_03 | 0.86130 | 0.85504 | 0.87531 | 0.86130 | 
| Logistic_Regression_00 | 0.52740 | 0.47327 | 0.51095 | 0.52737 |  
| Logistic_Regression_01 | 0.70625 | 0.67722 | 0.66970 | 0.70524 | 
| Logistic_Regression_02 | 0.49147 | 0.47439 | 0.51728 | 0.49249 | 
| Logistic_Regression_03 | 0.50047 | 0.46138 | 0.45757 | 0.50149 | 
| K_Nearest_Neighbors_00 | 0.27424 | 0.26853 | 0.29323 | 0.27424 | 
| K_Nearest_Neighbors_01 | 0.35727 | 0.31316 | 0.34434 | 0.35727 | 
| K_Nearest_Neighbors_02 | 0.34381 | 0.33723 | 0.36319 | 0.34381 | 
| K_Nearest_Neighbors_03 | 0.26551 | 0.25672 | 0.26872 | 0.26551 | 