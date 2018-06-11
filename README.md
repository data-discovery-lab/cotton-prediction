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


