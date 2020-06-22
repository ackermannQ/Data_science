# Bank Campaign

[Project overview](https://github.com/ackermannQ/Data_science/tree/master/2nd%20Project%20-%20Bank%20Campaign#project-overview)
* [Abstract](https://github.com/ackermannQ/Data_science/blob/master/2nd%20Project%20-%20Bank%20Campaign/README.md#abstract-)

[Resources Used](https://github.com/ackermannQ/Data_science/blob/master/2nd%20Project%20-%20Bank%20Campaign/README.md#resources-used)

[Exploratory Data Analysis](https://github.com/ackermannQ/Data_science/blob/master/2nd%20Project%20-%20Bank%20Campaign/README.md#exploratory-data-analysis)
* [Form analysis](https://github.com/ackermannQ/Data_science/blob/master/2nd%20Project%20-%20Bank%20Campaign/README.md#form-analysis)  
* [Substance analysis](https://github.com/ackermannQ/Data_science/blob/master/2nd%20Project%20-%20Bank%20Campaign/README.md#substance-analysis)  
* [Advanced analysis](https://github.com/ackermannQ/Data_science/blob/master/2nd%20Project%20-%20Bank%20Campaign/README.md#advanced-analysis)
* [Conclusions](https://github.com/ackermannQ/Data_science/blob/master/2nd%20Project%20-%20Bank%20Campaign/README.md#conclusions)

[Preprocessing and encoding](https://github.com/ackermannQ/Data_science/blob/master/2nd%20Project%20-%20Bank%20Campaign/README.md#preprocessing-and-encoding)
* [Preprocessing](https://github.com/ackermannQ/Data_science/blob/master/2nd%20Project%20-%20Bank%20Campaign/README.md#preprocessing)
* [Encoding](https://github.com/ackermannQ/Data_science/blob/master/2nd%20Project%20-%20Bank%20Campaign/README.md#encoding)
* [Conclusions](https://github.com/ackermannQ/Data_science/blob/master/2nd%20Project%20-%20Bank%20Campaign/README.md#conclusions-1)

[Modelisation](https://github.com/ackermannQ/Data_science/blob/master/2nd%20Project%20-%20Bank%20Campaign/README.md#modelisation)
* [Set of models tested](https://github.com/ackermannQ/Data_science/blob/master/2nd%20Project%20-%20Bank%20Campaign/README.md#set-of-models-tested)
* [Model optimization](https://github.com/ackermannQ/Data_science/blob/master/2nd%20Project%20-%20Bank%20Campaign/README.md#model-optimization)
* [Prediction Recall Curve](https://github.com/ackermannQ/Data_science/blob/master/2nd%20Project%20-%20Bank%20Campaign/README.md#precision-recall-curve)

[Conclusion](https://github.com/ackermannQ/Data_science/blob/master/2nd%20Project%20-%20Bank%20Campaign/README.md#conclusion)

[Annexes](https://github.com/ackermannQ/Data_science/blob/master/2nd%20Project%20-%20Bank%20Campaign/README.md#annexes)


## [Project overview](https://github.com/ackermannQ/Data_science/tree/master/2nd%20Project%20-%20Bank%20Campaign#bank-campaign)
Predictions are used in a variety of fields, one of them is marketing. How great would it be to target the customers more likely to subscrive to an offer ? This is on what focus this dataset. Extracted from a bank marketing campaign, the objective is to determine the key strategy to make a potential client subscribe, based on the features availabled !

### [Abstract](https://github.com/ackermannQ/Data_science/tree/master/2nd%20Project%20-%20Bank%20Campaign#bank-campaign) :
_«  It is a dataset that describing Portugal bank marketing campaigns results.
Conducted campaigns were based mostly on direct phone calls, offering bank client to place a term deposit.
If after all marking afforts client had agreed to place deposit - target variable marked 'yes', otherwise 'no'»_

## [Resources Used](https://github.com/ackermannQ/Data_science/tree/master/2nd%20Project%20-%20Bank%20Campaign#bank-campaign)
**Python Version:** 3.8.

**Packages:** pandas, numpy, sklearn, matplotlib, seaborn, scipy.

**Dataset :** https://www.kaggle.com/volodymyrgavrysh/bank-marketing-campaigns-dataset

**Documentation of the library created:** [Data library documentation](https://ackermannq.github.io/Data_lib_documentation/)

## [Exploratory Data Analysis](https://github.com/ackermannQ/Data_science/tree/master/2nd%20Project%20-%20Bank%20Campaign#bank-campaign)

It's good to get familiar with the dataset using the pandas head() function once the dataframe is loaded in a variable called df  :

```python    
df = load_dataset(dataset_path=DATASET_PATH, filetype='csv', separator=';')
print(df.head())
shapeOfDF(df)
typeOfDFValues(df)
```

Row number | age | ... | y
----- | ----- | ----- | ----- 
0 | 56 | ... | no
1 | 57 | ... | no
2 | 37 | ... | no
3 | 40 | ... | no
4 | 56 | ... | no

[5 rows x 21 columns]

### [Form analysis](https://github.com/ackermannQ/Data_science/tree/master/2nd%20Project%20-%20Bank%20Campaign#bank-campaign)
The target is « y » column, taking « yes » or « no » qualitiatives values is the person subscribed or not to the offer, in a dataset of 41188 lines and 21 columns. The analysis shows 10 quantitatives and 11 qualitatives variables.

It's seems that all the columns are filled up with values :

```python    
print(missing_values_percentage(df, 0.9))
print(missing_rate(df))
```

Variable name | Percentage
-------- | --------
age | 0.000000
euribor3m | 0.000000
cons.conf.idx | 0.000000
cons.price.idx | 0.000000
emp.var.rate | 0.000000
... | ...
education | 0.000000
marital | 0.000000
job | 0.000000
day_of_week | 0.000000
y | 0.000000

However, we would have to check all the values are usefull for the interpretion of the dataset

### [Substance analysis](https://github.com/ackermannQ/Data_science/tree/master/2nd%20Project%20-%20Bank%20Campaign#bank-campaign)
Let's analyze the target, especially the rate of positive and negative respondings :

Using the analyse_target from the data science library I developed previsouly, we get :

```python
analyse_target(df, "SARS-Cov-2 exam result", True)    
```

Results for 'yes' and 'no' responses :

    11% positives ;
    89% negatives.

It’s very unbalanced, so we will sample the negatives results during the subset analysis to get relevant information. Let's have a look at our variables with the draw_histograms() function :

* <ins>Quantitatives :</ins>
```python
draw_histograms(df, 'float')
```
![Histograms](https://raw.githubusercontent.com/ackermannQ/Data_science/master/2nd%20Project%20-%20Bank%20Campaign/Plots/Quantitatives/Sumup.png)


* <ins>Qualitatives :</ins>
```python
draw_histograms(df, 'object')
```
![MonthPie](https://raw.githubusercontent.com/ackermannQ/Data_science/master/2nd%20Project%20-%20Bank%20Campaign/Plots/Qualitatives/Pie_objects_month.png)
Some months are more exploited for the prospection, maybe extending the time allocated for some of them may improve the result

A better way of examining the data is to check the count plots :

```python
count_histogram(df, 'job', 'y')
count_histogram(df, 'marital', 'y')
count_histogram(df, 'contact', 'y')
count_histogram(df, 'loan', 'y')
count_histogram(df, 'housing', 'y')
count_histogram(df, 'age', 'y')
```
![countHisto](https://raw.githubusercontent.com/ackermannQ/Data_science/master/2nd%20Project%20-%20Bank%20Campaign/Plots/Target-variables%20relations/sumup.png)

* In terms of marital situation, married consumers more often agreed to the service, in relative terms the single was responded better.
* The best channel is celullar, probably because it's also the most commun, fewer people tend to use a landline
* Great difference appears between consumers already using the banks services and received a loan


## [Advanced analysis](https://github.com/ackermannQ/Data_science/tree/master/2nd%20Project%20-%20Bank%20Campaign#bank-campaign)


### [Conclusions](https://github.com/ackermannQ/Data_science/tree/master/2nd%20Project%20-%20Bank%20Campaign#bank-campaign)


## [Preprocessing and encoding](https://github.com/ackermannQ/Data_science/tree/master/2nd%20Project%20-%20Bank%20Campaign#bank-campaign)
### [Preprocessing](https://github.com/ackermannQ/Data_science/tree/master/2nd%20Project%20-%20Bank%20Campaign#bank-campaign)

### [Encoding](https://github.com/ackermannQ/Data_science/tree/master/2nd%20Project%20-%20Bank%20Campaign#bank-campaign)

### [Conclusions](https://github.com/ackermannQ/Data_science/tree/master/2nd%20Project%20-%20Bank%20Campaign#bank-campaign)

## [Modelisation](https://github.com/ackermannQ/Data_science/tree/master/2nd%20Project%20-%20Bank%20Campaign#bank-campaign)

### [Set of models tested](https://github.com/ackermannQ/Data_science/tree/master/2nd%20Project%20-%20Bank%20Campaign#bank-campaign)

### [Model optimization](https://github.com/ackermannQ/Data_science/tree/master/2nd%20Project%20-%20Bank%20Campaign#bank-campaign)
### [Precision Recall Curve](https://github.com/ackermannQ/Data_science/tree/master/2nd%20Project%20-%20Bank%20Campaign#bank-campaign)


## [Conclusion](https://github.com/ackermannQ/Data_science/tree/master/2nd%20Project%20-%20Bank%20Campaign#bank-campaign)



## [Annexes](https://github.com/ackermannQ/Data_science/tree/master/2nd%20Project%20-%20Bank%20Campaign#bank-campaign)
Feel free to consult the documentation of the library I developped : [Data library documentation](https://ackermannq.github.io/Data_lib_documentation/)
