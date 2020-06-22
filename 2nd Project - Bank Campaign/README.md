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

It's good to get familiair with the dataset using the pandas head() function once the dataframe is loaded in a variable called df  :

```python
def displayHead(df, every_column=False, every_row=False, column_nbr):
    """
    Display the relation between diff
    display_relations(blood_columns, relation)
    :param column_name: Column the relation are being tested with
    :param relation: List of relation to observe
    Ex : relation = [(positive_df, 'positive'), (negative_df, 'negative')] shows the relation between the
    blood_column and the positive and negative results
    :return:
    """
    if every_column:
        pd.set_option('display.max_column', column_nbr)

    if every_row:
        pd.set_option('display.max_row', column_nbr)

    print(df.head())
    return df.head()
    
displayHead(df, every_column=False, every_row=False, column_nbr=111)
```

Row number | Patient ID | ... | ctO2 (arterial blood gas analysis)
----- | ----- | ----- | ----- 
0 | 44477f75e8169d2 | ... | NaN
1 | 126e9dd13932f68 | ... | NaN
2 | a46b4402a0e5696 | ... | NaN
3 | f7d619a94f97c45 | ... | NaN
4 | d9e41465789c2b5 | ... | NaN

### [Form analysis](https://github.com/ackermannQ/Data_science/tree/master/2nd%20Project%20-%20Bank%20Campaign#bank-campaign)
### [Substance analysis](https://github.com/ackermannQ/Data_science/tree/master/2nd%20Project%20-%20Bank%20Campaign#bank-campaign)

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
