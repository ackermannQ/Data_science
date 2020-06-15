# Covid-19 Dataset Analysis


[Project overview](https://github.com/ackermannQ/Data_science/blob/master/1st%20Project%20-%20Covid19/README.md#project-overview)
* [Abstract](https://github.com/ackermannQ/Data_science/blob/master/1st%20Project%20-%20Covid19/README.md#abstract-)

[Resources Used](https://github.com/ackermannQ/Data_science/blob/master/1st%20Project%20-%20Covid19/README.md#resources-used)

[Exploratory Data Analysis](https://github.com/ackermannQ/Data_science/blob/master/1st%20Project%20-%20Covid19/README.md#exploratory-data-analysis)
* [Form analysis](https://github.com/ackermannQ/Data_science/blob/master/1st%20Project%20-%20Covid19/README.md#form-analysis)  
* [Substance analysis](https://github.com/ackermannQ/Data_science/blob/master/1st%20Project%20-%20Covid19/README.md#substance-analysis)  
* [Advanced analysis](https://github.com/ackermannQ/Data_science/blob/master/1st%20Project%20-%20Covid19/README.md#advanced-analysis)
* [Conclusions](https://github.com/ackermannQ/Data_science/blob/master/1st%20Project%20-%20Covid19/README.md#conclusions)

[Preprocessing and encoding](https://github.com/ackermannQ/Data_science/blob/master/1st%20Project%20-%20Covid19/README.md#preprocessing-and-encoding)

[Modelization](https://github.com/ackermannQ/Data_science/blob/master/1st%20Project%20-%20Covid19/README.md#modelization)

Conclusion


## [Project overview](https://github.com/ackermannQ/Data_science/blob/master/1st%20Project%20-%20Covid19/README.md#covid-19-dataset-analysis)
The Covid-19 virus crushed the world during the year 2019-2020, killing thousands of people, destroyed companies and the economy of many countries - among other things. It became primarly important to analyze every aspect known of the virus to prevent another outbreak, synthesize a vaccine or understand how to manage the people likely to be contaminated considering their symptoms.

The current dataset focuses on this ending point. Using different analysis conducted on patients, the objective is to determine if they are really infected (false positives) and therefore which unit they need to be conducted to.

### [Abstract](https://github.com/ackermannQ/Data_science/blob/master/1st%20Project%20-%20Covid19/README.md#covid-19-dataset-analysis) : 
_« Until March 27, the state of São Paulo had recorded 1,223 confirmed cases of COVID-19, with 68 related deaths, while the county of São Paulo, with a population of approximately 12 million people and where Hospital Israelita Albert Einstein is located, had 477 confirmed cases and 30 associated death, as of March 23. Both the state and the county of São Paulo decided to establish quarantine and social distancing measures, that will be enforced at least until early April, in an effort to slow the virus spread.
One of the motivations for this challenge is the fact that in the context of an overwhelmed health system with the possible limitation to perform tests for the detection of SARS-CoV-2, testing every case would be impractical and tests results could be delayed even if only a target subpopulation would be tested. »_

## [Resources Used](https://github.com/ackermannQ/Data_science/blob/master/1st%20Project%20-%20Covid19/README.md#covid-19-dataset-analysis)
**Python Version:** 3.8.

**Packages:** pandas, numpy, sklearn, matplotlib, seaborn, scipy.

**Dataset :** https://www.kaggle.com/einsteindata4u/covid19


## [Exploratory Data Analysis](https://github.com/ackermannQ/Data_science/blob/master/1st%20Project%20-%20Covid19/README.md#covid-19-dataset-analysis)

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

[5 rows x 111 columns]

### [Form analysis](https://github.com/ackermannQ/Data_science/blob/master/1st%20Project%20-%20Covid19/README.md#covid-19-dataset-analysis)
The target is the « SARS-Cov-2 exam result » taking « positive » or « negative » state, in a dataset of 5644 lines and 111 columns. The analysis shows 74 quantitatives and 37 qualitatives variables.



A very large part of the dataset is missing :

```python
def missing_values_percentage(df, rate_checked):
    """
    Print out the percentage of the missing values, compared to the rate checked
    :param rate_checked: How many values are missing compared to this rate
    :param df: Dataframe used
    :return: The global percentage of missing values in the dataset
    """
    missing_values = (checkNan(df).sum() / df.shape[0]).sort_values(ascending=True)
    print(len(missing_values[missing_values > rate_checked])  # Ex : rate_checked = 0.98 : 98% of missing values
        / len(missing_values[missing_values > 0.0]))  # Give the percentage of missing values > 90% compared to all
    # the missing values : 68 % (more than half the variables are > 90% of NaN)
    return missing_values

def missing_rate(df):
    """
    Get for each column (feature) the percentage of missing value
    :param df: Dataframe used
    :return: Percentage of missing value
    """
    return df.isna().sum() / df.shape[0]

```

Variable name | Percentage
-------- | --------
Patient ID | 0.000000
Patient age quantile | 0.000000
SARS-Cov-2 exam result | 0.000000
Patient addmited to regular ward (1=yes, 0=no) | 0.000000
Patient addmited to semi-intensive unit (1=yes, 0=no) | 0.000000
... | ...
HCO3 (arterial blood gas analysis) | 0.995216
pO2 (arterial blood gas analysis) | 0.995216
Arteiral Fio2 | 0.996456
Phosphor | 0.996456
ctO2 (arterial blood gas analysis) | 0.995216


Some values are missing (displayed one the next plot).

Two main groups appears separated :
* ~ 76 % missing values corresponding to virus diagnostic ;
* ~ 89 % missing values corresponding to blood analysis.

Moreover it's easier to understand how much values are actually missing using a visual representation :
![Representation of missing values](https://raw.githubusercontent.com/ackermannQ/Data_science/master/1st%20Project%20-%20Covid19/images/MissingValues.png)

_In black_ : Actual values

_In pink_ : Missing values

It's going to be difficult to extrapolate the missing data, therefore every variable with a rate of 90% missing values are going to be removed.
A function is created to conserve the variables with less than 90% of missing values :

```python
def keep_values(df, percentage_to_keep=0.9):
    """
    Keeps the values where there are less than a certain percentage of missing values
    :param df: Dataframe used
    :param percentage_to_keep: Percentage to conserve
    :return: A new dataframe where the variables with more than percentage_to_keep are conserved
    """
    return df[df.columns[df.isna().sum() / df.shape[0] < percentage_to_keep]]  # Keep the values where there are
    # less than 90% of missing values
    
df = keep_values(df, percentage_to_keep=0.9)   
```

Once this function is applied only 39 columns remains, easing the treatment of the data.
Finally, the "Patient ID" is not a relevant parameter and can be removed using :
```python
def dropColumn(df, columnName):
    """
    Remove a column
    :param df: Dataframe used
    :param columnName: Name of the column to remove
    :return: A dataframe without the column droped
    """
    return df.drop(columnName, axis=1)

df = dropColumn(df, 'Patient ID')
```

### [Substance analysis](https://github.com/ackermannQ/Data_science/blob/master/1st%20Project%20-%20Covid19/README.md#covid-19-dataset-analysis)
Now let's analyze furthermore the target, especially the rate of positive and negative results :

```python
def analyse_target(df, target, normalized=False):
    """
    Compares the number (or percentage) of each value taken by the feature
    :param df: Dataframe used
    :param target: The feature analyzed
    :param normalized: True: Give the proportion of each value taken by the feature
    :return: The number/proportion of each value taken by the feature (ex :
    negative    1
    positive    9

    or

    negative    0.1
    positive    0.9
    """
    print(df[target].value_counts(normalize=normalized))
    return df[target].value_counts(normalize=normalized)
    
analyse_target(df, "SARS-Cov-2 exam result", True)    
```
Results of the test for SARS-Cov-2 :
* 10% positives ;
* 90% negatives.

It’s very unbalanced, and will require to sample the negatives results during the subset analysis to get relevant information.
Let's have a look at our variables :
```python
def draw_histograms(df, data_type='float', nb_columns=4):
    """
    Draw the histograms/plot pie of the quantitatives/qualitatives variables
    :param df: Dataframe used
    :param data_type: type of the data : int, int64, float, float64 or object
    :param nb_columns: Number of column for the subplot created to display the plots
    """
    cols = df.select_dtypes(data_type)
    ceiling = math.ceil(len(cols.columns) / nb_columns)
    f, axes = plt.subplots(nb_columns, ceiling, figsize=(7, 7), sharex=True)
    for index, col in enumerate(cols):
        col_index = index % nb_columns
        row_index = index // nb_columns

        if data_type == 'float' or data_type == 'int':
            sns.distplot(df[col], ax=axes[row_index, col_index])

        if data_type == 'object':
            df[col].value_counts().plot.pie()
            plt.show()
    plt.show()
```

* <ins>Quantitatives :</ins>
```python
draw_histograms(df, 'float')
```

![Quantitatives](https://raw.githubusercontent.com/ackermannQ/Data_science/master/1st%20Project%20-%20Covid19/images/Variables_plots/Figure_1.png)


* <ins>Qualitatives :</ins>
```python
draw_histograms(df, 'object')
```

![Qualitatives](https://raw.githubusercontent.com/ackermannQ/Data_science/master/1st%20Project%20-%20Covid19/images/Variables_plots/Rhino.png)


* <ins>Age quantile :</ins>

![Age quantile](https://raw.githubusercontent.com/ackermannQ/Data_science/master/1st%20Project%20-%20Covid19/images/Variables_plots/Age.png)

```python
sns.distplot(df['Patient age quantile'])
plt.show()
```

Signification of the variables :
* Variables standardized, somethimes asymetrics, concerning the blood samples ;
* Age quantile : hard to conclude anything because the data have been mathematically shifted or transformed ;
* Qualitatives variables : are binaries (0/1, ex: detected/not detected).

<ins>NB :</ins> Rhinovirus seems to be anormaly high, this hypothesis needs to be checked later.

Relation variables to target :

The first thing to do in order to facilitate the analysis is to create subsets associated with the data coming from blood exams and viral exams, and to regroup positive and negative result of the Covid19 test exam:
```python
def qual_to_quan(df, target, criteria):
    """
    Creates a subset (or collection) of the target with a certain criteria
    Ex: positive_df = qual_to_quan(df, "SARS-Cov-2 exam result", 'positive') creates a subset of the positive results
    to the Covid19 exam
    :param df: Dataframe used
    :param target: Target we want to create a subset from
    :param criteria: Criteria used to create a subset
    :return: A dataframe responding to the criteria chosed
    """
    return df[df[target] == criteria]

positive_df = qual_to_quan(df, "SARS-Cov-2 exam result", 'positive')
negative_df = qual_to_quan(df, "SARS-Cov-2 exam result", 'negative')

def missing_rate(df):
    """
    Get for each column (feature) the percentage of missing value
    :param df: Dataframe used
    :return: Percentage of missing value
    """
    return df.isna().sum() / df.shape[0]

def rate_borned(df, missing_rate, rate_inf, rate_sup):
    """
    Creates a subset based on the missing rates
    Ex:
    blood_columns = rate_borned(df, MR, 0.88, 0.9) create column where the missing rate MR is included between
    0.88 and 0.9
    :param df: Dataframe used
    :param missing_rate: missing_rate to compare the rates
    :param rate_inf: Decision rate inf
    :param rate_sup: Decision rate sups
    :return: The column labels of the DataFrame corresponding to the criteria missing rate, rate inf and sup

    """
    return df.columns[(missing_rate < rate_sup) & (missing_rate > rate_inf)]
    
MR = missing_rate(df)
blood_columns = rate_borned(df, MR, 0.88, 0.9)
viral_columns = rate_borned(df, MR, 0.75, 0.88)

```

Now, a relation between these subset can be searched :
```python
def display_relations(column_name, relation):
    """
    Display the relation between diff
    display_relations(blood_columns, relation)
    :param column_name: Column the relation are being tested with
    :param relation: List of relation to observe
    Ex : relation = [(positive_df, 'positive'), (negative_df, 'negative')] shows the relation between the
    blood_column and the positive and negative results
    :return:
    """
    for col in column_name:
        plt.figure()
        for rel, lab in relation:
            sns.distplot(rel[col], label=lab)
        plt.legend()
    plt.show()
    
relation = [(positive_df, 'positive'), (negative_df, 'negative')]
display_relations(blood_columns, relation)
```
Some parameters doesn't seem to vary if the result of the exam is positive or negative:

![Hematocrit](https://raw.githubusercontent.com/ackermannQ/Data_science/master/1st%20Project%20-%20Covid19/images/Variables_plots/Hematocrit.png)


At the contrary, three variables seems to be correlated to the SARS-Cov-2 exam result.

* Target/Blood, idea of features that may be correlated :
  * Leukocytes ;
  *	Monocytes ;
  *	Platelets.


![Leukocytes](https://raw.githubusercontent.com/ackermannQ/Data_science/master/1st%20Project%20-%20Covid19/images/Variables_plots/Leukocytes.png)

![Monocytes](https://raw.githubusercontent.com/ackermannQ/Data_science/master/1st%20Project%20-%20Covid19/images/Variables_plots/Monocytes.png)

![Platelets](https://raw.githubusercontent.com/ackermannQ/Data_science/master/1st%20Project%20-%20Covid19/images/Variables_plots/Platelets.png)


--> These rates are different between patients positively and negatively tested for the Covid19. We have to check later if it seems correlated.

*	Target/Age : Young individuals seems less likely to be tested positives (it doesn’t mean they are not infected). The exact age is unknown ;

```python
def count_histogram(df, x, hue, show=True):
    """
    Shows the counts of observations in each categorical bin using bars
    :param show: True to display the plot
    :param df: Dataframe used
    :param x: abscisse
    :param hue:Legend title
    """
    sns.countplot(x=x, hue=hue, data=df)
    if show:
        plt.show()

count_histogram(df, 'Patient age quantile', 'SARS-Cov-2 exam result')
```

![Target_Age](https://raw.githubusercontent.com/ackermannQ/Data_science/master/1st%20Project%20-%20Covid19/images/Variables_plots/Target_Age.png)

*	Target/Viral : It’s rare to find people with more than one sickness at a time.

```python
def crossTable(df, cross1, cross2):
    """
    Compute a simple cross tabulation of two (or more) factors
    :param df: Dataframe used
    :param cross1: First variable to cross with
    :param cross2: Second variable to cross with
    :return: Cross tabulation of the data
    """
    return pd.crosstab(df[cross1], df[cross2])


def crossTables(df, column_name, cross):
    """
    Compute a cross tab for every value of the column with a parameter
    :param df: Dataframe used
    :param column_name: The column where the values are taken from
    :param cross: The parameter which the one the values are crossed with
    """
    cols = column_name.unique()
    ceiling = math.ceil(len(cols) / 5)
    f, axes = plt.subplots(5, ceiling, figsize=(12, 12), sharex=True)
    for index, col in enumerate(cols):
        col_index = index % 5
        row_index = index // 5
        sns.heatmap(pd.crosstab(df[cross], df[col]), annot=True, fmt='d', ax=axes[col_index, row_index])

    plt.show()
    
crossTables(df, viral_columns, "SARS-Cov-2 exam result")    
```

![crossTable_sickness](https://raw.githubusercontent.com/ackermannQ/Data_science/master/1st%20Project%20-%20Covid19/images/Variables_plots/crossTable_sickness.png)



As already said, Rhinovirus/Entérovirus positive may implied a negative Covid19 result. This hypothesis requires to be validate because it’s likely that the area from where the data are collected just suffered an outberak simultenously to the Covid19.

It may be unrelated.


## [Advanced analysis](https://github.com/ackermannQ/Data_science/blob/master/1st%20Project%20-%20Covid19/README.md#covid-19-dataset-analysis)

* Relation between the variables :
  * Blood_data / Blood_data : some variables are correlated (+ 0.9 !) ;
  
```python
def pairwise_relationships(df, variable, cluster=True):
    """
    Display a pairplot, clustermap or heatmap
    :param df: Dataframe used
    :param variable: Variable studied
    :param cluster: True: Clustermap display, False: Heatmap displayed
    """
    sns.pairplot(df[variable])
    if cluster:
        sns.clustermap(df[variable].corr())
    else:
        sns.heatmap(df[variable].corr())
    plt.show()
    
    
pairwise_relationships(df, blood_columns)
```
  
![Visu_correlation](https://raw.githubusercontent.com/ackermannQ/Data_science/master/1st%20Project%20-%20Covid19/images/Variables_plots/Visu_correlation.png)


  * Blood_data / Age : week correlation ;
  
  ```python
  def check_correlation(df, value_for_correlation):
    """
    Check if the featurs are correlated
    :param df: Dataframe used
    :param value_for_correlation: Specified the value with which every other parameter would be correlated checked
    :return: The values corresponding to the correation between value_for_correlation and every other parameters
    """
    print(df.corr()[value_for_correlation].sort_values())
    return df.corr()[value_for_correlation].sort_values()

check_correlation(df, "Patient age quantile"))
```    

Variable name | Correlation
-------- | --------
Leukocytes          |                                    -0.166386
Platelets            |                                   -0.158683
Lymphocytes           |                                  -0.125935
Mean corpuscular hemoglobin concentration (MCHC) |       -0.124671
Red blood Cells          |                               -0.037510
Patient addmited to intensive care unit (1=yes, 0=no) |  -0.035772
Patient addmited to semi-intensive unit (1=yes, 0=no) |   0.015736
Eosinophils                    |                          0.022085
Patient addmited to regular ward (1=yes, 0=no)    |       0.046166
Monocytes                      |                          0.050962
Hemoglobin        |                                       0.060320
Hematocrit        |                                       0.096808
Basophils            |                                    0.107525
Mean platelet volume       |                              0.119449
Red blood cell distribution width (RDW)    |              0.166429
Mean corpuscular hemoglobin (MCH)       |                 0.197394
Mean corpuscular volume (MCV)       |                     0.281655
Patient age quantile            |                         1.000000

The highest correlation is 0.281655, which is very week

  * Viral / Viral : After some research, the influeza rapid tests giv bad results and may need to be droped ;
  
  
  *	Relation sickness / Blood_data : Blood rates between regular patient and Covid19 patient are different (lymphocyte, hemoglobine et hematocrit) ;
  ```python
  def display_relations(column_name, relation):
    """
    Display the relation between diff
    display_relations(blood_columns, relation)
    :param column_name: Column the relation are being tested with
    :param relation: List of relation to observe
    Ex : relation = [(positive_df, 'positive'), (negative_df, 'negative')] shows the relation between the
    blood_column and the positive and negative results
    :return:
    """
    for col in column_name:
        plt.figure()
        for rel, lab in relation:
            sns.distplot(rel[col], label=lab)
        plt.legend()
    plt.show()
    
display_relations(blood_columns, relation)
  ```

![Differences](https://raw.githubusercontent.com/ackermannQ/Data_science/master/1st%20Project%20-%20Covid19/images/Variables_plots/Differences.png)

  * Relation hospitalisation/blood : Important parameter if we need to predict the service a patient needs to be orientated to !
  
  ```python
      def hospitalisation(df):
        if df['Patient addmited to regular ward (1=yes, 0=no)'] == 1:
            return 'Surveillance'

        elif df['Patient addmited to semi-intensive unit (1=yes, 0=no)'] == 1:
            return 'Semi-intensives'

        elif df['Patient addmited to intensive care unit (1=yes, 0=no)'] == 1:
            return 'ICU'

        else:
            return 'Unknown'
    
  def relation_in_newcol(df, column, newcol, show=False):
    """
    Display the relation between a specified column and another one
    :param show: True: Display the plot(s)
    :param df: Dataframe used
    :param column: First column
    :param newcol: Second column
    :return:
    """
    cols = column.unique()
    ceiling = math.ceil(len(cols) / 5)
    f, axes = plt.subplots(5, ceiling, figsize=(12, 12), sharex=True)
    for index, col in enumerate(cols):
        plt.figure()
        col_index = index % 5
        row_index = index // 5
        for cat in newcol.unique():
            sns.distplot(df[newcol == cat][col], label=cat, ax=axes[col_index, row_index])
        plt.legend()

    if show:
        plt.show()

df['status'] = df.apply(hospitalisation, axis=1)
relation_in_newcol(df, blood_columns, df['status'], True)

  ```
![Hospitalisation](https://raw.githubusercontent.com/ackermannQ/Data_science/master/1st%20Project%20-%20Covid19/images/Variables_plots/Hospitalisation.png)  
  
  
*	NaN analyse : viral 1350 (92%/8%), blood sample 600 (87%/13%), previously : 90% of the dataset.


--> Some parameters are not related, as shown for the MCH and the hopsitalisation service :

![MCH](https://raw.githubusercontent.com/ackermannQ/Data_science/master/1st%20Project%20-%20Covid19/images/MCH.png)

Others seems to be in direct correlation with the service where the patient get into !
![Lymphocytes](https://raw.githubusercontent.com/ackermannQ/MachineLearning/master/1st%20Project%20-%20Covid19/images/Lymphocytes.png)

![Monocytes](https://raw.githubusercontent.com/ackermannQ/MachineLearning/master/1st%20Project%20-%20Covid19/images/Monocytes.png)



__Student’s test (H0) :__
*	Patients infected with Covid19 have higher leucocyte, monocyte et platelets (+ eosinophils) rate than regular individuals ;
  *	H0 Rejected = These average rates are EQUALS between people tested positive and negative to covid-19.

X : Not relevant because it was not an hypothesis needed to be tested.


Hematocrit | Hemoglobin | Platelets | MPV | Red blood Cells | Lymphocytes | MCHC | Leukocytes | Basophils | MCH | Eosinophils | MCV | Monocytes | RDW
------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ 
X | X | H0 rejected | X | X | X | X | H0 rejected | X | X | H0 rejected | X | H0 rejected | X


### [Conclusions](https://github.com/ackermannQ/Data_science/blob/master/1st%20Project%20-%20Covid19/README.md#covid-19-dataset-analysis)
*	A large part of the dataset is missing : only 20% is exploitable ;
*	Two main groups interesting : blood and viral analysis ;
*	The blood sample can’t give the certainty of Covid19 cases ;
*	Some missing values need to be replaced, we can’t just drop them all. If we do so, we get 99 lines instead of 5644, so we lose to much information ;
*	Blood_column : 600 values, viral_column : 1354.


## [Preprocessing and encoding](https://github.com/ackermannQ/Data_science/blob/master/1st%20Project%20-%20Covid19/README.md#covid-19-dataset-analysis)
* We create a brand new dataset we can work on without corrupting the previous one
* The missing values are eliminated
* The dataset is splitted between a trainset and a trainset

```python
df2 = load_dataset(dataset_path=DATASET_PATH)  # Working on a different version of the dataset is a good practice
MR2 = missing_rate(df2)
blood_columns2 = list(rate_borned(df2, MR2, 0.88, 0.9))
viral_columns2 = list(rate_borned(df2, MR2, 0.75, 0.88))
important_columns = ['Patient age quantile', 'SARS-Cov-2 exam result']

df2 = df2[important_columns + blood_columns2]  # + viral_columns2], finally viral_columns2 does not have a great impact

trainset, testset = train_test_split(df2, test_size=0.2, random_state=0)
```    

### Encoding 
Creation of a dictionnary to change the qualitative values to quantitatives
```python
def encoding(df, type_values, swap_these_values):
    """
    Encode qualitatives values
    :param df: Dataframe used
    :param type_values: Type of the value worked with
    :param swap_these_values: Swap the values {"Value 0": 0, "Value1": 1, ...}
    :return: A dataframe with the qualitatives values swaped to quantitatives values
    """
    for col in df.select_dtypes(type_values).columns:
        df.loc[:, col] = df[col].map(swap_these_values)
    return df
swap_values = {'positive': 1, 'negative': 0, 'detected': 1, 'not_detected': 0}
encoding(df2, 'object', swap_these_values)
```

Then, an imputation function is created to transform the data.
First, the missing values are removed. Since it's going to create some issues, we are likely to change this function later

```python
def imputation(df):
    """
    Imputation function
    :param df: Dataframe used
    :return: The dataframe with the imputation applied
    """
    return df.dropna(axis=0)
    
imputation(df2)
```

Now, we can use these function to preprocess our dataset:
```python
def preprocessing(df, Target, type_values, swap_these_values, new_feature, column):
    """
    Global preprocessing function
    :param df: Dataframe used
    :param Target: Target variable wanted predicted
    :param type_values: Type of the values
    :param swap_these_values: Values swaped
    :param new_feature: New_feature used for the feature engineering
    :param column: Column concerned by the feature engineering
    :return: X: Features and y: Target to predict
    """
    df = encoding(df, type_values, swap_these_values)
    feature_engineering(df, column)
    df = imputation(df)
    X = dropColumn(df, Target)
    y = df[Target]
    
    return X, y
    
swap_values = {'positive': 1, 'negative': 0, 'detected': 1, 'not_detected': 0}
target = 'SARS-Cov-2 exam result'
X_train, y_train = preprocessing(trainset, target, 'object', swap_values, 'is sick',
                                 viral_columns2)
X_test, y_test = preprocessing(testset, target, 'object', swap_values, 'is sick',
                               viral_columns2)    
```

Thus, the training and testing datasets are created.
Now, a first model is quickly tested to find out the reliability of what we are doing from the beginning
We are going to test a decision tree classifier because it's quick and will give us a good understanding of the important parameters
```python
def evaluation(model, X_train, y_train, X_test, y_test):
    """
    Evaluation of the model, compare on the same plot the prediction on the trainset and the testset
    Display the confusion matrix and classification report
    Shows the overfitting/undefitting
    :param model: model used
    :param X_train: X trainset
    :param y_train: y trainset
    :param X_test: X testset
    :param y_test: y testset
    """
    model.fit(X_train, y_train)
    ypred = model.predict(X_test)

    print(confusion_matrix(y_test, ypred))
    print(classification_report(y_test, ypred))

    N, train_score, val_score = learning_curve(model, X_train, y_train, cv=4, scoring='f1',
                                               train_sizes=np.linspace(0.1, 1, 10))

    plt.figure(figsize=(12, 8))
    plt.plot(N, train_score.mean(axis=1), label='Train score')
    plt.plot(N, val_score.mean(axis=1), label='Validation score')
    plt.legend()
    plt.show()


model = DecisionTreeClassifier(random_state=0)
evaluation(model, X_train, y_train, X_test, y_test)
```

When the learning curves are plotted,  immediately it seems that our model is overfitting : the train set is perfectly learnt but the machine can't adjust to the testing set

<ins>Decision Tree Classifier :</ins>
![DecisionTreeClassifier](https://raw.githubusercontent.com/ackermannQ/Data_science/master/1st%20Project%20-%20Covid19/images/Variables_plots/DecisionTreeClassifier.png) 


[[87  8]
 [10  6]]
 
      X      | precision  |  recall | f1-score |  support
------------ | ------------ | ------------ | ------------ | ------------

           0    |   0.90  |    0.92  |    0.91   |     95
           1    |   0.43   |   0.38  |    0.40   |     16
    accuracy    |     X    |   X     |    0.84   |    111
    macro avg    |   0.66   |   0.65  |    0.65   |    111   
    weighted avg    |   0.83   |   0.84  |    0.83   |    111




## [Modelisation](https://github.com/ackermannQ/Data_science/blob/master/1st%20Project%20-%20Covid19/README.md#covid-19-dataset-analysis)
Four different models were tested and evaluated, using the learning curve method.

<ins>RandomForest :</ins>
Very flexible and can be applied to both classification and regression.
![RandomForest](https://raw.githubusercontent.com/ackermannQ/MachineLearning/master/1st%20Project%20-%20Covid19/images/RandomForest.png)

<ins>AdaBoost :</ins>
![AdaBoost](https://raw.githubusercontent.com/ackermannQ/MachineLearning/master/1st%20Project%20-%20Covid19/images/Adaboost.png)

<ins>Svm :</ins>
![Svm](https://raw.githubusercontent.com/ackermannQ/MachineLearning/master/1st%20Project%20-%20Covid19/images/SVM.png)

<ins>KNN :</ins>
![KNN](https://raw.githubusercontent.com/ackermannQ/MachineLearning/master/1st%20Project%20-%20Covid19/images/KNN.png)
