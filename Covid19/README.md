# Covid-19 Dataset Analysis
## Project overview
The Covid-19 virus crushed the world during the year 2019-2020, killing thousands of people, destroyed companies and the economy of many countries - among other things. It became primarly important to analyze every aspect known of the virus to prevent another outbreak, synthesize a vaccine or understand how to manage the people likely to be contaminated considering their symptoms. 
The current dataset focuses on this ending point. Using different analysis conducted on patients, the objective is to determine if they are really infected (false positives) and therefore which unit they need to be conducted to.

### Abstract : 
_« Until March 27, the state of São Paulo had recorded 1,223 confirmed cases of COVID-19, with 68 related deaths, while the county of São Paulo, with a population of approximately 12 million people and where Hospital Israelita Albert Einstein is located, had 477 confirmed cases and 30 associated death, as of March 23. Both the state and the county of São Paulo decided to establish quarantine and social distancing measures, that will be enforced at least until early April, in an effort to slow the virus spread.
One of the motivations for this challenge is the fact that in the context of an overwhelmed health system with the possible limitation to perform tests for the detection of SARS-CoV-2, testing every case would be impractical and tests results could be delayed even if only a target subpopulation would be tested. »_

## Code and Resources Used
**Python Version:** 3.8
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn, scipy
**Dataset :** https://www.kaggle.com/einsteindata4u/covid19

## Form analysis
The target is the « SARS-Cov-2 exam result » taking « positive » or « negative » state, in a dataset of 5644 lines and 111 columns. The analysis shows 74 quantitatives and 37 qualitatives variables.
Some values are missing and two groups appears separated :
* ~ 76 % missing values for other virus tests
*	~ 89 % missing valeurs for blood analysis

## Substance analysis
For the target :
* 10 % positives
*	90% negatives

It’s very unbalanced, and we will need to sample the negatives results during the subset analysis to get relevant information.
Signification of the variables :
*	Variables standardized, somethimes asymetrics, concerning the blood samples
* age quantile : hard to conclude anything because the data have been mathematically shifted or transformed
* qualitatives variables : are binaries (0, 1) detected/not detected, 

NB : Rhinovirus seems to be anormaly high, this hypothesis needs to be checked later.

Relation variables to target 
:
* Target/Blood, idea of features that may be correlated :
* Leucocyte
*	Monocyte
*	Platelets

--> These rates are different between patient positively and negatively tested for the Covid1. We have to check later if it seems likely correlated.
