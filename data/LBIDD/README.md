# Linked Births and Infant Deaths Database
 - [Overview](#overview)
 - [Questions](#questions)
   - [Censoring](#censoring)
   - [Scaling](#scaling)
 - [File Types](#file-types)
   - [Covariate File](#covariate-file)
   - [Factual Files](#factual-files)
   - [Counterfactual Files](#counterfactual-files)
 - [File Description](#file-description)
 - [Sources](#sources)
 
## Overview
The Framework utilizes Linked Births and Infant Deaths Database (LBIDD) [[1]](#references)
so as to based on real-world medical measurements.  
This main covariate file in name [`x.csv`](x.csv) and all other simulated are based on this
table, no hidden confounders.

The data contained in this repository is only a small sample out of the entire available datasets.
These are located under the [LBIDD directory](https://www.synapse.org/#!Synapse:syn11738963)
in the [corresponding Synapse project](https://www.synapse.org/IBMCausalityData)'s
Files tab [[2]](#references).

## Questions
There are two main directories for the following two questions:
#### Censoring
Determine the effect of treatment from observational data, 
where some of the subjects in the dataset may be censored, 
i.e. have no observed outcome. 
The censoring may have shared confounding effects with either treatment assignment or with outcome.

#### Scaling
Determine the effect of treatment from observational data, 
using data of various sizes, while taking advantage of this increasing in data size.

## File Types
#### Covariate File
A table which columns' are features and rows are different subjects. 
Derived from LBIDD and used as the basis for all simulated data.    
Since there are differently sized (number-of-samples-wise) counter/factual files, 
each sample (row) in the covariates file will have a unique `sample_id` to match the 
corresponding `sample_id` from the counter/factual files.  
```
| sample_id | x_1 | x_2 | ... | x_m |
|-----------|-----|-----|-----|-----|
|           |     |     |     |     |
|           |     |     |     |     |
|           |     |     |     |     |
```

#### Factual-Files
Each factual-file based on a sub-group of samples from the covariates file and created 
by a different data generating processes describing some relation between the treatment 
assignment, the outcome (the censoring, when applicable) and the covariates.  
Each such file will have some unique file identifier as its name (with `.csv` suffix), 
for example `8c5f509.csv` and is a comma-delimited-file with it's first row being a header:
```
| sample_id | z | y |
|-----------|---|---|
|           |   |   |
|           |   |   |
|           |   |   |
```
Where `z` column is the treatment assignment and `y` columns is the observed outcome.
#### Counterfactual-Files 
Serve as the "ground truth" and are to be  evaluated against the estimates. 
Each factual file will have a corresponding counter-factual file, 
having the same file identifier but with a `_cf` suffix to it 
(for example `8c5f509_cf.csv`).  
These files contain both the outcome under no intervention 
($$\(y^0_i\)$$) and the outcome under positive intervention ($$\(y^1_i\)$$) for each 
individual $$\(i\)$$ in the dataset, the same individuals appearing in the corresponding 
factual tables. The counterfactual files are comma-delimited and their first row is a header:
```
| sample_id | y0 | y1 |
|-----------|----|----|
|           |    |    |
|           |    |    |
|           |    |    |
```

## File Description
* [main covariate](x.csv): The main covariate file derived from LBID Database [[1]](#reference).
* [censoring directory](censoring): A directory containing both factual and counterfactual files
  corresponding to them (same file-id but with `_cf.csv` suffix) for the censoring question.
* [scaling directory](scaling): See *censoring directory*.
* [censoring_params.csv](censoring_params.csv): A tabular file containing various parameters used
  for simulating the censored datasets. In case user would like to have a closer look on 
  their algorithm performance.  
  The parameters are as following:
  * **ufid**: Unique file identifier of the simulated data instance 
    (corresponding with the counter/factual file name).
  * **size**: Number of subjects in the dataset.  
  * **%_treated**: Fraction of the population that underwent intervention (from 0% to 100%).
  * **%_censored**: Fraction of the population with no observed outcome (from 0% to 100%).
  * **effect_size**: Average effect size in the population
  * **%_effect_size**: Average effect size divided by the average outcome under intervention (Y^1). 
  * **snr**: Signal-to-noise ratio, stated as signal/(signal+noise) (from 0.0 to 1.0).
  * **link_type**: Type of activation function combining parents variables to create a simulated
    variable (either polynomial, logarithmic or exponential activations)
  * **deg(y)**: The degree of polynomial used to create the outcomes (not applicable to
    *log* and *exp* linking)
  * **deg(z)**: The degree of polynomial used to create the treatment assignment (not applicable to
    *log* and *exp* linking)
  * **deg(c)**: The degree of polynomial used to create the censor decision (not applicable to
    *log* and *exp* linking)
  * **n_conf(y)**: Number of variables that serve as parents to the outcome variable 
    and do not affect the treatment assignment or the censoring decision.
  * **n_conf(z)**: Number of variables that serve as parents to the treatment variable 
    and do not affect the outcome or the censoring decision.
  * **n_conf(c)**: Number of variables that serve as parents to the censor variable 
    and do not affect the treatment assignment or the outcome.
  * **n_conf(yz)**: Number of variables that serve as parents to both the outcome and the treatment variable.
  * **n_conf(cy)**: Number of variables that serve as parents to both the censor and the outcome variable.
  * **n_conf(cz)**: Number of variables that serve as parents to both the censor and the treatment variable.
  * **n_conf(cyz)**: Number of variables that serve as parents to all censor, outcome and treatment variables.
  * **dgp**: Identification number of the data generating process (DGP).
  * **instance**: Specific instance of the above DGP.
  * **random_seed**: Used for reproducibility in internal use.
* [scaling_params.csv](scaling_params.csv): See *censoring_params.csv* and omit un-relevant 
  (censoring-related) parameters.


### Sources
[1] Marian F MacDorman and Jonnae O Atkinson. Infant mortality statistics from the linked birth/infant death data set - 
1995 period data. Mon Vital Stat Rep, 46(suppl 2):1-22, 1998.  
[2] [LBIDD wiki page](https://www.synapse.org/#!Synapse:syn11738767/wiki/512854)