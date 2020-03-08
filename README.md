## Final project Effective Programming Practices for Economists

This repository contains the final project for the course Effective Programming Practices for Economists at the University of Bonn (Winter Semester 2019/2020).

*Title*: Sensitivity of OLS results to diﬀerent imputation methods in a RCT framework

*Instructor*: Prof. Hans-Martin von Gaudecker

*Authors*: Adelina Garamow, Sofia Badini

<hr />

#### Abstract

In the framework of our Eﬀective Programming Course, we explore the sensitivity of Intention-to-Treat 
estimates to diﬀerent imputation methods using real data from the Growing America through Entrepreneurship 
(GATE) experiment, a longitudinal study conducted by the US Department of Labor in which free 
entrepreneurship training was randomly oﬀered to individuals interested in starting or running a business. 
We contrast the results of the complete-case analysis with four other scenarios in which we complete the 
missing data in the outcome of interest and covariates with the k-Nearest-Neighbor imputation method and 
a random draw from a normal distribution. We also do a bound analysis, by imputing the minimum or maximum 
value for the outcome of interest. The results show that the understanding of the missing pattern is
crucial for the interpretation of the results, based on the imputed data. 
 
<hr />

#### Contributions

Sofia Badini:

+ Set-up of the original datasets in the folder *src.original_data* 
+ Pre-processing of the original data in the folder *src.data_management* 
+ All the files in the folder *src.auxiliary* 
+ ``analysis_functions.py`` in the folder *src.analysis* 
+ All the files in the folder *src.final* 
+ The documentation for all of the previous steps 
+ The sections "The GATE Experiment" and "Missing Data Analysis" in the final paper

Adelina Garamow:

+ Set-up of the skeletton of the project
+ Write functions for the imputation methods and tests, see folder ``imputation_method``
+ Implement the imputation on the data, see folder ``imputation_implement``
+ Writing the "Introduction", "Imputation and Hypothesis", "ITT Estimates" and "Conclusion"



