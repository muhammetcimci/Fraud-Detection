Fraud-Detection
This project was handled to finish the Data Science Bootcamp training organized by the VBO organization. In this project we worked as a group, and it took one month. The group members are:

Berkan ACAR
Mert Ozan INAL
Muhammed CIMCI
Ismail KAYA
Umit CEYLAN
Scope of The Project
The aim of the project is benchmarking machine learning models on a challenging large-scale dataset to determine if transaction is fraud or not.

About Data Set
There was a competition hosted by IEEE Computational Intelligence Society (IEEE-CIS) on Kaggle in 2019. The data is originally coming from the world’s leading payment service company, Vesta Corporation. You can read about the fraud detection competition and find the data set here..

Abstract
The data set used in this project was given as 4 different csv files, two of which are Train sets and the other two belong to the Test set. I fixed the structural errors in the column names of the files and combined the files with left join with Train and Test set. Since more than 95% of the variables given in terms of data privacy are kept confidential, I grouped the variables with different pattern determination techniques. After applying the EDA analysis to the grouped variables, I defined the userid to identify each person to whom the transactions belong. Based on Userid, I did feature engineering and got a Train set with dimension of 590540 x 284 before I entered it into the model. I predicted the Test set using tree-based models like XGBoost, LightGBM and CatBoost using this Train set I obtained. I used the GridSearch method to determine the hyperparameters of these models. Then I applied Kfold to the models and achieved higher accuracy results. The accuracy of these 3 models turned out to be very close to each other. Accurcy values of XGBoost, LightGBM and CatBosst were 92%, 91% and 90%, respectively.
