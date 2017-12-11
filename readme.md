# Ames Housing Data and Kaggle Challenges
## Thomas Brewer

In this project, we are practicing two important skills:

1. Creating and iteratively refining regression and classification models
2. Using [Kaggle](https://www.kaggle.com/) to practice the modeling process

We are tasked with creating two models with the highest possible accuracy based on the Ames Housing Dataset. Those models will predict the following:

- The price of a house at sale (regression)
- Whether a house sale was abnormal or not (classification)

The Ames Housing Dataset is an exceptionally detailed and robust dataset with over 70 columns of different features relating to houses. While the two models we make will predict different targets (and will require different features, model choices, and hyperparameters).

This project also involves two privately hosted competitions on Kaggle (one for the regression and one for the classification) to give us the opportunity to practice the following skills:

- Refining models over time
- Use of train-test split, cross-validation, and data with unknown values for the target to simulate the modeling process
- The use of Kaggle as a place to practice data science

## The Modeling Process

1. The train and test datasets for both challenges are the **same**, with the exception of the target that you are trying to predict.
2. The train dataset has all of the columns that you will need to generate and refine your models. The test dataset has all of those columns except for the two targets that you are trying to predict in your Regression and Classification models.
3. Generate regression and classification models using the training data.  This project makes use of :
    - train-test split
    - cross-validation / grid searching for hyperparameters
    - strong exploratory data analysis to question correlation and relationship across predictive variables
    - code that reproducibly and consistently applies feature transformation (such as the preprocessing library) 
4. Predict the values for your target columns in the test dataset and submit your predictions to Kaggle to see how your model does against unknown data. 