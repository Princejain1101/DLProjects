This is a Kaggle competition problem - Santander Product Recommendation.
Primary intention behind this attempt (3 yrs after launch) was to build a real 
world problem using machine learning algorithms and doing feature and data engineering using python.

Getting back to implentation.

Input:

    The input is a csv file which contains information about the users of a bank.
    Such as date of joining, products bought, age, sex etc. over the period of 1.5 yr
    The data is logged on 28th of each month. (Jan 2015 to May 2016)
Problem:

    The task is the predict what new product the user might buy for next month (june 2016).

Data/Feature Engineering:

    1. There are number of features given for different users for different months.
    2. It is possible that some users have left the bank and some are joining.
    3. A lot of information (csv columns) are strings and can be categorized.
       There are multiple ways to categorize the string('object') data. Such as
       converting ['A', 'B', 'C'] 
       a. to numbers       'A' -> 0,  'B' -> 1,  'C' -> 2
       b. one hot encoding 'A' -> 0, 0
                           'B' -> 0, 1
                           'C' -> 1, 1
       one hot encoding gives better feature separation, but that is not what is implemented
       in this solution. "strToCatCode2" method is doing along with "createDict" method
       achieves this. "createDict" creates a dictionary for given columns based on all the
       string values. This Dictionary is then used by "strToCatCode2" to assign specific
       numbers to the particular field in a columns.
       The same dictionary is then later used to assign values to columns in test data 
       as well because we dont want different numbers for same strings in test and train data.
    4. Information such as Income, rent, age etc. are continuous inputs.
    5. The input data only gives the products used by customer as of 28th of that month,
       but we want to know what product they added/bought this month. So, we have to 
       somehow make a diff from previous month and get a new column ('sol') for the user
       for given month, which tells what new product is added.
       Since there are millions of rows and we will have to operate on each row by row, 
       this takes most of the computation times. 
       Also, we have do this for each user, so it is better to sort the input data by users
       first.
       "getTrain2" method achieves this task.
    Other problems in feature engineering:
    1. Many of the column fields will be Nan which can be handled with fillna('')
    2. Some strings fields also contained string '   Nan' which needs to be stripped
       and replaced with '' i.e. empty string. 
    3. we can also fill nan or 'Nan' with the most common string input of that column
       which is done in this solution (check "createDict" method).
    4. For the numerical data such as renta, age and income, normalization is performed
       to bring the values to same scale.

Machine Learning:

    xgboost library is used to train and predict the model. It is one of the most famous
    machine learning library which uses gradient boosting. The input to xgb is numerical,
    so all the training and test data is first converted to numerical format(int/float).
    
    Brief: Extreme Gradient Boosting-
           Boosting: it is a method of which uses ensemble of decision trees and weak learners
                     and thus reducing bias and variance.
           Gradient: The weak learners are chosen that points in the negative gradient direction
                     for the given differentiable cost function.
           Extreme: optimized and high performant, distributed library which is flexible and portable.
                    It has regularization, handle missing values, better pruning etc. 
    https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/
    https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

    XGB parameters:
        objective: multi:softprob - for multi-class classification with soft probability for each class.
        max_depth: 8 - the depth of the tree to make the model more complex for so many features.
        eta: 0.05 - learning rate.
        silent: 1 - specify how much to print on terminal
        num_class: 24 - since there are 24 products, so 24 classes.
        eval_metric: "mlogloss" - m(multi class) log loss.
        min_child_weight: 1 - default- threshold for partitioning tree. 
        subsample: 0.7 - to prevent overfitting, how much data to sample for training.
        colsample_bytree: 0.7 - columns to sample 
        seed: a random no.
        num_rounds: 50 - number of trees that are built.

    Before feeding the training data to XGB, it is first converted into matrix using
    to_numpy() function.

Prediction:

    Test data is prepared in same fashion as the train data.
    There was one issue with train data though, it only gives the user information but
    not the products it has so far so we take that informatiom from the training data.
    Then xgboost is used to predict the products. As a result we get the probability of
    each product for each user. 7 highest probability products are dumped in the result.
