# Data-classification---ML
Given the MAGIC gamma telescope dataset that can be obtained using the link below.
https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope. This dataset is generated to simulate
registration of high energy gamma particles in a ground-based atmospheric Cherenkov gamma telescope using
the imaging technique. The dataset consists of two classes; gammas (signal) and hadrons (background). There
are 12332 gamma events and 6688 hadron events. You are required to apply use the dataset to construct
different classification models such as Decision Trees, Naïve Bayes Classifier, Random Forests, AdaBoost
and K-Nearest Neighbor (K-NN). You are also required to tune the parameters of these models, compare the
performance of models with each other.
Lab session

# I.Data Balancing
Note that the dataset is class-imbalanced. To balance the dataset, randomly put aside the extra readings for
the gamma “g” class to make both classes equal in size.

# II.Data Split
Split your dataset randomly so that the training set would form 70% of the dataset and the testing set would
form 30% of it.

# III. Classification
Apply the classifiers from the following models on your dataset, tune parameter(s) (if any), compare the
performance of models with each other

# 1. Decision Tree
Parameters to be tuned: None
# 2. AdaBoost
Parameter to be tuned: n_estimators
# 3. K-Nearest Neighbor (K-NN)
Parameter to be tuned: K
# 4. Random Forests
Parameter to be tuned: n_estimators
# 5. Naïve Bayes
Parameters to be tuned: None
