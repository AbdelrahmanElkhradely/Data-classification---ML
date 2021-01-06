MAGIC-telescope-data
====================
#Problem Statement
Given the MAGIC gamma telescope dataset that can be obtained using the link below.
https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope. This dataset is generated to simulate
registration of high energy gamma particles in a ground-based atmospheric Cherenkov gamma telescope using
the imaging technique. The dataset consists of two classes; gammas (signal) and hadrons (background). There
are 12332 gamma events and 6688 hadron events. You are required to apply use the dataset to construct
different classification models such as Decision Trees, Naïve Bayes Classifier, Random Forests, AdaBoost
and K-Nearest Neighbor (K-NN). You are also required to tune the parameters of these models, compare the
performance of models with each other.


#I.Data Balancing (Done)
Note that the dataset is class-imbalanced. To balance the dataset, randomly put aside the extra readings for
the gamma “g” class to make both classes equal in size.


#II.Data Split (Done)
Split your dataset randomly so that the training set would form 70% of the dataset and the testing set would
form 30% of it.

#III. Classification 
Apply the classifiers from the following models on your dataset, tune parameter(s) (if any), compare the
performance of models with each other
1. Decision Tree (Done)
Parameters to be tuned: None

2. AdaBoost
Parameter to be tuned: n_estimators

3. K-Nearest Neighbor (K-NN) (Done)
Parameter to be tuned: K

4. Random Forests
Parameter to be tuned: n_estimators

5. Naïve Bayes 
Parameters to be tuned: None

#IV.Model Parameter Tuning
Use cross-validation to tune the parameters of classifiers. Test the models trained with best obtained
parameter values on the separate testing set.
V. Report Requirements
• For all the requirements mentioned above you should report the model accuracy, precision, recall
and F- measure as well as the resultant confusion matrix using the testing data.

• Comment on all comparisons made.

• Provide screenshots of the main parts of the code with comments.


#VI. Bonus
Use Keras to build a neural network with dense layers and apply the model on your dataset. Use 2 layers and
tune the number of hidden units in every layer. You should provide the above report requirement also for the
bonus part.


#VII. Notes
• You should write your code in python
• You can use a third-party machine learning implementation like scikit-learn.
• You should work in a group of three.
• You must deliver a report showing all your work and conclusions.
• Only send your report, don’t send the code. There is no discussion.
