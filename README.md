MAGIC-telescope-data
====================

A set of files examining the performance of different classifiers on the MAGIC
telescope simulated data.

Files: 

magic.py - computes and plots ROC curves for a variety of classifiers.

magic_nn.py - Trains a feedforward neural network on the MAGIC dataset using PyBrain.
Saves trained network to file magic_nn.xml.  Plots cost function vs training round.

magic2_nn.py - Similar to magic_nn.py, but uses different PyBrain commands.  Saves 
trained network to file magic2_nn.xml.  Plots training accuracy vs training round.

