1) nets_to_compurte_smoothness.ipynb notebook contains code for training the networks as well as learning curves and performance on the test data. It contains all the nets from the presentation and some experiments with architecture. It is run on Google colab GPU.

2) extract_CI_ResNet12.ipynb is a code to extract representation for each layer of ResNet12. There are similar ones for Plain and tanh models. (not in the repo) It saves on my google drive cloud, then i download it to the 'data' folder. It is run on Google colab GPU.

3) random_forest.py is a code of Kobi Gurcan. I changed only the evaluate_smoothness function so it will also return MS errors. It is used in run_smoothness.py.

4) run_smoothness.py calculates alpha smoothness of each layer of required network and pickles it into a folder 'results'.  It is run on my PC.

5) mislabeling.py performs mislabeling of the labels and calculates the besov smoothness for the last layer, it saves the result to the 'results' folder.  It is run on my PC.

6) Visualisation_of_results.ipynb - After I load my results on my google drive cloud and visualize what i've got here. - Google colab notebook.

7) final_project_ML_presentation_Anna_Shchukina.pdf - final presentation.
