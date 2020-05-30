#########################################
PCAM ZC321-C1-REPORT-Loan Processing-G11
#########################################

Instructions for testing the trained models

Step 1: We have trained all the models on the train.csv shared by Prof. Satyaki and saved those models in form of binaries in trained-models directory.
Since a few models are trained using h2o AutoML please install h2o using either of the below comands depending upon the python distribution installed on your machine.

For Anaconda Distributions: 
conda install -c h2oai h2o	
	OR
For Python: 
pip install -f http://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o

The models binaries that are trained require h2o version 3.30.0.3 otherwise you will not be able to load them on h2o server. Please verify installed version before proceeding.

Step 2 
