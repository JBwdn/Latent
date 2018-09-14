# A combanation method of moderated t-test and radom forest algorithm in micro-array

The pipeline consists of 5 main functions: biosensor prediction, biosensor prediction with tranfer learning (based on solubility), solubility prediction and thermostability prediction

Introduction
-------------

The tool is developed to support training neural networks on different synthetic biology datasets, including biosensor datasets, solubility datasets and thermostability datasets. 
The training is based on Keras.

Here are the scripts of the tool:
Bioneural.py - The main package to import
Biosensor_NN.py - Biosesor prediction without Transfer Learning
Biosensor_NN_Transfer_Learning.py - Biosesor prediction with Transfer Learning
EC_number.py - EC Number prediction
KM.py - Km Prediction
seqDetectable.py - Crawling sequences
Solubility.py - Solubility prediction
Thermostability.py - Thermostability prediction




Compatibility
-------------

The pipeline should be run on Python>=3 (it was tested on 3.6.5).

As of Python >= 3.6.5, the script require some python standard library. For users who still need to support several packages are required: 
IPython.display, numpy, scipy.stats, pandas, time, keras, sklearn, matplotlib, csv, Biopython, rdkit


Configration
------------

A config file (in the main root of the script named Config.txt) can be customized, the format can be found as follows:

DATATYPE:LAYER_TYPE,NEURONS,OPTIMIZER,INIT,DROPOUT_RATE,OUTPUT_MODE,BATCHSIZE,EPOCH

%%%%%%%%%%%%Examples Configs%%%:

ECnumber Prediction:

MolecularStructuresNeuralNetworks:DFF,[0],-,-,-,0,0
AminoAcidSequencesNeuralNetworks:LSTM,[32,32,502],Adam,he_init,0.5,multiple_classifier,64,20
CombinedNeuralNetworks:[0],-,-,0,0

Biosensor Pridiction:

MolecularStructuresNeuralNetworks:DFF,[32,0],Adam,he_init,-,0,0
AminoAcidSequencesNeuralNetworks:LSTM,[32,32,0],Adam,he_init,0.5,-,0,0
CombinedNeuralNetworks:[128,128,128,1],Adam,he_init,128,30

Biosensor Pridiction with Tranfer Learning:

MolecularStructuresNeuralNetworks:DFF,[32,1],Adam,he_init,binary_classifier,128,20
AminoAcidSequencesNeuralNetworks:LSTM,[32,32,0],Adam,he_init,0.5,-,0,0
CombinedNeuralNetworks:[128,128,128,1],Adam,he_init,128,30

Thermostability Prediction:
MolecularStructuresNeuralNetworks:DFF,[0],-,-,-,0,0
AminoAcidSequencesNeuralNetworks:LSTM,[32,32,1],Adam,he_init,0.5,binary_classifier,6,200
CombinedNeuralNetworks:[0],-,-,0,0

KM Prediction:
MolecularStructuresNeuralNetworks:DFF,[32,32,0],Adam,he_init,-,0,0
AminoAcidSequencesNeuralNetworks:LSTM,[32,32,0],Adam,he_init,0.5,-,0,0
CombinedNeuralNetworks:[128,128,128,1],Adam,he_init,68,30

Solubility Prediction Linear Regression:
MolecularStructuresNeuralNetworks:DFF,[20,7,5,1],Adam,he_init,linear_regression,1144,107
AminoAcidSequencesNeuralNetworks:LSTM,[0],-,-,0,-,0,0
CombinedNeuralNetworks:[0],-,-,0,0

Solubility Prediction Binary Clssifier:
MolecularStructuresNeuralNetworks:DFF,[20,7,5,1],Adam,he_init,binary_classifier,1144,107
AminoAcidSequencesNeuralNetworks:LSTM,[0],-,-,0,-,0,0
CombinedNeuralNetworks:[0],-,-,0,0

Instructions
------------

1. Change the input_filename in seqDetectable.py and run it to crawl sequences.
2. Set up the Config.txt. The example config can be found in the Configration column in this document.
3. Run Biosensor_NN.py (or Biosensor_NN_Transfer_Learning.py) to train an artificial neural network. The example shell commend can be found in the example column in this document.
4. Retrieved outputs (the introduction of the outputs can be found in the outputs column in this document) from the customized output_path.

Examples
------------

-i path of the input file
-s path of the input solubility data
-t path of the suplemental input thermostability data
-o absolute output path


%%Biosensor prediction:
python Biosensor_NN.py -i biosensor_seqs.csv -o F:/result/

%%Biosensor prediction with Tranfer Learning:
python Biosensor_NN_Transfer_Learning.py -i biosensor_seqs.csv -o F:/result/ -s delaney.csv

%%Thermostability Prediction:
python Thermostability.py -i non-thermophilic_proteins.txt -t thermophilic_proteins.txt -o F:/result/

%%KM Prediction:
python KM.py -i kmseq.csv  -o F:/result/

%%EC Number:
python EC_number.py -i ecseq.csv -o F:/result/

%%Solubility Prediction:
python Solubility.py -i delaney.csv -o F:/result/

Outputs
----

%%Solubility Prediction:

.h5 : The trained artificial neural network in each fold.
Density Plot of chem solubility.svg : The density distribution of chemicals solubility properties
model_info.csv : The information of hyperparameters
linear_regression.svg : The linear regression on the predicted output and the labels
train_fold.svg : The training accuracy and loss value distribution
training_result.csv : The summary information of training result

%%Thermostability Prediction:

.h5 : The trained artificial neural network in each fold.
Density Plot of seqs length.svg : The density distribution of sequences lengths
model_info.csv : The information of hyperparameters
Frequencies Hist of Amino Acids Thermostability.svg : The information of amino acid count (Enzymes)
train_fold.svg : The training accuracy and loss value distribution
training_result.csv : The summary information of training result

%%Biosensor Prediction:

.h5 : The trained artificial neural network in each fold.
Density Plot of chem solubility.svg : The density distribution of chemicals solubility properties
Frequencies Hist of Amino Acids Thermostability.svg : The information of amino acid count (Biosensors)
train_dist.svg : The training accuracy and loss value distribution

%%EC Prediction:
.h5 : The trained artificial neural network in each fold.
Density Plot of seqs length.svg : The density distribution of sequences lengths
model_info.csv : The information of hyperparameters
Frequencies Hist of Amino Acids EC.svg : The information of amino acid count (Enzymes)
train_fold.svg : The training accuracy and loss value distribution
training_result.csv : The summary information of training result

%%Km Prediction:
Density Plot of Km.svg : The density distribution of Km properties
.h5 : The trained artificial neural network in each fold.
Density Plot of seqs length.svg : The density distribution of sequences lengths
linear_regression.svg : The linear regression on the predicted output and the labels
Frequencies Hist of Amino Acids Thermostability.svg : The information of amino acid count (Enzymes)
train_fold.svg : The training accuracy and loss value distribution
                
Bugs
----

If you find a bug, please try to reproduce it with Python 3.6.5 and latest packages.

If it does not happen in 3.6.5, please file a bug in the python issue tracker.
If it happens there also, file a bug to 413677671@qq.com.


Ran Duan
Email: 413677671@qq.com