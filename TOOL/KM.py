#Enable the bell ring after trainingin IPython
#%load_ext ipython_bell
#Get input_file and output_path from shell para
import pandas as pd
import Bioneural as bn
"""Read arguments"""
import sys,getopt
import re
import numpy as np
def read_info():
    data=[]
    for line in open("Config.txt","r"): 
        data.append(line)
    for i in data:
        if re.match('MolecularStructuresNeuralNetworks:',i):
            info=np.squeeze(re.findall('MolecularStructuresNeuralNetworks:(\w*),\[(.*)\],(.*),(.*),(.*),(\d*),(\d*)',i))
            number=re.split(r',',info[1])
            number=[int(i) for i in number]
            chem = {'Type' : info[0], 'Neuron' : number,'Optimizer' : info[2], 'InitializationMethod' : info[3],'Output_type':info[4] ,'Batch_size':int(info[5]),'Epoch':int(info[6])}
        elif re.match('AminoAcidSequencesNeuralNetworks:',i):
            info=np.squeeze(re.findall('AminoAcidSequencesNeuralNetworks:(\w*),\[(.*)\],(.*),(.*),(.*),(.*),(.*),(.*)',i))
            number=re.split(r',',info[1])
            number=[int(i) for i in number]
            seq = {'Type' : info[0], 'Neuron' : number,'Optimizer' : info[2], 'InitializationMethod' : info[3],'Dropout_rate':float(info[4]),'Output_type':info[5],'Batch_size':int(info[6]),'Epoch':int(info[7])}
        elif re.match('CombinedNeuralNetworks:',i):
            
            info=np.squeeze(re.findall('CombinedNeuralNetworks:\[(.*)\],(.*),(.*),(.*),(.*)',i))
            number=re.split(r',',info[0])
            number=[int(i) for i in number]
            combined = { 'Neuron' :number,'Optimizer' : info[1], 'InitializationMethod' : info[2],'Batch_size':int(info[3]),'Epoch':int(info[4])}
    return chem,seq,combined
def get_para():
    opts,args = getopt.getopt(sys.argv[1:], "i:o:")
    input_file=""
    output_file=""
    input_file_solubility=""
    input_file_thermostability=""
    for op, value in opts:
        if op == "-i":
            input_file = value
        elif op == "-s":
            input_file_solubility = value
        elif op == "-t":
            input_file_thermostability = value
        elif op == "-o":
            output_file = value
            #path=set_path(output_file)
        elif op == "-h":
            usage()
            sys.exit()
    return input_file,output_file,input_file_solubility,input_file_thermostability
"""Main"""
input_file,output_path,input_file_solubility,input_file_thermostability=get_para()
chem,seq,combined=read_info()
path=bn.set_path(str(output_path))
"""Load Data"""
kmseq = bn.load_chemicals("Solubility", input_file)
kmseq.drop(['Uniprot','Compound','Reaction'], axis=1,inplace=True)
kmseq = kmseq[pd.notnull(kmseq['Sequence'])]
kmseq = kmseq[pd.notnull(kmseq['SMILES'])]
kmseq = kmseq[pd.notnull(kmseq['Km'])]
kmseq=kmseq.sample(frac=1)
kmseq.reset_index(drop=True, inplace=True)
X_chem_kmseq=kmseq['SMILES']
X_seq_kmseq=kmseq['Sequence']
Y_kmseq=kmseq['Km']
#bn.visu_KDE(Y_kmseq,"Km",path+'Density Plot of km.svg')
#bn.visu_KDE(X_seq_kmseq,"seq length",path+'Density Plot of seqs length.svg')

"""Tokenize data"""
# Generate vocab
d4_to_index=bn.vis_seq_elements(X_seq_kmseq,path+'Frequencies Hist of Amino Acids Thermostability.svg')
print(X_chem_kmseq,X_seq_kmseq,d4_to_index,Y_kmseq)
# Tokenlize chem
X_chem_kmseq=bn.flatten_chem(X_chem_kmseq)
# Tokenlize seq_thermostability
X_seq_kmseq=bn.tensor_pad(X_seq_kmseq,d4_to_index)
# Hardmax thermostability
Y_kcat_hardmax=bn.hardmax(Y_kmseq)

#Build combined NN
"""Build NN"""
# Build two sequential NN
Seq_NN=bn.create_network(layer_type=("LSTM",[X_seq_kmseq.shape[1],32,32,0]),outputlayer_type='linear_regression',optimizer='Adam',Init='he_init',vocab=d4_to_index,drop_out=1)
Chem_NN=bn.create_network(layer_type=("DFF",[X_chem_kmseq.shape[1],32,32,0]),outputlayer_type='linear_regression',optimizer='Adam',Init='he_init')
combine_model = bn.combine_models(Seq_NN,X_seq_kmseq.shape[1],Chem_NN,X_chem_kmseq.shape[1],combined['Neuron'])
combine_model.summary()

combine_model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])
c_m=combine_model.fit([X_seq_kmseq]+[X_chem_kmseq],Y_kmseq,batch_size=combined['Batch_size'], epochs=combined['Epoch'],validation_split=0.3)
bn.training_vis(c_m,'multiple_classifier',output_path+"train_dist.svg")
combine_model.save(output_path+'combined_model.h5')
