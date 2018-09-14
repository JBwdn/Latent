#Enable the bell ring after trainingin IPython
#%load_ext ipython_bell
#Get input_file and output_path from shell para
import Bioneural as bn
import pandas as pd
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
    opts,args = getopt.getopt(sys.argv[1:], "i:o:t:s:")
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

# Load seqs_thermoatability
non_thermophilic_proteins = bn.readfasta(input_file_thermostability)
thermophilic_proteins = bn.readfasta(input_file)
thermophilic_proteins['Thermostability']='1'
non_thermophilic_proteins['Thermostability']='0'
# Append the thermophilic to the non-thermophilic
sequence_data=pd.concat([thermophilic_proteins,non_thermophilic_proteins])
sequence_data=sequence_data.sample(frac=1)
# Init input and Y
X_seq=sequence_data['seq']
Y_seq=sequence_data['Thermostability']
bn.visu_KDE(X_seq,"seq length",output_path+'Density Plot of seqs length.svg')
#Tokenization
d1_to_index=bn.vis_seq_elements(X_seq,path+'Frequencies Hist of Amino Acids Thermostability.svg')
# Tokenlize seq_thermostability
X_seq=bn.tensor_pad(X_seq,d1_to_index)
# Hardmax thermostability
Y_seq_hardmax=bn.hardmax(Y_seq)

#Build combined NN
# Build two sequential NN
Seq_NN=bn.create_network(layer_type=(seq['Type'],[X_seq.shape[1]]+seq['Neuron']),outputlayer_type=str(seq['Output_type']),optimizer=seq['Optimizer'],Init=seq['InitializationMethod'],vocab=d1_to_index,drop_out=seq['Dropout_rate'])
Seq_NN.summary()
hist=bn.k_folds_NN(network=Seq_NN,X=[X_seq],Y=Y_seq_hardmax,batch_size=seq['Batch_size'], epochs=seq['Epoch'],path=output_path,Init=seq['InitializationMethod'],outputlayer_type=str(seq['Output_type']))


