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

#Load solubility data
EC=bn.read_csv(input_file)
EC.drop(['Uniprot'], axis=1,inplace=True)
EC = EC[pd.notnull(EC['EC'])]
EC = EC[pd.notnull(EC['Sequence'])]
X_seq_EC=EC['Sequence']
Y_seq_EC=EC['EC']



#bn.visu_KDE(X_seq_EC,"seq length",path+'Density Plot of seqs length.svg')

# Tokenlize seq_EC

d2_to_index=bn.vis_seq_elements(X_seq_EC,path+'Frequencies Hist of Amino Acids EC.svg')
X_seq_EC=bn.tensor_pad(X_seq_EC,d2_to_index,max_length=200)
# Tokenize EC
Y_seq_EC_tokenized_index=bn.catagorite_EC_index(Y_seq_EC)
Y_seq_EC_tokenized=bn.catagorite_EC(Y_seq_EC)


#Build combined NN
print(str(chem['Output_type']),type(str(chem['Output_type'])))
Seq_NN=bn.create_network(layer_type=(seq['Type'],[X_seq_EC.shape[1]]+seq['Neuron']),outputlayer_type=seq['Output_type'],optimizer=seq['Optimizer'],Init=seq['InitializationMethod'],vocab=d2_to_index,drop_out=seq['Dropout_rate'])
Seq_NN.summary()
hist=bn.k_folds_NN(network=Seq_NN,X=[X_seq_EC],Y=Y_seq_EC_tokenized_index,batch_size=seq['Batch_size'], epochs=seq['Epoch'],path=output_path,Init=seq['InitializationMethod'],outputlayer_type=seq['Output_type'],vocab=d2_to_index)

