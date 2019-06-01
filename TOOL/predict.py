import Bioneural as bn
import importlib
importlib.reload(bn) 
"""Read arguments"""
import sys,getopt
import re
from keras.models import load_model
import numpy as np
import pickle
def read_info():
    data=[]
    for line in open("Config.txt","r"): 
        data.append(line)
    for i in data:
        if re.match('MolecularStructuresNeuralNetworks:',i):
            info=np.squeeze(re.findall('MolecularStructuresNeuralNetworks:(\w*),\[(.*)\],(.*),(.*),(.*),(.*),(.*)',i))
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
    opts,args = getopt.getopt(sys.argv[1:], "i:t:o:")
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

if __name__ == '__main__':
    input_file,output_path,input_file_solubility,input_file_thermostability=get_para()
    chem,seq,combined=read_info()
    #bn.visu_KDE(Train_seq,"seq length",path+'Density Plot of seqs length.svg')
    path=bn.set_path(str(output_path))
    cv=bn.load_biosensor_not_depli(input_file,output_path)
    #Load processed biosensor data
    Train_seq,Train_chemical,Label,d_to_index=bn.load_processed_biosensor(cv,input_file,output_path)
    
    print("Loading  model & predicting...")
    load_model = load_model(input_file_thermostability)
    predicted = load_model.predict([Train_seq]+[Train_chemical])
    
    print("\nPredicted posibility is: ")
    print(predicted)
