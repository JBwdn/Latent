#Enable the bell ring after trainingin IPython
#%load_ext ipython_bell
#Get input_file and output_path from shell para
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
chem_data = bn.load_chemicals("Solubility",input_file)
chem_data.rename(columns={ chem_data.columns[1]: "Solubility"}, inplace=True)
X_chem=chem_data['SMILES']

Y_chem=np.array(chem_data['Solubility'])

bn.visu_KDE(Y_chem,"Chemicals Solubility",output_path+'Density Plot of chem solubility.svg')
X_chem=bn.flatten_chem(X_chem)

#Build combined NN
print(str(chem['Output_type']),type(str(chem['Output_type'])))
Chem_NN=bn.create_network(layer_type=(chem['Type'],[X_chem.shape[1]]+chem['Neuron']),outputlayer_type=str(chem['Output_type']),optimizer=chem['Optimizer'],Init=chem['InitializationMethod'])
Chem_NN.summary()
if chem['Output_type']=='binary_classifier':
    Y_chem_hardmax=bn.hardmax(Y_chem)
    history=bn.k_folds_NN(network=Chem_NN,X=[X_chem],Y=Y_chem_hardmax,batch_size=chem['Batch_size'], epochs=chem['Epoch'],path=output_path,Init=chem['InitializationMethod'],outputlayer_type=str(chem['Output_type']),optimizer_type=chem['Optimizer'])
else:
    history=bn.k_folds_NN(network=Chem_NN,X=[X_chem],Y=Y_chem,batch_size=chem['Batch_size'], epochs=chem['Epoch'],path=output_path,Init=chem['InitializationMethod'],outputlayer_type=str(chem['Output_type']),optimizer_type=chem['Optimizer'])

