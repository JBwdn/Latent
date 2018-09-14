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
bn.visu_KDE(Train_seq,"seq length",path+'Density Plot of seqs length.svg')
path=bn.set_path(str(output_path))
cv=bn.load_biosensor(input_file,output_path)
#Load processed biosensor data
Train_seq,Train_chemical,Label,d_to_index=bn.load_processed_biosensor(cv,input_file,output_path)
#Build combined NN
Seq_NN=bn.create_network(layer_type=(seq['Type'],[Train_seq.shape[1]]+seq['Neuron']),outputlayer_type=seq['Output_type'],optimizer=seq['Optimizer'],Init=seq['InitializationMethod'],vocab=d_to_index,drop_out=seq['Dropout_rate'])
Chem_NN=bn.create_network(layer_type=(chem['Type'],[Train_chemical.shape[1]]+chem['Neuron']),outputlayer_type=chem['Output_type'],optimizer=chem['Optimizer'],Init=chem['InitializationMethod'])
combine_model = bn.combine_models(Seq_NN,Train_seq.shape[1],Chem_NN,Train_chemical.shape[1],combined['Neuron'])
combine_model.summary()
#Train models
#Enable the bell sound in Ipython
#%%bell -n say
combine_model.compile(loss='binary_crossentropy',optimizer= 'Adam',metrics=['accuracy'])
c_m=combine_model.fit([Train_seq]+[Train_chemical],Label,batch_size=combined['Batch_size'], epochs=combined['Epoch'],validation_split=0.3)
bn.training_vis(c_m,"binary_classifier",output_path+"train_dist.svg")
combine_model.save(output_path+'combined_model.h5')
