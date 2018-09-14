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
    opts,args = getopt.getopt(sys.argv[1:], "i:o:s:t:")
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

path=bn.set_path(output_path)

"""Load Data"""
cv=bn.load_biosensor(input_file,output_path)
#Load processed biosensor data
Train_seq,Train_chemical,Label,d_to_index=bn.load_processed_biosensor(cv,input_file,output_path)
X_chem,Y_chem_hardmax=bn.load_solubility(input_file_solubility,output_path)
bn.visu_KDE(Train_seq,"seq length",path+'Density Plot of seqs length.svg')
#Build combined NN
Seq_NN=bn.create_network(layer_type=(seq['Type'],[Train_seq.shape[1]]+seq['Neuron']),outputlayer_type=seq['Output_type'],optimizer=seq['Optimizer'],Init=seq['InitializationMethod'],vocab=d_to_index,drop_out=seq['Dropout_rate'])
Chem_NN=bn.create_network(layer_type=(chem['Type'],[X_chem.shape[1]]+chem['Neuron']),outputlayer_type=chem['Output_type'],optimizer=chem['Optimizer'],Init=chem['InitializationMethod'])
print(chem['Output_type'],chem['Optimizer'])

"""Output_mode"""
if chem['Output_type']=='linear_regression':
    if chem['Optimizer'] == 'bgd':
        Chem_NN.compile(loss='mean_squared_error',optimizer= chem['Optimizer'],metrics=[coeff_determination]) # Accuracy performance metric-R2 sgd
    elif chem['Optimizer'] == 'Adam':
        Chem_NN.compile(loss='mean_squared_error',optimizer= chem['Optimizer'],metrics=[coeff_determination]) # Accuracy performance metric-R2 Adam
elif chem['Output_type']=='binary_classifier':
    if chem['Optimizer'] == 'sgd':
        Chem_NN.compile(loss='binary_crossentropy',optimizer= 'sgd',metrics=['accuracy']) # Accuracy performance metric sgd
    elif chem['Optimizer'] == 'Adam':
        Chem_NN.compile(loss='binary_crossentropy',optimizer= 'Adam',metrics=['accuracy']) # Accuracy performance metric Adam
elif chem['Output_type']=='multiple_classifier':
    if chem['Optimizer'] == 'bgd':
        Chem_NN.compile(loss='categorical_crossentropy',optimizer= 'sgd',metrics=['accuracy']) # Accuracy performance metric-R2 sgd
    elif chem['Optimizer'] == 'Adam':
        Chem_NN.compile(loss='categorical_crossentropy',optimizer= 'Adam',metrics=['accuracy']) # Accuracy performance metric-R2 Adam
    
Chem_NN.fit([X_chem],Y_chem_hardmax,batch_size=chem['Batch_size'], epochs=chem['Epoch'])
# Here we need wo exclude the outputlayer manually
Chem_NN_new=bn.Model(inputs=Chem_NN.input,outputs=Chem_NN.get_layer('dense_77').output)
combine_model = bn.combine_models(Seq_NN,Train_seq.shape[1],Chem_NN_new,Train_chemical.shape[1],archi=combined['Neuron'])
combine_model.summary()
#Train models
#Enable the bell sound in Ipython
#%%bell -n say
combine_model.compile(loss='binary_crossentropy',optimizer= 'Adam',metrics=['accuracy'])
c_m=combine_model.fit([Train_seq]+[Train_chemical],Label,batch_size=combined['Batch_size'], epochs=combined['Epoch'],validation_split=0.3)
bn.training_vis(c_m,"binary_classifier",output_path+"train_dist.svg")
combine_model.save(output_path+'combined_model.h5')