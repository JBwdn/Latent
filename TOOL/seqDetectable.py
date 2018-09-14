# import libraries
import numpy as np
from Bio import SeqIO
from Bio.PDB.Polypeptide import d1_to_index
from rdkit.Chem import AllChem
from __future__ import print_function
import pandas as pd
import csv

%load_ext ipython_bell

from Bio import Entrez
import re

def download_info (patterm,database="Protein"):
    '''download information from relervant database'''
 
    handle = Entrez.esearch(db=database, term=patterm, retmax=500)
    record = Entrez.read(handle)
    handle.close()
 
    return record

def download_seq (id_array):
    '''HAVE GeneID to download seqs'''
 
    result_handle = Entrez.efetch(db="nucleotide", rettype="fasta",  id=id_array,retmode="text")
    result = result_handle.read()
 
    return result

""" Screen for biosensor sequences """
# List of detectable chemicals from https://github.com/brsynth/detectable_metabolites
file_info = 'detectable_metabolites_current.csv'
Entrez.email = 'dr413677671@gmail.com'
#with open(file_info) as header_input, open(file_output, 'w') as header_output:
#    cw = csv.writer(header_output)
#    cw.writerow( ('Name', 'Inchi', 'Gene', 'Organism', 'NCBI_Gene', 'Name', 'Description', 'Comment', 'Annotation') )
#    cv = csv.DictReader( filter( lambda row: row[0] != '#', header_input ) )
cv=pd.read_csv(file_info,header=0,sep=',')
#    counter=0
record=[]

# eliminate unuseful columns
cv.drop(['Reference', 'TypeOfExperiment','Comments','TypeOfDetection'], axis=1,inplace=True)
# Change columns names
cv.rename(columns={'DetectedBy':'Sensor'}, inplace = True)
cv = cv[pd.notnull(cv['Sensor'])]
cv = cv[pd.notnull(cv['InChI'])]

cv['info']=None
cv['Seq'] = None

%%bell -n say
import os


path='F:/'
if not os.path.exists(path):
        os.mkdir(path)

#Slect the rows you would like to crawl data by
for i in range(600,1390,1):
    tmp=cv[i:(i+1)]
   
    for index, row in tmp.iterrows():
        print(i)
        name = str(row['Name'])
        inchi = row['InChI']
        organism = str(row['Organism'])
        sensor = str(row['Sensor'])
        # Query based on gene name
        # TO DO: curate generic names (riboswitch, TF, etc.)
        if len(sensor) > 0:
            # Query NCBI by gene name
            term=""
            term=term+ sensor+' [GN]' 
            if organism!= 'nan':
                term=term+ ' AND ' + organism+' [ORGN]' 
            try:
                record=download_info(term,database="Protein") 
            except:
                continue
            # Fetch the full record or just the summary (should be enough)
            # Things to do:
            # - Select the most convenient format (xml, etc.)
            # - Query by name to avoid false hits
            # - Use Description to double check transcriptional activity
            # - Double-check organims (better through postprocessing)
            if  record['IdList']:
                id_array =download_seq( record['IdList'][0] )
#                 id_info.append(id_array)
                try:
                    infotmp,seqtmp=np.squeeze(re.findall(r'>(.*)\n([\s\S]*$)',id_array))
                    cv.loc[index:index, 'info']=infotmp.encode('utf-8').strip()
                    cv.loc[index:index, 'Seq']=re.sub("\n","",seqtmp)
                except:
                    continue
cv = cv[pd.notnull(cv['Seq'])]
# write csv

file_output = path + 'detectable_metabolites_seq2.csv'
cv.to_csv(file_output,index=True,sep=',')
