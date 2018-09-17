import Bioneural as bn
from keras.models import load_model
import csv
import os
import pandas as pd
import numpy as np

import importlib
importlib.reload(bn) 

# This index needs to be saved with the model
d_to_index = ['Y',
              'C',
              'A',
              'N',
              'E',
              'G',
              'K',
              'L',
              'S',
              'W',
              'T',
              'V',
              'M',
              'I',
              'Q',
              'P',
              'D',
              'H',
              'F',
              'R',
              'X']


testmol = [("pinocembrin", "InChI=1S/C15H12O4/c16-10-6-11(17)15-12(18)8-13(19-14(15)7-10)9-4-2-1-3-5-9/h1-7,13,16-17H,8H2/t13-/m0/s1")]
testmol = [("benzoate","InChI=1S/C7H6O2/c8-7(9)6-4-2-1-3-5-6/h1-5H,(H,8,9)/p-1")]
testmol = [("p-coumaric acid","InChI=1S/C9H8O3/c10-8-4-1-7(2-5-8)3-6-9(11)12/h1-6,10H,(H,11,12)/b6-3+")]
testmol = [("2,4,6-trinitrotoluene","InChI=1S/C7H5N3O6/c1-4-6(9(13)14)2-5(8(11)12)3-7(4)10(15)16/h2-3H,1H3")]
testmol = [("mevalonate","InChI=1S/C6H12O4/c1-6(10,2-3-7)4-5(8)9/h7,10H,2-4H2,1H3,(H,8,9)/p-1")]
testmol = [("acyl-homoserine lactone","InChI=1S/C5H7NO3/c7-3-6-4-1-2-9-5(4)8/h3-4H,1-2H2,(H,6,7)/t4-/m0/s1")]
testmol = [("(4R)-limonene", "InChI=1S/C10H16/c1-8(2)10-6-4-9(3)5-7-10/h4,10H,1,5-7H2,2-3H3/t10-/m0/s1")]
input_file = 'biosensor_seqs.csv'
sidx = {}
with open(input_file) as h:
    cv = csv.reader( h )
    head = next(cv)
    seql = set()
    seql1 = set()
    for row in cv:
        seql.add( tuple(row[3:]) )
        seql1.add( row[-1] )
        sidx[ row[-1] ] = row[3:]
chem = []
seq = []
for mol in testmol:
    chem.extend( list(np.repeat( mol[1], len(seql1) )) )
    seq.extend( list(sorted(seql1)) )
lab = np.repeat( '-1', len(seq) )
cvtest = pd.DataFrame({'chem': chem,'seq': seq ,'label': lab})

Test_seq,Test_chemical=bn.load_processed_biosensor_pablo(cvtest,d_to_index)

output_path = os.path.abspath('.')
modelfile = os.path.join( output_path, 'combined_model.h5' )
model = load_model( modelfile )
ytest = model.predict( [Test_seq] + [Test_chemical] )
h = 0
for i in sorted( np.arange(0, len(ytest)) , key = lambda x: -ytest[x] ):
    s = cvtest.iloc[i]['seq']
    print( ytest[i], sidx[s][0], sidx[s][1], sidx[s][2] )
    h += 1
    if h>10:
        break

