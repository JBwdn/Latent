#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 11:41:51 2018

synbioTools (c) University of Manchester 2015
synbioTools is licensed under the MIT License.
To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author: Pablo Carbonell
@description: Collection of tools to encode common synbio data 
    into useful information for creating training sets.
    Some of the possible input objects:
        - Sequence (amino acids, nucleotides)
        - Chemicals
        - Chemical reaction
        - Metabolic model, pathway
        - Experimental conditions
        - Experimental design
"""
import numpy as np
from Bio import SeqIO
from Bio.PDB.Polypeptide import d1_to_index
from rdkit.Chem import AllChem

""" Sequence encoding """

def aaindex(seq):
    """ Convert amino acid to numerical index """
    ix = []
    for a in seq:
        if a in d1_to_index:
            ix.append( d1_to_index[a] )
    return ix

def readfasta(ffile):
    """ Read fast file, return dictionary """
    record_dict = SeqIO.to_dict(SeqIO.parse(ffile, "fasta"))
    return record_dict

def tensorSeq(seqs, MAX_SEQ_LENGTH, SEQDEPTH, TOKEN_SIZE=20):
    """ Encode an amino acid sequence as a tensor 
    by concatenating one-hot encoding up to desired depth """
    TRAIN_BATCH_SIZE = len(seqs)
    Xs = np.zeros( (TRAIN_BATCH_SIZE, MAX_SEQ_LENGTH, SEQDEPTH*TOKEN_SIZE) )
    for i in range(0, len(seqs)):
        for j in range(0, len(seqs[i])):
            aaix = aaindex( seqs[i][j]  )
            for l in range(0, len(aaix)):
                for k in range(0, SEQDEPTH):
                    try:
                        Xs[i, l, aaix[l+k] + TOKEN_SIZE*k] = 1
                    except:
                        continue
    """ Flip sequences (zero-padding at the start) """
    Xsr = np.flip( Xs, 1 )
    return Xsr

""" Chemical encoding """

def chemFP(chem, FINGERPRINT_SIZE, MIN_PATH=1, MAX_PATH=5):
    fpix = AllChem.RDKFingerprint(chem, minPath=MIN_PATH, maxPath=MAX_PATH, fpSize=FINGERPRINT_SIZE)    
    fpix = [int(x) for x in list(fpix.ToBitString())]
    return fpix

def tensorChem(chems, FINGERPRINT_SIZE, CHEMDEPTH, TOKEN_SIZE=1, MIN_PATH=1, MAX_PATH=5):
    """ Encode a chemical as a tensor by concatenating fingerprints
    up to desired depth """
    TRAIN_BATCH_SIZE = len(chems)   
    Xs = np.zeros( (TRAIN_BATCH_SIZE, FINGERPRINT_SIZE, CHEMDEPTH*TOKEN_SIZE) )
    for i in range(0, len(chems)):
        fpix = chemFP(chems[i])
        for l in range(0, len(fpix)):
            for k in range(0, CHEMDEPTH):
                try:
                    Xs[i, l, fpix[l+k] + TOKEN_SIZE*k] = 1
                except:
                    continue
    return Xs

""" Chemical reaction encoding """

def reacFP(reac, FINGERPRINT_SIZE, MIN_PATH=1, MAX_PATH=5):
    """ Reaction fingerprint """
    left, right = reacs[i]
    left = [chemFP(m, MIN_PATH, MAX_PATH) for m in left]
    right = [chemFP(m, MIN_PATH, MAX_PATH) for m in right]
    lfp = left[0]
    for m in left:
        lfp = lfp | m
    rfp = right[0]
    for m in right:
        rfp = rfp | m
    rfp = lfp ^ rfp
    return rfp

def tensorReac(reacs, FINGERPRINT_SIZE, CHEMDEPTH, TOKEN_SIZE=1,  MIN_PATH=1, MAX_PATH=5):
    TRAIN_BATCH_SIZE = len(reacs)   
    Xs = np.zeros( (TRAIN_BATCH_SIZE, FINGERPRINT_SIZE, CHEMDEPTH*TOKEN_SIZE) )
    for i in range(0, len(reacs)):
         fpix = reacFP(reacs[i], MIN_PATH, MAX_PATH)
         for l in range(0, len(fpix)):
            for k in range(0, CHEMDEPTH):
                try:
                    Xs[i, l, fpix[l+k] + TOKEN_SIZE*k] = 1
                except:
                    continue
    return Xs
    
    