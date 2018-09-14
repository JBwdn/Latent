""" Screen for biosensor sequences """
from __future__ import print_function
import csv
from Bio import Entrez
import re

# List of detectable chemicals from https://github.com/brsynth/detectable_metabolites
f = 'detectable_metabolites_current.csv'
fw = 'detectable_metabolites_seq.csv'
Entrez.email = 'lab@lab.org'

with open(f) as h, open(fw, 'w') as h2:
    cw = csv.writer(h2)
    cw.writerow( ('Name', 'Inchi', 'Gene', 'Organism', 'NCBI_Gene', 'Name', 'Description', 'Comment', 'Annotation') )
    cv = csv.DictReader( filter( lambda row: row[0] != '#', h ) )
    for row in cv:
        name = row['Name']
        inchi = row['InChI']
        organism = row['Organism']
        sensor = row['DetectedBy']
        # Query based on gene name
        # TO DO: curate generic names (riboswitch, TF, etc.)
        if len(sensor) > 0:
            # Query NCBI by gene name
            term = []
            term.append( sensor+'[GN]' )
            term.append( 'AND' )
            term.append( organism+'[ORGN]' )
            term = ' '.join(term)
            try:
                handle = Entrez.esearch(db='Gene', term=term, retmax=500)
                record = Entrez.read(handle)
                handle.close()
            except:
                continue
            # Fetch the full record or just the summary (should be enough)
            # Things to do:
            # - Select the most convenient format (xml, etc.)
            # - Query by name to avoid false hits
            # - Use Description to double check transcriptional activity
            # - Double-check organims (better through postprocessing)
            for rid in record['IdList']:
                try:
                    h1 = Entrez.efetch(db='Gene', id=rid, retmode='text')
                except:
                    continue
#                h2 = Entrez.esummary(db='Gene', id=rid, retmode='text')
#                r2 = [x for x in h2]
                r1 = [x.rstrip() for x in h1] 
                if re.findall(sensor, r1[2]):
                    try:
                        h3 = Entrez.efetch(db='sequences', id=rid, rettype='fasta')
                    except:
                        continue
                    out = ( name, inchi, sensor, organism, rid, r1[1], r1[2], r1[5], r1[4] )
                    print(out)
                    cw.writerow( out )
