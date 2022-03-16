#!/usr/bin/env python

import gzip
from lxml import etree as ET
from sys import argv

class Dummy():
    def __init__(self):
        self.attrib = {'id': ''}
        self.text = ''
safe = lambda x: Dummy() if x is None else x

with gzip.open(argv[1], 'r') as f:
    context = ET.iterparse(f, tag='entry')
    for event, entry in context:
        name = safe(entry.find('name')).text
        ensg = safe(entry.find('identifier')).attrib['id']
        uniprot = safe(entry.find('identifier/xref')).attrib['id']
        for ab in entry.findall('antibody'):
            ab_name = safe(ab).attrib['id']
            for data in ab.findall('tissueExpression/data'):
                tissue = safe(data.find('tissue')).text.lower()
                for tissue_cell in data.findall('tissueCell'):
                    cell_type = safe(tissue_cell.find('cellType')).text.lower()
                    staining = safe(tissue_cell.find('level[@type="staining"]')).text.lower()
                    for patient in data.findall('patient'):
                        sex = safe(patient.find('sex')).text
                        age = safe(patient.find('age')).text
                        patient_id = safe(patient.find('patientId')).text
                        image_url = safe(patient.find('sample/assayImage/image/imageUrl')).text
                        row = (name,ensg,uniprot,ab_name,tissue,cell_type,staining,sex,age,patient_id,image_url)
                        print('\t'.join(row))
        entry.clear()