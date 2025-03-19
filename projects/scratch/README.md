# Canonical Biomarkers

## Brief

We want to identify all possible *biomarkers* currently known by using canonical data sources.
Then, we will want to identify where each *biomarkers* reside in our Elastic / Cog stack

## Data Sources

Source 1: [National Genomic Source Directory by UK NHS](https://www.genomicseducation.hee.nhs.uk/genotes/knowledge-hub/the-national-genomic-test-directory/)

Comments: Excel file 
Quality: Regularly updated

Source 2: [Association of Cancer Care Centres](https://www.accc-cancer.org/home/learn/precision-medicine/cancer-diagnostics/biomarkers/biomarkerlive/lexicon/cancer-biomarkers)

Comments: Web page
Quality: private 

Source 3: [USA NIH Cancer Tumour markers](https://www.cancer.gov/about-cancer/diagnosis-staging/diagnosis/tumor-markers-list)

Comments: Web page
Quality: potenitally high but not UK

Source 4: [USA NIH source](https://edrn.nci.nih.gov/data-and-resources/biomarkers/)
Comments: Web page
Quality: Currently down likely due to US Administration pulling NIH pages

## Query savings

Queries will be saved in `sde_aic_internal_docs/nlp/gstt_elastic_directory.md`

## Other solutsions

Other people have been attempting similar solution. See here by [Qgenomeapp](https://qgenome.co.uk/)


## What does the script do

`oncbiomarker.py` will essentially generate a unique strings of biomarkers.  It is a work in progress.
Future iterations will include paediatric tumours and haematological, currently only for solid tumours