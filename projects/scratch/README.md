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


## What does the scripts do

* `oncbiomarker.py` will ingest 3 data sources 1,2 and 3.
* source 4 is now down and not available.
* It can be called as a python module and `test.py` file tests this.
* It can be called as a script from command line in which case the outputs will be saved to a directory `data/output` at the moment. 
* NOTE: Future feature to give user added functionality about saving location and naming the files. 

* The files are one for each data source with a bit more columns unique to each source. These are useful for downstream NLP tasks
* Then one final `canonical_combined_unique_genes.csv` which includes all the 3 sources concatenated.
