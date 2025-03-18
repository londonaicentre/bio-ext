import pandas as pd
import generate_canon_biomarkers

# testing functionality as a module
dfs = generate_canon_biomarkers.clean_and_generate_genes()

# load each dataframe and print them
for i in dfs:
    print(i)
    print(i.head())
