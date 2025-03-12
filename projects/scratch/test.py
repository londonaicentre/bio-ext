import pandas as pd
import generate_canon_biomarkers

# testing functionality as a module
df1,df2,df3,df4 = generate_canon_biomarkers.clean_and_generate_genes()

print(df3.head())