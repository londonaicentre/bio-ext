import pandas as pd
import oncbiomarker

# testing functionality as a module
df1,df2,df3,df4 = oncbiomarker.clean_and_generate_genes()

print(df3.head())