import pandas as pd
from collections import Counter
import json
import math
import numpy as np
import re

import requests
from bs4 import BeautifulSoup

sheets_dict = pd.read_excel(
    "data/cancer-national-genomic-test-drectory-V11-January-2025.xlsx",
    engine="openpyxl",
    sheet_name=None,
)

solid = sheets_dict["Solid tumours"]
solid = solid.drop(index=solid.index[0], axis=0)

solid.columns = [
    "group",
    "ci_code",
    "clinical_indication_name",
    "test_code",
    "test_name",
    "target_genes",
    "target_gene_clinicaltrials",
    "test_scope",
    "technology",
    "optimal_family_structure",
    "further_eligibility",
    "further_eligibility_criteria",
    "changes_since_v1.0",
]

solid[["group", "ci_code", "clinical_indication_name"]] = solid[
    ["group", "ci_code", "clinical_indication_name"]
].ffill()

solid["target_genes"] = solid["target_genes"].replace({np.nan: None})

genes = list(solid["target_genes"])

genes = [gene for gene in genes if gene != "All including burden / signature"]

genes = [gene for gene in genes if gene is not None]
# this code wont work without removing nan's
genes = [item for sublist in [x.split(",") for x in genes] for item in sublist]


genes = [re.sub(r"^\s+|\s+$", "", gene) for gene in genes]
len(genes)

genes = ["Chromosome 1p" if gene == "1p" else gene for gene in genes]
genes = ["Chromosome 3" if gene == "3" else gene for gene in genes]
genes = ["Chromosome 6" if gene == "6" else gene for gene in genes]
genes = ["Chromosome 8" if gene == "8" else gene for gene in genes]
genes = ["Chromosome 7" if gene == "NTRK1/2/3" else gene for gene in genes]
genes = ["NTRK1" if gene == "Chromosome 7 & 17" else gene for gene in genes]
genes.append("Chromosome 17")
genes.append("NTRK2")
genes.append("NTRK3")

counts = Counter(genes)

unique_genes = list(set(genes))
# 71

# source 2

source2 = (
    "https://www.cancer.gov/about-cancer/diagnosis-staging/diagnosis/tumor-markers-list"
)

response = requests.get(source2)
soup = BeautifulSoup(response.content, "html.parser")


# Save the list to a file
with open("genes.json", "w") as f:
    json.dump(unique_genes, f)
