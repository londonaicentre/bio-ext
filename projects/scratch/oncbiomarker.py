import pandas as pd
from collections import Counter
import json

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

genes = list(solid["target_genes"])

genes = [gene for gene in genes if gene != "All including burden / signature"]

len(genes)

counts = Counter(genes)

unique_genes = list(set(genes))
# 71


# Save the list to a file
with open("genes.json", "w") as f:
    json.dump(unique_genes, f)
