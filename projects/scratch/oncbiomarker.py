import pandas as pd
from collections import Counter
import json
import math
import numpy as np
import re
import requests
from bs4 import BeautifulSoup


# FUNCTIONS


def create_soup(url: str):
    """create a beautiful soup object given a url

    Args:
        url (str): a web link

    Returns:
        _type_: soup object
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    return soup


def clean_uk_nhs(
    sheetname: str,
    colnames: list,
    ffillcols: list,
    source: dict,
):
    """basic subsetting and cleaning to be iterated over the list of sheets in UK dataset

    Args:
        source(pandas dataframe object as dict): source dataframe
        sheetname (string): sheet name to be cleaned
        colnames (list): colnames that we will rename to aid this function
        ffillcols (list): colnames where we will do forward fill only

    Returns:
        list object: python list object of genes
    """

    _df = source[sheetname]
    _df = _df.drop(index=_df.index[0], axis=0)
    _df.columns = colnames
    _df[ffillcols] = _df[ffillcols].ffill()

    _df["target_genes"] = _df["target_genes"].replace({np.nan: None})
    finaldf = _df[
        [
            "clinical_indication_name",
            "test_name",
            "target_genes",
            "test_scope",
            "technology",
        ]
    ]
    finaldf = finaldf.assign(datasource=sheetname)
    _genes = list(_df["target_genes"])
    _genes = [gene for gene in _genes if gene != "All including burden / signature"]
    _genes = [gene for gene in _genes if gene is not None]

    # this code wont work without removing nan's
    _genes = [item for sublist in [x.split(",") for x in _genes] for item in sublist]
    _genes = [re.sub(r"^\s+|\s+$", "", gene) for gene in _genes]
    print(len(_genes))

    return _genes, finaldf


def generate_unique_biomarkers(receptacle, listobject, label):
    """create a unique list

    Args:
        receptacle (where to store / dict): where the results will be stored
        listobject (the list to work on): _description_
        label (str): custom label

    Returns:
        list: unique list
    """
    unique = list(set(listobject))
    receptacle["biomarker"].append(unique)
    receptacle["source"].append(label)
    return unique


# DATA SOURCES

source_links = [
    "https://www.england.nhs.uk/wp-content/uploads/2018/08/cancer-national-genomic-test-drectory-V11-January-2025.xlsx",
    "https://www.cancer.gov/about-cancer/diagnosis-staging/diagnosis/tumor-markers-list",
    "https://www.accc-cancer.org/home/learn/precision-medicine/cancer-diagnostics/biomarkers/biomarkerlive/lexicon/cancer-biomarkers#B",
]

sheets_dict = pd.read_excel(
    io=source_links[0],
    engine="openpyxl",
    sheet_name=None,
)

##########
# TASK1: working on SOURCE 1 : UK NHS
# WILL ONLY PARSE NECESSARY COLUMNS

sheetslist = [
    "Solid tumours",
    "Neurological tumours",
    "Sarcoma",
    "Haematological",
    "Paediatric",
]

colnames = [
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

fillcolumns = ["group", "ci_code", "clinical_indication_name"]

# will collect result in
result1 = {"biomarker": [], "source": []}
# at the moment, not iterated over sheets as there are unique sheet specific cleanings

## TASK1a solid tumours
solid, df_solid = clean_uk_nhs(
    "Solid tumours", colnames=colnames, ffillcols=fillcolumns, source=sheets_dict
)

solid = ["Chromosome 1p" if gene == "1p" else gene for gene in solid]
solid = ["Chromosome 3" if gene == "3" else gene for gene in solid]
solid = ["Chromosome 6" if gene == "6" else gene for gene in solid]
solid = ["Chromosome 8" if gene == "8" else gene for gene in solid]
solid = ["Chromosome 7" if gene == "NTRK1/2/3" else gene for gene in solid]
solid = ["NTRK1" if gene == "Chromosome 7 & 17" else gene for gene in solid]
solid.append("Chromosome 17")
solid.append("NTRK2")
solid.append("NTRK3")


unique_solids = generate_unique_biomarkers(
    receptacle=result1, listobject=solid, label="uk_nhs_solidtumours"
)

## TASK 1b: neuro tumours
neuro, df_neuro = clean_uk_nhs(
    "Neurological tumours", colnames=colnames, ffillcols=fillcolumns, source=sheets_dict
)
neuro = [
    (
        "SNP array"
        if gene == "Dependent on clinical indication or specified request"
        else gene
    )
    for gene in neuro
]
unique_neuro = generate_unique_biomarkers(
    receptacle=result1, listobject=neuro, label="uk_nhs_neurotumours"
)

## TASK 1c: sarcoma
sarcoma, df_sarcoma = neuro = clean_uk_nhs(
    "Sarcoma", colnames=colnames, ffillcols=fillcolumns, source=sheets_dict
)
sarcoma = [
    (
        "SNP array"
        if gene == "Dependent on clinical indication or specified request"
        else gene
    )
    for gene in sarcoma
]
unique_sarcoma = generate_unique_biomarkers(
    receptacle=result1, listobject=sarcoma, label="uk_nhs_sarcoma"
)


## TASK1d: haematological
## not different colnames
colnameshaem = [
    "group",
    "ci_code",
    "clinical_indication_name",
    "test_code",
    "test_name",
    "target_genes",
    "test_scope",
    "technology",
    "optimal_family_structure",
    "further_eligibility_criteria",
    "changes_since_v1.0",
]
haematol, df_haem = clean_uk_nhs(
    "Haematological", colnames=colnameshaem, ffillcols=fillcolumns, source=sheets_dict
)
haematol.remove("Genome-wide (high resolution)")
haemwgs = ["WGS Germline and tumour", "WGS Tumour First", "WGS Follow-up Germline"]
haemwgsrepeat = haemwgs * 24
haematol.extend(haemwgsrepeat)

haematol = [("FUS-ERG" if gene == "e.g. FUS-ERG" else gene) for gene in haematol]
haematol = [
    ("FISH copy number and rearrangement" if gene == "As appropriate" else gene)
    for gene in haematol
]

unique_haem = generate_unique_biomarkers(
    receptacle=result1, listobject=haematol, label="uk_nhs_haematological"
)

## TASK 1e: paediatric
paeds, df_paed = clean_uk_nhs(
    "Paediatric", colnames=colnames, ffillcols=fillcolumns, source=sheets_dict
)
unique_paed = generate_unique_biomarkers(
    receptacle=result1, listobject=paeds, label="uk_nhs_paeds"
)


## TASK 1 outputs
df_uknhs = pd.concat([df_solid, df_neuro, df_sarcoma, df_haem, df_paed])
df_uknhs = df_uknhs.reset_index(drop=True)
unique_biomarkers = (
    unique_solids + unique_neuro + unique_sarcoma + unique_haem + unique_paed
)
df_unique = pd.DataFrame(result1)
df_unique = df_unique.explode("biomarker").reset_index(drop=True)

######
# TASK2 : USA NIH

## create receptacle
results2 = {"biomarker": [], "conditions": [], "analysed": [], "usage": []}

soup2 = create_soup(source_links[0])

## Find all p tags
p_tags = soup2.find_all("p")

## custom conditions
condition_pattern = r":\s*(.*?)\s*What"
analysed_pattern = r"analyzed:\s*(.*?)\s*How"
usage_pattern = r"How used:\s*(.*)"

## iterate over the <strong>
for p in p_tags:
    strong = p.find("strong")
    if strong and not p.get("style"):
        biomarker = strong.get_text()
        results2["biomarker"].append(biomarker)

        description = p.find_next("p", style=lambda x: "padding-left" in x)
        text = description.get_text()

        if text is None:
            pass
        else:

            condition = re.findall(condition_pattern, text, re.DOTALL)
            results2["conditions"].append(condition[0])

            tissue = re.findall(analysed_pattern, text, re.DOTALL)
            results2["analysed"].append(tissue[0])

            usage = re.findall(usage_pattern, text, re.DOTALL)
            results2["usage"].append(usage[0])

## Task 2 outputs
df_nih = pd.DataFrame(results2)
df_nih = df_nih.assign(source="usa_nih")

# TASK 2
## create task 3 receptacle
result3 = {"biomarker": [], "definition": [], "synonyms": [], "conditions": []}
soup3 = create_soup(source_links[1])

## find the correct <div>
markers = soup3.find_all("div", class_="marker")

## regex conditions specific for task 3
synonym_condition = r"d:\s*(.*)"
associated_condition = r"rs:\s*(.*)"

## iterate over the divs
for m in markers:
    name = m.find("p")
    result3["biomarker"].append(name.get_text())

    ptags = m.find_all("p")
    if len(ptags) > 3:
        # get definitions
        definitions = ptags[1].get_text()
        result3["definition"].append(definitions)
        # get synonyms which need parsing
        synonyms = ptags[2].get_text()
        synonyms_parsed = re.search(synonym_condition, synonyms, re.DOTALL)
        result3["synonyms"].append(synonyms_parsed)
        # get conditions which need parsing
        conditions = ptags[3].get_text()
    else:
        definitions = ptags[1].get_text()
        result3["definition"].append(definitions)
        result3["synonyms"].append("none")
        conditions = ptags[2].get_text()

    condition_parsed = re.search(associated_condition, conditions, re.DOTALL)
    result3["conditions"].append(condition_parsed[1])


## TASK 3 outputs
df_acc = pd.DataFrame(result3)
df_acc = df_acc.assign(source="usa_acc")


# WRITES OUTPUTS
## OUTPUT1 :
df_unique.to_csv("data/output/canonical_uk_nhs_genes.csv")
df_nih.to_csv("data/output/canonical_usa_nih_genes.csv")
df_acc.to_csv("data/output/canonical_usa_acc_genes.csv")
print("all dataframes written.")

## OUTPUT 2:
all_unique_df = pd.concat(
    [
        df_unique[["biomarker", "source"]],
        df_nih[["biomarker", "source"]],
        df_acc[["biomarker", "source"]],
    ]
)
all_unique_df = all_unique_df.reset_index(drop=True)
all_unique_df = all_unique_df.drop_duplicates(subset="biomarker")
all_unique_df = all_unique_df.reset_index(drop=True)
all_unique_df.to_csv("data/output/canonical_combined_unique_genes.csv")
print("unique biomarkers complete")
