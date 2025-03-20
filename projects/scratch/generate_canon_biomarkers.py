import pandas as pd
import numpy as np
import re
import requests
from bs4 import BeautifulSoup
import yaml
from datetime import datetime
from pathlib import Path

# FUNCTIONS
with open("config.yaml", "r") as file:
    """load config file using safe_load"""
    config = yaml.safe_load(file)


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


def generate_nhsbiomarker(
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
    # this code unpacks list andalso if pipe symbol is used.
    _genes = [item for sublist in [x.split(",") for x in _genes] for item in sublist]
    _genes = [re.sub(r"^\s+|\s+$", "", gene) for gene in _genes]

    return _genes, finaldf


def generate_unique_biomarkers(receptacle, listobject, label):
    """create a unique list

    Args:
        receptacle (where to store / dict): where the results will be stored
        listobject (list): unique gene list
        label (str): custom label

    Returns:
        list: unique list
    """
    unique = list(set(listobject))
    receptacle["biomarker"].append(unique)
    receptacle["source"].append(label)
    return unique


def save_to_local(destination, dataframes, filenames, nocsv=True):
    """save files to local as parquet

    Args:
        destination (str): where files will be saved
        dataframes (list): list of dataframes
        filenames (list): list of strings filenames
    """
    dest = Path(destination)
    if not dest.exists():
        dest.mkdir(parents=True)
        print(f"new directory created as {dest}")
    if nocsv:
        pathslist = [dest / f"{filename}.gzip" for filename in filenames]
        for i, df in enumerate(dataframes):
            dataframes[i].to_parquet(pathslist[i])
    else:
        pathslist = [dest / f"{filename}.csv" for filename in filenames]
        for i, df in enumerate(dataframes):
            dataframes[i].to_csv(pathslist[i])

    print("files written")


# DATA SOURCES

sheets_dict = pd.read_excel(
    io=config["sources"][0],
    engine="openpyxl",
    sheet_name=None,
)

mappings = config["source0clean_names"]
# this mapping will be used to rename and clean some gene names


def clean_and_generate_genes():
    ##########
    # TASK1: working on SOURCE 1 : UK NHS
    # WILL ONLY PARSE NECESSARY COLUMNS

    # will collect result in
    result1 = {"biomarker": [], "source": []}
    # at the moment, not iterated over sheets as there are unique sheet specific cleanings

    ## TASK1a solid tumours
    solid, df_solid = generate_nhsbiomarker(
        config["source0sheetslist"][0],
        colnames=config["source0gencolnames"],
        ffillcols=config["source0fillcolumns"],
        source=sheets_dict,
    )

    solid = [mappings.get(gene, gene) for gene in solid]
    solid.append("Chromosome 17")
    solid.append("NTRK2")
    solid.append("NTRK3")

    unique_solids = generate_unique_biomarkers(
        receptacle=result1, listobject=solid, label="uk_nhs_solidtumours"
    )

    ## TASK 1b: neuro tumours
    neuro, df_neuro = generate_nhsbiomarker(
        config["source0sheetslist"][1],
        colnames=config["source0gencolnames"],
        ffillcols=config["source0fillcolumns"],
        source=sheets_dict,
    )
    neuro = [mappings.get(gene, gene) for gene in neuro]
    unique_neuro = generate_unique_biomarkers(
        receptacle=result1, listobject=neuro, label="uk_nhs_neurotumours"
    )

    ## TASK 1c: sarcoma
    sarcoma, df_sarcoma = generate_nhsbiomarker(
        config["source0sheetslist"][2],
        colnames=config["source0gencolnames"],
        ffillcols=config["source0fillcolumns"],
        source=sheets_dict,
    )
    sarcoma = [mappings.get(gene, gene) for gene in sarcoma]
    unique_sarcoma = generate_unique_biomarkers(
        receptacle=result1, listobject=sarcoma, label="uk_nhs_sarcoma"
    )

    ## TASK1d: haematological
    ## note different colnames
    haemonc, df_haem = generate_nhsbiomarker(
        config["source0sheetslist"][3],
        colnames=config["source0haemcolnames"],
        ffillcols=config["source0fillcolumns"],
        source=sheets_dict,
    )
    haemonc.remove("Genome-wide (high resolution)")
    haemwgs = ["WGS Germline and tumour", "WGS Tumour First", "WGS Follow-up Germline"]
    haemwgsrepeat = haemwgs * 24
    haemonc.extend(haemwgsrepeat)

    haemonc = [mappings.get(gene, gene) for gene in haemonc]

    unique_haem = generate_unique_biomarkers(
        receptacle=result1, listobject=haemonc, label="uk_nhs_haematological"
    )

    ## TASK 1e: paediatric
    paeds, df_paed = generate_nhsbiomarker(
        config["source0sheetslist"][4],
        colnames=config["source0gencolnames"],
        ffillcols=config["source0fillcolumns"],
        source=sheets_dict,
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
    unique_biomarkers = list(set(unique_biomarkers))
    df_unique = pd.DataFrame(result1)
    df_unique = df_unique.explode("biomarker").reset_index(drop=True)

    ######
    # TASK2 : USA NIH

    ## create receptacle
    results2 = {"biomarker": [], "conditions": [], "analysed": [], "usage": []}

    soup2 = create_soup(config["sources"][1])

    ## Find all p tags
    p_tags = soup2.find_all("p")

    ## custom conditions
    condition_pattern = r":\s*(.*?)\s*What"  # looks for colon and any text until "what"
    analysed_pattern = (
        r"analyzed:\s*(.*?)\s*How"  # looks for analyzed: and selects until How"
    )
    usage_pattern = r"How used:\s*(.*)"  # looks for any after "how used"

    ## iterate over the <strong>
    for p in p_tags:
        strong = p.find("strong")
        if strong and not p.get("style"):
            biomarker = strong.get_text()
            results2["biomarker"].append(biomarker)

            description = p.find_next("p", style=lambda x: "padding-left" in x)
            text = description.get_text()

            if text is not None:
                condition = re.findall(condition_pattern, text, re.DOTALL)
                results2["conditions"].append(condition[0])

                tissue = re.findall(analysed_pattern, text, re.DOTALL)
                results2["analysed"].append(tissue[0])

                usage = re.findall(usage_pattern, text, re.DOTALL)
                results2["usage"].append(usage[0])

    ## Task 2 outputs
    df_nih = pd.DataFrame(results2)
    df_nih = df_nih.assign(source="usa_nih")

    # TASK 3
    ## create task 3 receptacle
    result3 = {"biomarker": [], "definition": [], "synonyms": [], "conditions": []}
    soup3 = create_soup(config["sources"][2])

    ## find the correct <div>
    markers = soup3.find_all("div", class_="marker")

    ## regex conditions specific for task 3
    synonym_condition = r"d:\s*(.*)"  # looks for "d:" then all after
    associated_condition = r"rs:\s*(.*)"  # similarly for "rs:"

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
            result3["synonyms"].append(synonyms_parsed.group(1))
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

    ## TASK 4 new data source from JZ
    path = Path("data/") / config["sources"][3]
    df_source4 = pd.read_excel(
        io=path,
        engine="openpyxl",
        sheet_name=None,
    )

    df_conf = df_source4["Sheet1"]
    # TASK 4 outputs: extract the biomarkers only
    df_conf = df_conf[df_conf["Category"] == "Biomarker"]
    df_conf = pd.DataFrame(
        {"biomarker": df_conf["Criteria"].tolist(), "source": "jzconf"}
    )

    # MAKE A UNIQUE GENE NAMES ONLY CSV
    all_unique_df = pd.concat(
        [
            df_unique[["biomarker", "source"]],
            df_nih[["biomarker", "source"]],
            df_acc[["biomarker", "source"]],
            df_conf[["biomarker", "source"]],
        ],
        ignore_index=True,
    )
    all_unique_df = all_unique_df.drop_duplicates(subset="biomarker")

    return df_unique, df_nih, df_acc, df_conf, all_unique_df


if __name__ == "__main__":
    """
    when called as script, will save to local dev
    first make dataframes, then will add time stamp
    then write files as per below.

    """
    dataframes = clean_and_generate_genes()

    # OUTPUT DESTINATION
    outputpath = config["outputpath"]
    csvnames = config["outputfilenames"]
    timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    csv_timestamped = [f"{filename}_{timestamp}" for filename in csvnames]
    save_to_local(
        destination=outputpath,
        dataframes=dataframes,
        filenames=csv_timestamped,
        nocsv=True,
    )
