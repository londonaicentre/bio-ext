import pandas as pd
import re
from datetime import datetime
import pandera as pa
from pandera import Column, DataFrameSchema

# DEFINE FINAL SCHEMA

schema = DataFrameSchema(
    {
        "nhsnumber": Column(str, nullable=True),
        "gstt_hospnumber": Column(str, nullable=True),
        "mrn": Column(str, nullable=True),
        "ethnicity": Column(pa.Category, nullable=True),
        "sex": Column(pa.Category, nullable=True),
        "dateofbirth": Column(nullable=True),
        "dateofdeath": Column(nullable=True),
        "age": Column(float, nullable=True),
        "document_content": Column(str, nullable=True),
        "document_ordername": Column(str, nullable=True),
        "sampletest_datetime": Column(pa.DateTime, nullable=True),
        "jak2value_raw": Column(str, nullable=True),
        "jak2value_clean": Column(float, nullable=True),
        "source": Column(str, nullable=True),
    }
)


## CUSTOM FUNCTIONS
def extract_text(value, start_string, end_string):
    """extract strings between start and end

    Args:
        value (pandas column of strings): string col
        start_string (str): start words
        end_string (str): end words

    Returns:
        str: will return a string
    """
    pattern = re.escape(start_string) + r"(.*?)" + re.escape(end_string)
    match = re.search(pattern, value)
    if match:
        return match.group(1).strip()
    return None


# Function to extract JAK2 extract 10 characters either side
def extract_jak(text, step="firststep"):
    """this function will extract JAK2 valu
    THIS IS HARD CODED AND FUNCTION CAN BE IMPROVED
    WONT WORK IF FRINGE
    Args:
        text (_type_): _description_

    Returns:
        _type_: _description_
    """
    if text is None:
        return None
    else:
        if step == "firststep":
            pattern = re.compile(r".{0,10}JAK2 V617F%.{0,10}")
            match = pattern.search(text)
            if match:
                return match.group(0)
            return None
        elif step == "secondstep":
            pattern = re.compile(r"=\s*([\d.]+)")
            match = pattern.search(text)
            if match:
                return match.group(1)
            return None


## LOAD DATA
legacy = pd.read_csv("EPR_lab (2).csv")
epic = pd.read_csv("EPIC_lab.csv")

legacystringcols = ["patient_MRN", "patient_NhsNumber", "patient_HospitalNumber_GSTT"]
epicstringcols = ["patient_NhsNumber", "patient_HospitalNumber"]

## CONVERT TO STRINGS
legacy[legacystringcols] = legacy[legacystringcols].astype(str)
epic[epicstringcols] = epic[epicstringcols].astype(str)

# BASIC EDA
legacy.shape
epic.shape
# 2962 rows and 59 columns

# REVIEW MISSING DATA
legacy.isna().sum()
epic.isna().sum()

# REVIEW COLUMNS
for x in legacy.columns:
    print(legacy[x].value_counts())


# Logic: removing all files with PID except hosp number and NHS number
# and other non useful columns
colstoremove = [
    "_index",
    "activity_VisitNumber",
    "document_UpdatedBy",
    "_id",
    "activity_VisitCreatedBy",
    "activity_VisitSourceId",
    "document_UpdatedWhen",
    "patient_SourceId",
    "patient_CreatedWhen",
    "patient_Title",
    "id",
    "patient_MiddleName",
    "activity_VisitCreatedWhen",
    "patient_MaritalStatus",
    "patient_AddressCity",
    "patient_Religion",
    "activity_VisitCloseDate",
    "activity_VisitCareLevel",
    "activity_VisitProviderName",
    "activity_VisitUpdatedWhen",
    "patient_FirstName",
    "patient_UpdatedBy",
    "document_sourcedb",
    "activity_VisitCurrentLocation",
    "patient_LastName",
    "activity_VisitDischargeDate",
    "document_OrderId",
    "activity_VisitDischargeDisposition",
    "patient_AddressGeoLocation",
    "patient_AddressPostalCode",
    "document_CreatedBy",
    "document_CreatedWhen",
    "activity_VisitInternalStatus",
    "activity_VisitStatus",
    "activity_VisitDischargeLocation",
    "document_AncillaryReferenceId",
    "document_FacilityId",
    "patient_Address",
    "patient_CreatedBy",
    "activity_VisitUpdatedBy",
    "document_Type",
    "_score",
    "patient_Language",
    "patient_UpdatedWhen",
    "activity_Date",
    "activity_VisitCurrentLengthOfStay",
]

legacy = legacy.drop(columns=colstoremove, axis=1)

# REVIEW EPIC
for x2 in epic.columns:
    print(epic[x2].value_counts())

### REMOVE COLS FROM EPIC
df2colstoremove = ["_id", "_index", "_score", "patient_FirstName", "patient_LastName"]
epic = epic.drop(columns=df2colstoremove, axis=1)

# STANDARDISE COL NAMES
# convert all column names to lower
epic.columns = epic.columns.str.lower()
legacy.columns = legacy.columns.str.lower()

# EXTRACT TEST TIME
# LOGIC : tme stamp from the test is extracted not blood collection
# FOR LEGACY: labelled as 'enteredDate' and next 'comma

legacy["test_entered_dirty"] = legacy["document_fields"].apply(
    lambda x: (
        re.search(r"enteredDate(.*?),", x).group(1)
        if re.search(r"enteredDate(.*?),", x)
        else ""
    )
)
legacy["test_entered_clean"] = legacy["test_entered_dirty"].apply(
    lambda x: x.replace("': ", "").replace("`", "").replace("'", "")
)

## FOR EPIC
epic["test_date_dirty"] = epic["document_fields"].apply(
    lambda x: extract_text(x, start_string="resultDate", end_string="collectionDate")
)
epic["test_date_clean"] = epic["test_date_dirty"].apply(
    lambda x: x.replace("': ", "").replace(", '", "").replace("'", "").replace("'", "")
)

# LEGACY: find JAK2 detected patients only
# FIRST STEP, find patients who are explicitly not detected
# THEN, among patients who are not (not detected), we review them for presence o


# there are quite a lot of JAK2 not detected
# for example
legacy.at[2949, "document_content"]
legacy.at[0, "document_content"]

# lets find all the explicitly stated not detected and make it as a column
legacy["jak_notdetected"] = legacy["document_content"].str.contains(
    r"not detected", regex=True
)

temp = legacy[legacy["jak_notdetected"] == False]

# lets see what this looks like
temp.at[4, "document_content"]
# this looks good

temp.at[2947, "document_content"]

# also looks good
temp.isna().sum()

# here all unique identities have missing numbers
# the only is hospital_number_gstt
# so we will use that
# here there is 567 unique hosital

# logic that there should be 2 encounters at least
id_dict = temp["patient_nhsnumber"].value_counts().to_dict()
id_morethanone_dict = {key: value for key, value in id_dict.items() if value > 1}
legacy_nhs_numbers_definite = list(id_morethanone_dict.keys())

id_once_dict = {key: value for key, value in id_dict.items() if value < 2}
legacy_nhs_numbers_maybe = list(id_once_dict)

# CAST DEFINITES INTO DFP1
dfp1 = legacy[legacy["patient_nhsnumber"].isin(legacy_nhs_numbers_definite)]
dfp1 = dfp1[dfp1["jak_notdetected"] == False]

# EXTRACT JAK2 value
dfp1["jak2value_raw"] = dfp1["document_content"].apply(
    lambda x: extract_jak(x, step="firststep")
)
dfp1["jak2value_clean"] = dfp1["jak2value_raw"].apply(
    lambda x: extract_jak(x, step="secondstep")
)
dfp1["jak2value_clean"] = dfp1["jak2value_clean"].str.rstrip(".")
# Activity visit service show that there are quite a lot of cases which are bizarre
# e.g., cardiology.

dfp1["activity_visitservice"].value_counts()
key_activity_list = [
    "30381-Clinical Haematology",
    "30361-Clinical Haematology Nurse",
    "30983-Thrombophilia",
    "30081-General Medicine",
    "37081-Medical Oncology",
]

# here the patients still present that are relevant.
dfp1[~dfp1["activity_visitservice"].isin(key_activity_list)]

# FINAL REMOVE UNNECESSARY COLUMNS
dfp1 = dfp1.reset_index()
dfp1 = dfp1.drop(
    columns=[
        "index",
        "document_fields",
        "activity_visittype",
        "activity_visitservice",
        "test_entered_dirty",
        "jak_notdetected",
    ],
    axis=1,
)

dfp1 = dfp1[
    [
        "patient_nhsnumber",
        "patient_hospitalnumber_gstt",
        "patient_mrn",
        "patient_ethnicity",
        "patient_gender",
        "patient_dateofbirth",
        "patient_dateofdeath",
        "patient_age",
        "document_content",
        "document_ordername",
        "test_entered_clean",
        "jak2value_raw",
        "jak2value_clean",
    ]
]

clean_colnames_legacy = {
    "patient_nhsnumber": "nhsnumber",
    "patient_hospitalnumber_gstt": "gstt_hospnumber",
    "patient_mrn": "mrn",
    "patient_ethnicity": "ethnicity",
    "patient_gender": "sex",
    "patient_dateofbirth": "dateofbirth",
    "patient_dateofdeath": "dateofdeath",
    "patient_age": "age",
    "document_content": "document_content",
    "document_ordername": "document_ordername",
    "test_entered_clean": "sampletest_datetime",
    "jak2value_raw": "jak2value_raw",
    "jak2value_clean": "jak2value_clean",
}

dfp1.rename(columns=clean_colnames_legacy, inplace=True)

dfp1 = dfp1.astype(
    {
        "nhsnumber": "str",
        "gstt_hospnumber": "str",
        "mrn": "str",
        "ethnicity": "category",
        "sex": "category",
        "age": "float",
        "document_content": "str",
        "document_ordername": "category",
        "jak2value_raw": "str",
        "jak2value_clean": "float",
    }
)

datecolumns = ["dateofbirth", "dateofdeath"]
for col in datecolumns:
    dfp1[col] = pd.to_datetime(dfp1[col]).dt.date

dfp1["sampletest_datetime"] = dfp1["sampletest_datetime"].apply(
    lambda x: datetime.fromisoformat(x)
)

# fringe case on jak2 value of 369.
dfp1.at[369, "jak2value_clean"]

dfp1["source"] = "legacy"

# this is legacy data complete.

### LEGACY MAYBE has no missing nhs number so we can use this
legacy_maybe = legacy[legacy["patient_nhsnumber"].isin(legacy_nhs_numbers_maybe)]
legacy_maybe = legacy_maybe[legacy_maybe["jak_notdetected"] == False]
# for epic too we will use NHS number
legacy_maybe["jak2value_raw"] = legacy_maybe["document_content"].apply(
    lambda x: extract_jak(x, step="firststep")
)
legacy_maybe["jak2value_clean"] = legacy_maybe["jak2value_raw"].apply(
    lambda x: extract_jak(x, step="secondstep")
)
legacy_maybe["jak2value_clean"] = legacy_maybe["jak2value_clean"].str.rstrip(".")


## NOW WE WORK ON EPIC and can use nhs number
# between 'valueText' and 'unitOfMeasure'
epic.at[0, "document_fields"]

epic.at[917, "document_fields"]

epic.at[914, "document_fields"]

# GENERATE JAK2 DETECTION STATUS
epic["text"] = epic["document_fields"].apply(
    lambda x: extract_text(x, start_string="valueText", end_string="unitOfMeasure")
)

epic["jak_notdetected"] = epic["text"].str.contains(r"not detected", regex=True)
epic["jak_notdetected"].value_counts()

temp2 = epic[epic["jak_notdetected"] == False]
epic_id_dict = temp2["patient_nhsnumber"].value_counts().to_dict()
epic_morethanonce_dict = {
    key: value for key, value in epic_id_dict.items() if value > 1
}
epic_nhs_definite = list(epic_morethanonce_dict.keys())

epic_once_dict = {key: value for key, value in epic_id_dict.items() if value < 2}
epic_nhsnumbers_maybe = list(epic_once_dict.keys())

# CAST EPIC DEFINITES INTO DFP2
dfp2 = epic[epic["patient_nhsnumber"].isin(epic_nhs_definite)]
dfp2 = dfp2[dfp2["jak_notdetected"] == False]
# here again we will use where nhs number has appeared more than once

# EXTRACT JAK2 value
dfp2["jak2value_raw"] = dfp2["text"].apply(lambda x: extract_jak(x, step="firststep"))
substring = r"ample.\r\n"
dfp2["jak2value_raw"] = dfp2["jak2value_raw"].str.replace(substring, "", regex=False)
dfp2["jak2value_clean"] = dfp2["jak2value_raw"].apply(
    lambda x: extract_jak(x, step="secondstep")
)
dfp2["jak2value_clean"] = dfp2["jak2value_clean"].str.rstrip(".")

dfp2 = dfp2.reset_index()
dfp2 = dfp2.drop(
    columns=["index", "document_fields", "test_date_dirty", "jak_notdetected"], axis=1
)

dfp2[["mrn", "dateofdeath", "document_ordername"]] = None

dfp2 = dfp2[
    [
        "patient_nhsnumber",
        "patient_hospitalnumber",
        "mrn",
        "patient_ethnicity",
        "patient_gender",
        "patient_dateofbirth",
        "dateofdeath",
        "patient_age",
        "text",
        "document_ordername",
        "test_date_clean",
        "jak2value_raw",
        "jak2value_clean",
    ]
]

clean_colnames_epic = {
    "patient_nhsnumber": "nhsnumber",
    "patient_hospitalnumber": "gstt_hospnumber",
    "patient_ethnicity": "ethnicity",
    "patient_gender": "sex",
    "patient_dateofbirth": "dateofbirth",
    "patient_age": "age",
    "text": "document_content",
    "test_date_clean": "sampletest_datetime",
    "jak2value_raw": "jak2value_raw",
    "jak2value_clean": "jak2value_clean",
    "mrn": "mrn",
    "dateofdeath": "dateofdeath",
    "document_ordername": "document_ordername",
}

dfp2.rename(columns=clean_colnames_epic, inplace=True)

dfp2["dateofbirth"] = pd.to_datetime(dfp2["dateofbirth"]).dt.date
dfp2["sampletest_datetime"] = pd.to_datetime(dfp2["sampletest_datetime"])


dfp2 = dfp2.astype(
    {
        "nhsnumber": "str",
        "gstt_hospnumber": "str",
        "ethnicity": "category",
        "sex": "category",
        "age": "float",
        "document_content": "str",
        "jak2value_raw": "str",
        "jak2value_clean": "float",
    }
)

dfp2["source"] = "epic"

# EPIC MAYBE
epic_maybe = epic[epic["patient_nhsnumber"].isin(epic_nhsnumbers_maybe)]
epic_maybe = epic_maybe[epic_maybe["jak_notdetected"] == False]

epic_maybe["jak2value_raw"] = epic_maybe["text"].apply(
    lambda x: extract_jak(x, step="firststep")
)
epic_maybe["jak2value_raw"] = epic_maybe["jak2value_raw"].str.replace(
    substring, "", regex=False
)

epic_maybe["jak2value_clean"] = epic_maybe["jak2value_raw"].apply(
    lambda x: extract_jak(x, step="secondstep")
)

# INTERSECT
epicset = set(epic_nhsnumbers_maybe)
legacyset = set(legacy_nhs_numbers_maybe)
nhs_span_two = epicset & legacyset
epic_span = epic_maybe[epic_maybe["patient_nhsnumber"].isin(nhs_span_two)]
legacy_span = legacy_maybe[legacy_maybe["patient_nhsnumber"].isin(nhs_span_two)]

# legacy_span tidy
legacy_span["source"] = "legacy_across"
epic_span["source"] = "epic_across"

legacy_span = legacy_span.reset_index()
epic_span = epic_span.reset_index()

legacy_span = legacy_span.drop(
    columns=[
        "index",
        "document_fields",
        "activity_visittype",
        "activity_visitservice",
        "test_entered_dirty",
        "jak_notdetected",
    ],
    axis=1,
)

legacy_span = legacy_span[
    [
        "patient_nhsnumber",
        "patient_hospitalnumber_gstt",
        "patient_mrn",
        "patient_ethnicity",
        "patient_gender",
        "patient_dateofbirth",
        "patient_dateofdeath",
        "patient_age",
        "document_content",
        "document_ordername",
        "test_entered_clean",
        "jak2value_raw",
        "jak2value_clean",
    ]
]

legacy_span.rename(columns=clean_colnames_legacy, inplace=True)

legacy_span = legacy_span.astype(
    {
        "nhsnumber": "str",
        "gstt_hospnumber": "str",
        "mrn": "str",
        "ethnicity": "category",
        "sex": "category",
        "age": "float",
        "document_content": "str",
        "document_ordername": "category",
        "jak2value_raw": "str",
        "jak2value_clean": "float",
    }
)

for col in datecolumns:
    legacy_span[col] = pd.to_datetime(legacy_span[col]).dt.date

legacy_span["sampletest_datetime"] = legacy_span["sampletest_datetime"].apply(
    lambda x: datetime.fromisoformat(x)
)

epic_span = epic_span.drop(
    columns=["index", "document_fields", "test_date_dirty", "jak_notdetected"], axis=1
)

epic_span[["mrn", "dateofdeath", "document_ordername"]] = None


epic_span = epic_span[
    [
        "patient_nhsnumber",
        "patient_hospitalnumber",
        "mrn",
        "patient_ethnicity",
        "patient_gender",
        "patient_dateofbirth",
        "dateofdeath",
        "patient_age",
        "text",
        "document_ordername",
        "test_date_clean",
        "jak2value_raw",
        "jak2value_clean",
    ]
]

clean_colnames_epic = {
    "patient_nhsnumber": "nhsnumber",
    "patient_hospitalnumber": "gstt_hospnumber",
    "patient_ethnicity": "ethnicity",
    "patient_gender": "sex",
    "patient_dateofbirth": "dateofbirth",
    "patient_age": "age",
    "text": "document_content",
    "test_date_clean": "sampletest_datetime",
    "jak2value_raw": "jak2value_raw",
    "jak2value_clean": "jak2value_clean",
    "mrn": "mrn",
    "dateofdeath": "dateofdeath",
    "document_ordername": "document_ordername",
}

epic_span.rename(columns=clean_colnames_epic, inplace=True)

epic_span["dateofbirth"] = pd.to_datetime(epic_span["dateofbirth"]).dt.date
epic_span["sampletest_datetime"] = pd.to_datetime(epic_span["sampletest_datetime"])

epic_span = epic_span.astype(
    {
        "nhsnumber": "str",
        "gstt_hospnumber": "str",
        "ethnicity": "category",
        "sex": "category",
        "age": "float",
        "document_content": "str",
        "jak2value_raw": "str",
        "jak2value_clean": "float",
    }
)

legacy_span["source"] = "legacy_across"
epic_span["source"] = "epic_across"


try:
    schema.validate(legacy_span)
    print(f"validated")
except pa.errors.SchemaError as e:
    print(f"Validation error {e}")

# All the validations passed.

combined_df = pd.concat([dfp1, dfp2, legacy_span, epic_span], axis=0)
combined_df = combined_df.reset_index(drop=True)
combined_df.to_csv("clean_jak2_patientlist.csv", index=False)
combined_df.to_pickle("clean_jak2_patientlist.pkl")

yamlschema = schema.to_yaml("outputschema")
