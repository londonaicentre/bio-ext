import json
import os
import pandas as pd
from dotenv import load_dotenv
from elasticsearch import helpers
import streamlit as st
import pickle
from bioext.elastic_utils import ElasticsearchSession, GsttProxyNode
import csv

# CUSTOM FUNCS
query = {
        "size": sample_size,
        "query":{"match_all":{}}
    }

query_2 =  {
        "_source":[
            "patient_NHSNumber",
            "document_Name"
            
        ],
        "size": sample_size,
        "query":{"match_all":{}}
    }

def fetch_sample_data(query, index_name, sample_size = 1000):
    response = es.search(index = index_name, body = query)
    _results = response["hits"]["hits"]
    _df = pd.json_normalize(_results)
    return _df 


# CHECK DOTENV IMPORT IS CORRECT
load_dotenv()
# load_dotenv("bio-ext/projects/scratch/.env")
print(os.environ.get("ELASTIC_API_ID"))

# PROJECT DIRECTORIES 
outputpath = "data/results.pkl"
results ={}

# ES SESSION INSTANTIATION 
es_session = ElasticsearchSession(conn_mode="API",proxy=GsttProxyNode)
es = es_session.es

# EXTRACT INDEXES AND COLUMNS LIST 
cogstack_indexes = list(es.indices.get_mapping().keys())

for index in cogstack_indexes:
    _data = fetch_sample_data(index_name = index)
    results[index] = _data
    print(f"{index} is appended to dict.")
    
print(f"{len(results)} is in the dict.")

colslist = []
for key in results.keys():
    _df = results[key]
    cols = list(_df.columns)
    colslist.append(cols)

colslist = sum(colslist,[])
colslist = list(set(colslist))


with open(outputpath,"wb") as handle:
    pickle.dump(results,handle, protocol = pickle.HIGHEST_PROTOCOL)
    
print("samples successfully pickled")

with open("data/cols.pkl","wb") as handle:
    pickle.dump(colslist,handle, protocol = pickle.HIGHEST_PROTOCOL)

print(colslist)
dfcols = pd.DataFrame(colslist)
dfcols.to_csv("data/cols.csv")
print("df cols written")
# have two 
# steps of st.write. 
# one to generalise data cleaning 

#st.write(test)
# print(test)

tdf = fetch_sample_data(query=query_2,index_name="gstt_clinical_discharge_letters_20230123")
# print("successful")


# query = {
#     "query": {
#         "bool": {
#             "must": [
#                 {"wildcard": {"document_Content": "*brca*"}},
#                 {"wildcard": {"document_Content": "*breast*"}},
#             ]
#         }
#     }
# }

# try:
#     results = helpers.scan(
#         client=es_session.es,
#         query=query,
#         scroll="2m",
#         index="gstt_clinical_geneworks_documents",
#     )

#     processed_count = 0
#     for hit in results:
#         doc_id = hit["_id"]
#         file_path = os.path.join(project_dir, f"{doc_id}.json")

#         with open(file_path, "w") as f:
#             json.dump(hit, f, indent=2)

#         processed_count += 1
#         if processed_count % 1000 == 0:
#             print(f"Up to {processed_count} docs...")

#     print(f"\nTotal: {processed_count} docs")

# except Exception as e:
#     print(f"Error: {str(e)}")

"""
{
    "_source": [
        "id",
        "document_EpicId",
        "activity_Date",
        "document_CreatedWhen",
        "document_UpdatedWhen",
        "patient_NHSNumber",
        "patient_DurableKey",
        "activity_EncounterEpicCsn",
        "activity_PatientAdministrativeCategory",
        "activity_PatientClass",
        "activity_Type",
        "activity_VisitClass",
        "activity_VisitType",
        "activity_DepartmentSpecialty",
        "activity_ChiefComplaint",
        "document_Content",
        "document_Name",
        "document_AuthorType",
        "document_Service",
        "document_Status"
    ],
    "query": {
        "bool": {
            "must": [],
            "filter": [
                {
                    "match_phrase": {
                        "activity_Type": "Clinic/Practice Visit"
                    }
                }
            ],
            "should": [],
            "must_not": [
                {
                    "match_phrase": {
                        "document_Name.keyword": "Appointment Note"
                    }
                },
                {
                    "match_phrase": {
                        "document_Name.keyword": "Nursing Note"
                    }
                }
            ]
        }
    }
}
```
"""
{
    "_source": [
        "id",
        "document_EpicId",
        "activity_Date",
        "document_CreatedWhen",
        "document_UpdatedWhen",
        "patient_NHSNumber",
        "patient_DurableKey",
        "activity_EncounterEpicCsn",
        "activity_PatientAdministrativeCategory",
        "activity_PatientClass",
        "activity_Type",
        "activity_VisitClass",
        "activity_VisitType",
        "activity_DepartmentSpecialty",
        "activity_ChiefComplaint",
        "document_Content",
        "document_Name",
        "document_AuthorType",
        "document_Service",
        "document_Status"
    ],
    "query": {
        "bool": {
            "must": [],
            "filter": [
                {
                    "match_phrase": {
                        "activity_Type": "Clinic/Practice Visit"
                    }
                }
            ],
            "should": [],
            "must_not": [
                {
                    "match_phrase": {
                        "document_Name.keyword": "Appointment Note"
                    }
                },
                {
                    "match_phrase": {
                        "document_Name.keyword": "Nursing Note"
                    }
                }
            ]
        }
    }
}


index_sizes = es.cat.indices(format="json",h=["index","docs.count"])

mapping = es.indices.get_mapping(index = "gstt_clinical_epic_imaging_reports_20250123")