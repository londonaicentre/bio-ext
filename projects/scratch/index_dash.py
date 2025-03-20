import pickle
import streamlit as st
from collections import Counter
import json
import os
import pandas as pd
from dotenv import load_dotenv
from elasticsearch import helpers
import streamlit as st
import pickle
from bioext.elastic_utils import ElasticsearchSession, GsttProxyNode
import csv

# Queries


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
