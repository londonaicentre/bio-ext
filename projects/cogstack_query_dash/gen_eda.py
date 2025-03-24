import yaml
from csdash import dbaccess
from dotenv import load_dotenv
from tqdm import tqdm
from rich import print
from datetime import datetime
import os
from elasticsearch import helpers
from bioext.elastic_utils import ElasticsearchSession, GsttProxyNode

receptacle = []

load_dotenv()
print(os.environ.get("ELASTIC_API_ID"))

config = dbaccess.load_config("utils/config_dash.yaml")
es = dbaccess.connect_cogstack()
config["all_cols_query"]["size"] = 10
print("suceesfully connected ")

user_stats = {}
user_stats["user"] = os.environ.get("ELASTIC_API_ID")
user_stats["datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

cogstack_indexes = dbaccess.list_and_fetch_data(_es = es, query =config["all_cols_query"])
user_stats["indexes_available"] = cogstack_indexes[0]
receptacle.append(user_stats)


for i in tqdm(cogstack_indexes[0]):
    try:
        index_description = {}
        index_description["index_name"] = i
        index_description["docs_count"] = es.count(index=i)["count"]
        indexsize = es.indices.stats(index= i)
        indexsize = indexsize["indices"][i]["total"]["store"]["size_in_bytes"]
        indexsize_gb = indexsize/(1024*1024*1024)
        index_description["index_size"] = indexsize_gb
        print(f"A problem {i}")
        fields = es.indices.get_mapping(index=i)
        index_description["fields"] = fields
        print(f"Bproblem{i}")
        
        mappings = dbaccess.get_mapping_types(_es=es,indexname=i)
        index_description["mappings_by_type"] = mappings
        print(f"cproblem{i}")
        try: 
            keyword_fields = mappings["keyword"]
            top10 = dbaccess.get_top_10kw(_es=es,indexname=i, fieldlist=keyword_fields)
            index_description["keyword_fields_description"] = top10
            print(f"{i} has keyword Yay!")
        except:
            index_description["keyword_fields_none"] = True
            print(f"{i} does NOT have keyword")
        
        try: 
            datefields = mappings["date"]
            dateranges = dbaccess.get_date_ranges(_es=es,indexname=i,fieldlist = datefields)
            index_description["date_fields_ranges"] = dateranges
            print(f"{i} have datefields")
        except: 
            index_description["date_fields_none"] = True
            print(f"{i} do not have datefields")
        try:
            numcols = list(set(mappings["scaled_float"]+ mappings["integer"] + mappings["short"]))
            numsumm = dbaccess.get_num_stats(_es=es,indexname=i,fieldlist=numcols)
            index_description["numeric_columns_summary"] = numsumm
            print(f"{i} have numeric cols")
        except: 
            index_description["numeric_cols_none"] = True
            print(f"{i} do not have numeric columns")
       
        receptacle.append(index_description)
        
    except Exception as e:
        print(f"{e}")
        

# need to play with yaml indexes to actually get the correct format
with open("receptacle.yaml","w") as yaml_file:
    yaml_file.write(yaml.dump(receptacle, default_flow_style=False, sort_keys=False))


eg = es.indices.get_mapping(index=cogstack_indexes[0][0])
print(eg)