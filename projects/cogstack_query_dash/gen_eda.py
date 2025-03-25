import yaml
from csdash import dbaccess
from dotenv import load_dotenv
from tqdm import tqdm
from rich import print
from datetime import datetime
import os
from elasticsearch import helpers
from bioext.elastic_utils import ElasticsearchSession, GsttProxyNode
import argparse


def generate_eda(detail=False):
    """script to do eda

    Args:
        details (bool, optional): defaults to False. to ensure no additional eda is done on keywords such as NHS number
    """

    # object to hold the results
    receptacle = []
    # connect to elastic database
    config = dbaccess.load_config("utils/config_dash.yaml")
    es = dbaccess.connect_cogstack()
    config["all_cols_query"]["size"] = 10
    escapekws = config["escapekwlist"]
    print("suceesfully connected ")

    # hold user stats
    user_stats = {}
    user_stats["user"] = os.environ.get("ELASTIC_API_ID")
    user_stats["datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # fetch data from cogstack
    cogstack_indexes = dbaccess.list_and_fetch_data(
        _es=es, query=config["all_cols_query"]
    )
    user_stats["indexes_available"] = cogstack_indexes[0]
    receptacle.append(user_stats)

    # do eda for each of the cogstack indexes
    for i in tqdm(cogstack_indexes[0]):
        try:
            # index description statistics
            index_description = {}
            index_description["index_name"] = i
            index_description["docs_count"] = es.count(index=i)["count"]
            indexsize = es.indices.stats(index=i)
            indexsize = indexsize["indices"][i]["total"]["store"]["size_in_bytes"]
            indexsize_gb = indexsize / (1024 * 1024 * 1024)
            index_description["index_size"] = indexsize_gb

            print(f"Index description completed for {i}")

            fields = es.indices.get_mapping(index=i)
            index_description["fields"] = fields
            print(f"Fields mapping completed for {i}")

            mappings = dbaccess.get_mapping_types(_es=es, indexname=i)
            index_description["mappings_by_type"] = mappings
            print(f"Mapping typed for {i}")

            # try loop for eda on keyword data types
            try:
                keyword_fields = mappings["keyword"]
                print(f"{i} has keyword_fields")
                if not detail:
                    print("efficient summarising")
                    keyword_fields_escaped = [
                        field for field in keyword_fields if field not in escapekws
                    ]
                    top10 = dbaccess.get_top_10kw(
                        _es=es, indexname=i, fieldlist=keyword_fields_escaped
                    )
                    index_description["keyword_fields_description"] = top10
                    print(f"{i} has keywords and is on EFFICIENT option.")
                else:
                    top10 = dbaccess.get_top_10kw(
                        _es=es, indexname=i, fieldlist=keyword_fields
                    )
                    index_description["keyword_fields_description"] = top10
                    print(f"{i} has keyword field detailed option!")
            except Exception as e:
                index_description["keyword_fields_none"] = True
                print(f"{i} does NOT have keywords and error out as {e}")

            # try loop for eda on date data types
            try:
                datefields = mappings["date"]
                dateranges = dbaccess.get_date_ranges(
                    _es=es, indexname=i, fieldlist=datefields
                )
                index_description["date_fields_ranges"] = dateranges
                print(f"{i} have datefields")
            except Exception as e:
                index_description["date_fields_none"] = True
                print(f"{i} do not have datefields and error as {e}")

            # try loop for eda on numeric data types
            try:
                numcols = list(
                    set(list(mappings["scaled_float"]) + list(mappings["integer"]))
                )
                numsumm = dbaccess.get_num_stats(_es=es, indexname=i, fieldlist=numcols)
                index_description["numeric_columns_summary"] = numsumm
                print(f"{i} have numeric cols")
            except Exception as e:
                index_description["numeric_cols_none"] = True
                print(f"{i} do not have numeric columns and error out with {e}")

            receptacle.append(index_description)

        except Exception as e:
            print(f"Some random error on outside exception loop {e}")

    return receptacle


if __name__ == "__main__":

    # parse CLI args
    parser = argparse.ArgumentParser(
        prog="eda_cogstack", description="alternative to streamlit"
    )
    # argument for path of file
    parser.add_argument(
        "--path",
        type=str,
        default="receptacle.yaml",
        help="Name and path of output yaml file to be saved.",
    )
    # argument for whether details needed or not
    parser.add_argument(
        "--detail",
        action="store_true",
        help="When --detail flag is present, then it will summarise all columns. *HIGHLY INEFFICIENT*",
    )
    args = parser.parse_args()
    # check and load to ensure you have permissions
    load_dotenv()
    print(os.environ.get("ELASTIC_API_ID"))
    # generate eda data
    eda_data = generate_eda(args.detail)

    # save as yaml
    with open(args.path, "w") as yaml_file:
        yaml_file.write(yaml.dump(eda_data, default_flow_style=False, sort_keys=False))
