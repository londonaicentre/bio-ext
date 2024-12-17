import os
import json
from elasticsearch import helpers


def retrieve_docs(es_client, project_dir, query_cfg):
    index_name = query_cfg["index_name"]
    query = {"query": query_cfg["query"]}
    try:
        results = helpers.scan(
            client=es_client, query=query, scroll="2m", index=index_name
        )

        processed_count = 0
        for hit in results:
            doc_id = hit["_id"]
            file_path = os.path.join(project_dir, f"{doc_id}.json")

            with open(file_path, "w") as f:
                json.dump(hit, f, indent=2)

            processed_count += 1
            if processed_count % 1000 == 0:
                print(f"Up to {processed_count} docs...")

        print(f"\nTotal: {processed_count} docs")

    except Exception as e:
        print(f"Error: {str(e)}")
