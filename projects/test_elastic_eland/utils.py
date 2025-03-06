import pandas as pd
from elasticsearch import Elasticsearch


def aggregate_by_date(
    query: dict,
    index: str,
    date_field: str,
    session: Elasticsearch,
) -> pd.DataFrame:
    query_object = {
        "size": 0,
        "aggs": {
            "amount_per_month": {
                "date_histogram": {
                    "field": date_field,
                    "calendar_interval": "month",
                    "format": "yyyy-MM",
                }
            }
        },
    }

    query_object["query"] = query

    res = session.search(index=index, body=query_object)

    df = pd.DataFrame(res["aggregations"]["amount_per_month"]["buckets"])
    print(df.keys())

    df["date"] = pd.to_datetime(df["key_as_string"], format="%Y-%m")

    df = df.rename({"doc_count": "count"}, axis="columns")

    return df[["date", "count"]]


def aggregate_by_nhs_numbers(
    query: dict,
    index: str,
    session: Elasticsearch,
) -> pd.DataFrame:
    query_object = query_object = {
        "size": 0,
        "query": query,
        "aggs": {
            "nhs_number_count": {
                "terms": {
                    "field": "patient_NhsNumber",
                    "size": 100_000,
                }
            }
        },
    }

    res = session.search(index=index, body=query_object)

    df = pd.DataFrame(res["aggregations"]["nhs_number_count"]["buckets"])

    df["key"] = df["key"].astype(str)

    df = df.value_counts("doc_count", sort=True).reset_index(name="count_agg")
    df = df.rename(
        {"doc_count": "Count Frequency", "count_agg": "Density"}, axis="columns"
    )

    return df


def aggregate_by_event_age(
    query: dict,
    index: str,
    session: Elasticsearch,
) -> pd.DataFrame:
    query_object = {
        "size": 0,
        "query": query,
        "aggs": {"age_at_event": {"terms": {"field": "patient_Age", "size": 100_000}}},
    }

    res = session.search(index=index, body=query_object)

    df = pd.DataFrame(res["aggregations"]["age_at_event"]["buckets"])

    df = df.rename({"doc_count": "count"}, axis="columns")
    df["age"] = df["key"].astype(int)

    print(df)

    return df[["age", "count"]]

def get_number_of_results(query: dict,
    index: str,
    session: Elasticsearch) -> int:
    query_object = {
        "query": query,
    }

    return session.count(index=index, body=query_object)