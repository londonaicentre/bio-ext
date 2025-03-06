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
        "query": query,
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

    res = session.search(index=index, body=query)

    df = pd.DataFrame(res["aggregations"]["amount_per_month"]["buckets"])
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
            "amount_per_month": {
                "nhs_number_count": {
                    "field": "patient_NhsNumber",
                    "size": 100_000,
                }
            }
        },
    }

    res = es_session.es.search(index=INDEX, body=query)

    df = pd.DataFrame(res["aggregations"]["nhs_number_count"]["buckets"])

    df = df.rename({"key": "NHS Number", "doc_count": "count"})

    return df


def aggregate_by_event_age(
    query: dict,
    index: str,
    session: Elasticsearch,
) -> pd.DataFrame:
    query_object = query_object = {
        "size": 0,
        "query": query,
        "aggs": {"age_at_event": {"terms": {"field": "patient_Age", "size": 100_000}}},
    }

    res = es_session.es.search(index=INDEX, body=query)

    df = pd.DataFrame(res["aggregations"]["age_at_event"]["buckets"])

    df = df.rename({"doc_count": "count"})
    df['age'] = df['key'].astype(int)

    return df[['age', 'count']]
