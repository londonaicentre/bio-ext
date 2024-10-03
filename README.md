# bio-ext
Biomarker and deep phenotype extraction platform, as an extension to CogStack, or other unstructured data stores in hospital EHR systems 


## Elasticsearch connector
Usage

```
from elastic_connect import ElasticsearchSession
session = ElasticsearchSession()
## OR
session = ElasticsearchSession(server="https://sv-pr-elastic01:9200")
```