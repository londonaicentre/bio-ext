import os
from doccano_client import DoccanoClient
import json

PROJECT_NAME = 'Synthetic BRCA Classification'
PROJECT_DESCRIPTION = 'Classification of synthetic BRCA testing reports'
PROJECT_TYPE = 'DocumentClassification'
GUIDELINE = 'Classify BRCA results under the most appropriate label(s)'
LABELS = ['BRCA1 positive', 'BRCA2 positive', 'BRCA1 VUS', 'BRCA2 VUS', 'BRCA1 negative', 'BRCA2 negative', 'Invalid']
DATA_FILE = 'data/brca_reports.json'

def main():
    client = DoccanoClient('http://localhost:8000')
    client.login(username = os.getenv("DOCCANO_USERNAME"),
                password = os.getenv("DOCCANO_PASSWORD")
                )
    try:
        user = client.get_profile()
        print(f"Connected to Doccano as user: {user.username}")

    except Exception as e:
        print(f"Failed to connect to Doccano: {str(e)}")
        return

    # create project
    try:
        project = client.create_project(
            name=PROJECT_NAME,
            project_type=PROJECT_TYPE,
            description=PROJECT_DESCRIPTION,
            guideline=GUIDELINE
        )
        print(f"Created new project: {project.name}")

    except Exception as e:
        print(f"Failed to create project: {str(e)}")
        return
            
    # set up labels
    try:
        for label in LABELS:
            label_type = client.create_label_type(
                project_id=project.id,
                type='category',
                text=label
            )
            print(f"Created label: {label_type.text}")

    except Exception as e:
        print(f"Failed to setup labels: {str(e)}")
        return
    
    # load json
    try:
        with open(DATA_FILE, 'r') as file:
            data = json.load(file)
        
        for item in data:
            example = client.create_example(
                project_id=project.id,
                text=item['text']
            )
            print(f"Uploaded example: {example.id}")

    except Exception as e:
        print(f"Failed to load json samples: {str(e)}")
        return

if __name__ == "__main__":
    main()
    print("Setup complete")