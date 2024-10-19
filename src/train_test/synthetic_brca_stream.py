import os
from doccano_client import DoccanoClient
import json

PROJECT_ID = 6

def get_label_map(client, project_id):
    """
    Get dictionary of label id to label text 
    """
    label_types = client.list_label_types(project_id=project_id, type='category')
    return {label_type.id: label_type.text for label_type in label_types}

def stream_samples(client, project_id, label_map):
    """
    Gets text and human-readable labels
    """ 
    for example in client.list_examples(project_id=project_id):
        categories = list(client.list_categories(project_id=project_id, example_id=example.id))
        labels = [label_map.get(category.label, f"unexpeted label: {category.label}") for category in categories]
        yield example.text, labels

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
    
    try:
        label_map = get_label_map(client, PROJECT_ID)
        print("Label Map:")
        for label_id, label_text in label_map.items():
            print(f"  ID {label_id}: {label_text}")
        
        print("Streaming samples from Doccano:")
        for i, (text, labels) in enumerate(stream_samples(client, PROJECT_ID, label_map), 1):
            print(f"\nSample {i}:")
            print(f"Text: {text[:100]}...")  # Print first 100 characters of text
            print(f"Labels: {labels}")
        
    except Exception as e:
        print(f"Unable to stream samples: {str(e)}")

if __name__ == "__main__":
    main()