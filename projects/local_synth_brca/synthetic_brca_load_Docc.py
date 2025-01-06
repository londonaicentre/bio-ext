import json


def file2Doc(doc_session, data_file_path, doc_load_cfg):
    # connect and log on to doccano
    print(f"Connected to Doccano as user: {doc_session.username}")

    # create project
    project = doc_session.create_project(**doc_load_cfg["PROJECT_DETAILS"])
    print(f"Created new project: {project.name}, {project.id}")

    # set up labels
    doc_session.setup_labels(doc_load_cfg["LABELS"])
    print(f"Created {len(doc_load_cfg['LABELS'])} labels")

    # load json from data file
    try:
        with open(data_file_path, "r") as file:
            data = json.load(file)
    except Exception as e:
        print(f"Failed to load samples: {str(e)}")
        return

    # load json to doccano
    doc_session.load_simple_json(data)
    print(f"Uploaded {len(data)} examples")
