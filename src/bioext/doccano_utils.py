import os
import json
from doccano_client import DoccanoClient
from datetime import datetime
import yaml

class DoccanoSession:
    def __init__(self, server=None):
        self.username = os.getenv("DOCCANO_USERNAME")
        self.password = os.getenv("DOCCANO_PASSWORD")
        self.server = os.getenv("DOCCANO_SERVER", "http://localhost:8000")

        self.user = None
        self.current_project_id = None
        self.client = self.create_session()

    def create_session(self):
        """
        Connect and log on to a Doccano server
        """
        client = DoccanoClient(self.server)
        client.login(username=self.username, password=self.password)
        self.user = client.get_profile()
        return client

    def _save_projectmetadata(self,project,filepath="projectid.yaml"):
        """this internal method will save project metadata as yaml at project root

        Args:
            project (Doccano project object): object which is an output of create_or_update method
            filepath (str, optional): name and path to save. Defaults to "projectid.yaml".
        """
        formatted_datetime = datetime.now().astimezone().strftime("%d-%m-%Y %H:%M:%S %Z%z")
        metadata = {
                "Doccano Project name": project.name,
                "Doccano Project ID": project.id,
                "Project creation time": formatted_datetime
            }
        
        try:
            with open(filepath, "w") as f:
                yaml.dump(metadata,f,sort_keys=False,default_flow_style=False)
            print(f"Metadata of project has been successfully written to {filepath}.")
        except (IOError,yaml.YAMLError) as e:
            print(f"An error {e} encountered and metadata is not saved.")
    
    def create_or_update_project(
        self,
        name,
        project_type,
        description,
        guideline: None,
        labels: None,
        label_type: None,
    ):
        """
        Register a new Doccano project
        """

        project_ids = []

        # Find project ID based on name
        for proj in self.client.list_projects():
            if name == proj.name:
                project_ids.append((proj.id, proj.tags))
        assert len(project_ids) < 2, "Multiple projects were found with the same name"

        # TODO: would you ever want to add more docs to a project that already exists?
        # Project is found, update details if allowed
        if len(project_ids) == 1:
            # check allow_update tag
            assert (
                "allow_update" in project_ids[0][1]
            ), f"Project found with matching name and ID {project_ids[0][0]} is not allowed to be updated"
            project = self.client.update_project(
                project_ids[0][0],
                name=name,
                project_type=project_type,
                description=description,
                guideline=guideline,
            )
            self.current_project_id = project.id
            #dump project metadata using internal method
            self._save_projectmetadata(project,filepath="projectid.yaml")
            return project

        # Project is NOT found, create it
        try:
            project = self.client.create_project(
                name=name,
                project_type=project_type,
                description=description,
                guideline=guideline,
            )
            self.current_project_id = project.id
            self.create_labels(labels, label_type)
            self._save_projectmetadata(project,filepath="projectid.yaml")
            return project
        except Exception as e:
            print(f"Failed to create project")
            raise e

    def create_labels(self, labels: list, label_type: str):
        """
        Given list of labels, set up labels for specified or active project
        """
        # Identify project
        if not self.current_project_id:
            raise ValueError("No project ID specified or available")

        # Create labels for project
        for lab in labels:
            self.client.create_label_type(
                project_id=self.current_project_id, type=label_type, text=lab
            )

        return labels

    def load_document(self, text, metadata=None, project_id=None):
        """
        Load a single document into specified project
        """
        project_id = project_id or self.current_project_id
        if not project_id:
            raise ValueError("No project ID specified or available")

        try:
            example = self.client.create_example(
                project_id=project_id,
                text=text,
                meta=metadata,
            )
            return example
        except Exception as e:
            print(f"Failed to load document: {e}")
            raise e

    def get_labelled_samples(self, project_id=None):
        """
        Streams text and associated labels as generator from specified or active project
        """
        project_id = project_id or self.current_project_id
        if not project_id:
            raise ValueError("No project ID specified or available")
        label_map = self._get_label_map(project_id)

        for example in self.client.list_examples(project_id=project_id):
            categories = list(
                self.client.list_categories(
                    project_id=project_id, example_id=example.id
                )
            )
            labels = [
                label_map.get(category.label, f"unexpected label: {category.label}")
                for category in categories
            ]
            yield example.text, labels

    def _get_label_map(self, project_id):
        """
        Private method to map readable labels to label ids for specified or active project
        Required by get_labelled_samples
        """
        label_types = self.client.list_label_types(
            project_id=project_id, type="category"
        )
        return {label_type.id: label_type.text for label_type in label_types}


def load_from_file(doc_session, data_file_path, doc_load_cfg):
    """Bulk upload documents from a folder, where each file is a doc.

    Args:
        doc_session (_type_): _description_
        data_file_path (str): path to folder with document files
        doc_load_cfg (dict): config details
    """
    # create project
    project = doc_session.create_or_update_project(**doc_load_cfg)
    # doc_session.update_project()
    print(f"Using project: {project.name}, with ID {project.id}")

    # load json from data file 
    for file in os.listdir(data_file_path):
        with open(os.path.join(data_file_path, file), "r") as file:
            data = json.load(file)
            # load json to doccano - TODO: avoid uploading duplicates
            doc_session.load_document(
                data["_source"]["text"], metadata={"source_id": data["_id"]}
            )
    print(f"Uploaded {len(os.listdir(data_file_path))} examples")
    


def stream_labelled_docs(doc_session, doc_stream_cfg):
    print(f"Connected to Doccano as user: {doc_session.username}")

    # iterator
    labelled_samples = doc_session.get_labelled_samples(doc_stream_cfg["PROJECT_ID"])

    # print labelled samples
    for i, (text, labels) in enumerate(labelled_samples, 1):
        print(f"\nSample {i}:")
        print(f"Text: {text[:50]}...")
        print(f"Labels: {labels}")
