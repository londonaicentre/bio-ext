import os
from doccano_client import DoccanoClient

class DoccanoSession:
    def __init__(self, server=None):
        self.username = os.getenv("DOCCANO_USERNAME")
        self.password = os.getenv("DOCCANO_PASSWORD")
        self.server = server or 'http://localhost:8000'
        self.client = self.create_session()
        self.user = None

    def create_session(self):
        """
        Connect and log on to a Doccano server
        """      
        client = DoccanoClient(self.server)
        client.login(username=self.username, password=self.password)
        self.user = client.get_profile()
        return client

    def create_project(self, name, project_type, description, guideline):
        """
        Register a new Doccano project
        """
        try:
            project = self.client.create_project(
                name=name,
                project_type=project_type,
                description=description,
                guideline=guideline
            )
            self.current_project_id = project.id
            return project
        except Exception as e:
            print(f"Failed to create project")
            raise e

    def setup_labels(self, labels, project_id=None):
        """
        Given list of labels, set up labels for specified or active project
        """
        project_id = project_id or self.current_project_id
        if not project_id:
            raise ValueError("No project ID specified or available")        
        try:
            for label in labels:
                self.client.create_label_type(
                    project_id=project_id,
                    type='category',
                    text=label
                )
        except Exception as e:
            print(f"Failed to setup labels")
            raise e

    def load_simple_json(self, data, project_id=None):
        """
        Given json.load data, load in 'text' fields to specified or active project
        """
        project_id = project_id or self.current_project_id
        if not project_id:
            raise ValueError("No project ID specified or available")        
        for item in data:
            self.client.create_example(
                project_id=project_id,
                text=item['text']
            )
    
    def get_labelled_samples(self, project_id=None):
        """
        Streams text and associated labels as generator from specified or active project
        """
        project_id = project_id or self.current_project_id
        if not project_id:
            raise ValueError("No project ID specified or available")         
        label_map = self._get_label_map(project_id)

        for example in self.client.list_examples(project_id=project_id):
            categories = list(self.client.list_categories(project_id=project_id, example_id=example.id))
            labels = [label_map.get(category.label, f"unexpected label: {category.label}") for category in categories]
            yield example.text, labels

    def _get_label_map(self, project_id):
        """
        Private method to map readable labels to label ids for specified or active project
        Required by get_labelled_samples
        """
        label_types = self.client.list_label_types(project_id=project_id, type='category')
        return {label_type.id: label_type.text for label_type in label_types}