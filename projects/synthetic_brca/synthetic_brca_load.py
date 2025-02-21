import json
from bioext.doccano_utils import DoccanoSession

from dotenv import load_dotenv

load_dotenv()

########################## DEFINE SET-UP VARIABLES ##########################
PROJECT_DETAILS = {
    "name": "Testing Class: Synthetic BRCA Classification",
    "description": "Classification of synthetic BRCA testing reports",
    "project_type": "DocumentClassification",
    "guideline": "Classify BRCA results under the most appropriate label(s)",
}
LABELS = ["BRCA1 positive", "BRCA2 positive", "BRCA1 VUS", "BRCA2 VUS", "Invalid"]
DATA_FILE = "data/brca_reports.json"
#############################################################################


def main():
    # connect and log on to doccano
    session = DoccanoSession()
    print(f"Connected to Doccano as user: {session.username}")

    # create project
    project = session.create_project(**PROJECT_DETAILS)
    print(f"Created new project: {project.name}, {project.id}")

    # set up labels
    session.setup_labels(LABELS)
    print(f"Created {len(LABELS)} labels")

    # load json from data file
    try:
        with open(DATA_FILE, "r") as file:
            data = json.load(file)
    except Exception as e:
        print(f"Failed to load samples: {str(e)}")
        return

    # load json to doccano
    session.load_simple_json(data)
    print(f"Uploaded {len(data)} examples")


if __name__ == "__main__":
    main()
    print("\nSetup complete")
