from bioext.doccano_utils import DoccanoSession

from dotenv import load_dotenv

load_dotenv()

########################## DEFINE SET-UP VARIABLES ##########################
PROJECT_ID = 6
#############################################################################


def main():
    session = DoccanoSession()
    print(f"Connected to Doccano as user: {session.username}")

    # iterator
    labelled_samples = session.get_labelled_samples(PROJECT_ID)

    # print labelled samples
    for i, (text, labels) in enumerate(labelled_samples, 1):
        print(f"\nSample {i}:")
        print(f"Text: {text[:100]}...")
        print(f"Labels: {labels}")


if __name__ == "__main__":
    main()
    print("\nProcessing complete")
