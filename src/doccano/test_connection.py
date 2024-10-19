import os
from doccano_client import DoccanoClient
from dotenv import load_dotenv

load_dotenv()

client = DoccanoClient('http://localhost:8000')
client.login(username = os.getenv("DOCCANO_ADMIN_USERNAME"),
             password = os.getenv("DOCCANO_ADMIN_PASSWORD")
             )

user = client.get_profile()
print(user)