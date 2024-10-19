import os
from doccano_client import DoccanoClient

client = DoccanoClient('http://localhost:8000')
client.login(username = os.getenv("DOCCANO_ADMIN_USERNAME"),
             password = os.getenv("DOCCANO_ADMIN_PASSWORD")
             )

user = client.get_profile()
print(user)