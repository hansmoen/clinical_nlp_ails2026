from dotenv import dotenv_values
import os
import sys
from pathlib import Path


"""
NB! Regarding API keys.
If you are using Google Colab, open 'Secrets' and add a key with 'Name': "MY_API_KEY" and 'Value': {your key}.
Also make sure to also enable "Notebook access".
Otherwise, add a folder on root level named 'secrets', create a file inside called keys.env, then add to it: "OPENAI_API_KEY={your key}".
"""

def get_api_key():
    IN_COLAB = "google.colab" in sys.modules
    if IN_COLAB:
        from google.colab import userdata
        api_key = userdata.get('MY_API_KEY')
        return api_key
    else:
        project_root = Path(__file__).resolve().parents[2]
        secrets_file = os.path.join(project_root, "secrets", "keys.env")
        secrets = dotenv_values(secrets_file)
        api_key = secrets["MY_API_KEY"]
        return api_key

