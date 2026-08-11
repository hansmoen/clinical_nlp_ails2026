from dotenv import dotenv_values
import os
import sys
from pathlib import Path


def get_api_key():
    IN_COLAB = "google.colab" in sys.modules
    if IN_COLAB:
        from google.colab import userdata
        api_key = userdata.get('api_key_1')
        return api_key
    else:
        project_root = Path(__file__).resolve().parents[2]
        secrets_file = os.path.join(project_root, "secrets", "keys.env")
        secrets = dotenv_values(secrets_file)
        api_key = secrets["OPENAI_API_KEY"]
        return api_key

