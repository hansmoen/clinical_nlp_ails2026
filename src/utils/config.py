from dotenv import dotenv_values
import os
from pathlib import Path



def get_api_key():
    project_root = Path(__file__).resolve().parents[2]
    secrets_file = os.path.join(project_root, "secrets", "keys.env")
    secrets = dotenv_values(secrets_file)
    api_key = secrets["OPENAI_API_KEY"]
    return api_key

