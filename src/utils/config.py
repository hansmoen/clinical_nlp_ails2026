from dotenv import dotenv_values
import os
import sys
from pathlib import Path


"""
NB! Regarding API keys.
If you are using Google Colab, open 'Secrets' (key icon in the left sidebar) and
add a secret with 'Name': "MY_API_KEY" and 'Value': {your OpenAI key}.
Also make sure to enable "Notebook access" for that secret.
Otherwise, add a folder on root level named 'secrets', create a file inside
called keys.env, then add to it: MY_API_KEY={your key}
"""

SECRET_NAME = "MY_API_KEY"


def get_api_key():
    IN_COLAB = "google.colab" in sys.modules
    if IN_COLAB:
        from google.colab import userdata

        try:
            return userdata.get(SECRET_NAME)
        except Exception as err:
            raise RuntimeError(
                f'No usable Colab secret named "{SECRET_NAME}".\n'
                "Fix: open the Secrets panel (key icon, left sidebar), add a "
                f'secret named exactly "{SECRET_NAME}" holding your '
                'OpenAI API key, and switch on "Notebook access" for it.'
            ) from err
    else:
        project_root = Path(__file__).resolve().parents[2]
        secrets_file = os.path.join(project_root, "secrets", "keys.env")
        secrets = dotenv_values(secrets_file)
        if SECRET_NAME not in secrets:
            raise RuntimeError(
                f"No {SECRET_NAME} found in {secrets_file}\n"
                f"Fix: create that file with a line: {SECRET_NAME}=sk-..."
            )
        return secrets[SECRET_NAME]