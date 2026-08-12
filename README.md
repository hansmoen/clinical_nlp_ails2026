# Clinical NLP — AILS 2026

A RAG-based prototype for querying clinical notes, care guidelines, the CCC
taxonomy (diagnoses and interventions) and ICD-10 codes. Prebuilt FAISS indices
are included, so no indexing step is required to get started.

---

## Setup for students (Google Colab)

Steps 1–3 are done **once**. After that, opening the notebook takes seconds.

You need your own OpenAI API key — the notebook makes embedding and chat calls
against it.

### 1. Copy the project into your Google Drive

Open a blank Colab notebook (<https://colab.research.google.com> → *New
notebook*), paste this into a cell, and run it:

```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive
!git clone https://github.com/OWNER/REPO.git clinical_nlp_ails2026
```

Approve the Drive authorisation popup when it appears. The download is about
80 MB (mostly the prebuilt indices) and takes a minute or two.

You should end up with `MyDrive/clinical_nlp_ails2026`. You can now close this
scratch notebook.

*Not using git?* Download the project as a ZIP, unzip it, and upload the
resulting folder to the top level of your Drive. Either `clinical_nlp_ails2026`
or `clinical_nlp_ails2026-main` will be found automatically.

### 2. Add your API key to Colab Secrets

Do **not** paste your key into a code cell.

1. Click the **key icon** (🔑) in the Colab left sidebar to open *Secrets*.
2. Click **Add new secret**.
3. Name: `api_key_1` — exactly that, lowercase.
4. Value: your OpenAI API key (begins with `sk-`).
5. Switch on **Notebook access** for the secret.

Secrets live on your Google account rather than in the notebook, so this stays
private even if you share the file.

### 3. Open the notebook from Drive

In Drive, open the `clinical_nlp_ails2026` folder, right-click
`clinical_nlp_ails2026.ipynb` → **Open with** → **Google Colaboratory**.

Open it from Drive, not from GitHub — that way your edits and outputs save
automatically.

### 4. Run it

Run the cells from the top, in order. The first four handle setup:

| Cell | What it does |
| ---- | ------------ |
| 1 | Mounts your Drive — approve the popup |
| 2 | Locates the project and prints `PROJECT_ROOT` |
| 3 | Installs dependencies from `requirements.txt` (30–60s) |
| 4 | Imports libraries and reads your API key |

**Re-run these four cells at the start of every session.** Colab discards
installed packages and the Drive mount when the runtime disconnects (after
roughly 90 minutes idle, or a while after you close the tab). Your files in
Drive are unaffected — only the running machine resets.

---

## Troubleshooting

**`No usable Colab secret named "api_key_1"`**
The secret name is wrong, or *Notebook access* is off. Re-check step 2.

**`FileNotFoundError: Could not find the project in your Google Drive`**
The folder isn't at the top level of MyDrive, or has a different name. Check it
in Drive; if you want to keep a custom name, add it to `CANDIDATE_DIR_NAMES` in
cell 2.

**`ModuleNotFoundError` (`langchain`, `faiss`, `src`, …)**
The runtime restarted. Re-run cells 1–4 from the top.

**`NameError: name 'IN_COLAB' is not defined`**
Cell 1 was skipped. Run from the top.

**Drive mount hangs or the popup never appears**
Relax strict cookie/tracker blocking for `colab.research.google.com`, or use a
regular (non-incognito) window.

**Loading feels slow**
Reading from mounted Drive is slower than local disk. Expected — the indices
load once per session.

---

## Running locally

```bash
pip install -r requirements.txt
mkdir secrets && echo "OPENAI_API_KEY=sk-..." > secrets/keys.env
jupyter notebook clinical_nlp_ails2026.ipynb
```

Outside Colab, `PROJECT_ROOT` is the current working directory, so start Jupyter
from the project root.

## Layout

```
data/patients/     Six synthetic patient cases (clinical notes as CSV)
data/resources/    CCC diagnoses & interventions, ICD-10, care guidelines
src/rag/           Simple RAG question-answering
src/utils/         API key handling
src/vector_indexing/  Scripts that built the FAISS indices
vector_stores/     Prebuilt FAISS indices — no need to rebuild
```
