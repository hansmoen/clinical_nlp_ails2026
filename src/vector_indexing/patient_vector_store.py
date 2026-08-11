from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from dotenv import dotenv_values
from pathlib import Path
import os
import pandas as pd
from src.utils import config



def init_embedding_model(embedding_model_name, api_key):
    emb_model = OpenAIEmbeddings(
        model=embedding_model_name,
        api_key=api_key,
    )
    return emb_model


def create_patient_notes_index(df, emb_model, save_path):
    texts = []
    metadatas = []

    for _, row in df.iterrows():

        case_id = row["case_id"]
        case_name = row["case_name"]
        day = row["day"]
        note_order = row["note_order"]

        note_title = str(row["note_title"]).strip()
        author = str(row["author"]).strip()
        note_text = str(row["note_text"]).strip()

        metadata = {
            "source": "Patient_Notes",
            "case_id": case_id,
            "case_name": case_name,
            "day": day,
            "note_order": note_order,
            "note_title": note_title,
            "author": author,
        }

        # One document per clinical note.
        text = (
            f"{note_title}.\n"
            f"Author: {author}.\n\n"
            f"{note_text}"
        )

        texts.append(text)
        metadatas.append(metadata)

    vector_store = FAISS.from_texts(
        texts=texts,
        embedding=emb_model,
        metadatas=metadatas,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )

    vector_store.save_local(save_path)

    return vector_store


def load_index(filepath, emb_model):
    vector_store = FAISS.load_local(
        filepath,
        emb_model,
        allow_dangerous_deserialization=True,
    )
    return vector_store


def query_index(vector_store, query_str, top_k=5, do_print=True):
    res = vector_store.similarity_search_with_score(query_str, k=top_k)
    if do_print:
        for doc, score in res:
            print(f'Sim: {score:.2f}. Content: "{' '.join(doc.page_content.split())}". Metadata: {doc.metadata}')
    return res


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    print('PROJECT_ROOT:', PROJECT_ROOT)
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    VEC_STORE_DIR = os.path.join(PROJECT_ROOT, "vector_stores")
    data_resources_dir = os.path.join(DATA_DIR, "resources")
    data_pat_cases_dir = os.path.join(DATA_DIR, "patients")

    ##############################################################
    api_key = config.get_api_key()
    embedding_model_name = "text-embedding-3-small"
    llm_name = "gpt-5.6-luna"
    ##############################################################

    emb = init_embedding_model(embedding_model_name, api_key)

    # Read the patient files
    pat1_df = pd.read_csv(os.path.join(data_pat_cases_dir, "Case1_Left_Ventricular_Fibroma.csv"), encoding="utf-8-sig")
    pat2_df = pd.read_csv(os.path.join(data_pat_cases_dir, "Case2_Postoperative_Chylous_Ascites.csv"), encoding="utf-8-sig")
    pat3_df = pd.read_csv(os.path.join(data_pat_cases_dir, "Case3_Multifocal_Schwannoma.csv"), encoding="utf-8-sig")
    pat4_df = pd.read_csv(os.path.join(data_pat_cases_dir, "Case4_Acute_Limb_Ischemia.csv"), encoding="utf-8-sig")
    pat5_df = pd.read_csv(os.path.join(data_pat_cases_dir, "Case5_Bronchopleural_Fistula.csv"), encoding="utf-8-sig")
    pat6_df = pd.read_csv(os.path.join(data_pat_cases_dir, "Case6_Drug-Induced_Intoxication_with_Anxiety_Attack.csv"), encoding="utf-8-sig")

    pat1_df = pat1_df.sort_values(["day", "note_order"])
    pat2_df = pat2_df.sort_values(["day", "note_order"])
    pat3_df = pat3_df.sort_values(["day", "note_order"])
    pat4_df = pat4_df.sort_values(["day", "note_order"])
    pat5_df = pat5_df.sort_values(["day", "note_order"])
    pat6_df = pat6_df.sort_values(["day", "note_order"])

    #pat_all_df = pd.concat([pat1_df, pat2_df, pat3_df, pat4_df, pat5_df, pat6_df], ignore_index=True)
    #pat_all_df = pat_all_df.sort_values(["case_id", "day", "note_order"]).reset_index(drop=True)

    pat1_faiss_index_path = os.path.join(VEC_STORE_DIR, "pat1_faiss_index")
    pat2_faiss_index_path = os.path.join(VEC_STORE_DIR, "pat2_faiss_index")
    pat3_faiss_index_path = os.path.join(VEC_STORE_DIR, "pat3_faiss_index")
    pat4_faiss_index_path = os.path.join(VEC_STORE_DIR, "pat4_faiss_index")
    pat5_faiss_index_path = os.path.join(VEC_STORE_DIR, "pat5_faiss_index")
    pat6_faiss_index_path = os.path.join(VEC_STORE_DIR, "pat6_faiss_index")

    """
    # Create vector stores
    create_patient_notes_index(df=pat1_df, emb_model=emb, save_path=pat1_faiss_index_path)
    create_patient_notes_index(df=pat2_df, emb_model=emb, save_path=pat2_faiss_index_path)
    create_patient_notes_index(df=pat3_df, emb_model=emb, save_path=pat3_faiss_index_path)
    create_patient_notes_index(df=pat4_df, emb_model=emb, save_path=pat4_faiss_index_path)
    create_patient_notes_index(df=pat5_df, emb_model=emb, save_path=pat5_faiss_index_path)
    create_patient_notes_index(df=pat6_df, emb_model=emb, save_path=pat6_faiss_index_path)
    """

    # Load existing vector stores
    patN_vector_store = load_index(filepath=pat4_faiss_index_path, emb_model=emb)


    query = "The patient slipped and needed some help due to fractured toe"
    tok_k = 5
    print(f'\nTesting semantic search. Query: "{query}"')
    print(f'Search in patients index, top {tok_k}:')
    results_icd10 = query_index(patN_vector_store, query_str=query, top_k=tok_k, do_print=True)

