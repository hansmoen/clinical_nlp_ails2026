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


def create_icd10_index(df, emb_model, save_path):
    texts = []
    metadatas = []

    for _, row in df.iterrows():
        code = str(row["code"]).strip()
        kind = str(row["kind"]).strip()
        title = str(row["title"]).strip()

        parent_code = row["parent_code"]
        if pd.isna(parent_code):
            parent_code = None
        else:
            parent_code = str(parent_code).strip()

        metadata = {
            "source": "ICD10",
            "code": code,
            "kind": kind,
            "parent_code": parent_code,
        }
        """
        # Title only
        texts.append(title)
        metadatas.append({
            **metadata,
            #"content_type": "title",
        })
        """
        # Code + title
        texts.append(f"{code}: {title}")
        metadatas.append({
            **metadata,
            #"content_type": "code--title",
        })

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
            print(f'Sim: {score:.2f}. Content: "{doc.page_content}". Metadata: {doc.metadata}')
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

    # Read the ICD-10 taxonomy file
    icd10_df = pd.read_csv(os.path.join(data_resources_dir, "ICD10_clean.csv"), encoding="utf-8-sig")

    icd10_faiss_index_path = os.path.join(VEC_STORE_DIR, "icd10_faiss_index")


    # Create ICD-10 vector store
    create_icd10_index(df=icd10_df, emb_model=emb, save_path=icd10_faiss_index_path)

    # Load existing vector store
    icd10_vector_store = load_index(filepath=icd10_faiss_index_path, emb_model=emb)


    query = "The patient slipped"
    tok_k = 5
    print(f'\nTesting semantic search. Query: "{query}"')
    print(f'Search in ICD-10, top {tok_k}:')
    results_icd10 = query_index(icd10_vector_store, query_str=query, top_k=tok_k, do_print=True)

