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


def create_ccc_index(df, emb_model, save_path, source):
    # source = "CCC_Diagnosis" or "CCC_Interventions"
    texts = []
    metadatas = []

    for _, row in df.iterrows():

        name = " ".join(str(row["Name"]).split())
        description = " ".join(str(row["Description"]).split())

        metadata = {
            "source": source,
            "comp_letter": row["CompLetter"],
            "code": row["Code"],
        }

        # Name
        texts.append(name)
        metadatas.append({
            **metadata,
            "content_type": "name",
        })

        # Description
        texts.append(description)
        metadatas.append({
            **metadata,
            "content_type": "description",
        })

        # Name + description
        texts.append(f"{name} -- {description}")
        metadatas.append({
            **metadata,
            "content_type": "name--description",
        })

    vector_store = FAISS.from_texts(
        texts=texts,
        embedding=emb_model,
        metadatas=metadatas,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )

    # Save FAISS index, documents, and metadata
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

    # Read the CCC taxonomy files, diagnosis and interventions
    ccc_diag_df = pd.read_csv(os.path.join(data_resources_dir, "CCC_Diagnosisv25-20120309_sorted.csv"), encoding="utf-8-sig")
    ccc_interv_df = pd.read_csv(os.path.join(data_resources_dir, "CCC_Interventionsv25-20109_sorted.csv"), encoding="utf-8-sig")

    ccc_diag_faiss_index_path = os.path.join(VEC_STORE_DIR, "ccc_diag_faiss_index"); ccc_diag_source = "CCC_Diagnosis"
    ccc_interv_faiss_index_path = os.path.join(VEC_STORE_DIR, "ccc_interv_faiss_index"); ccc_interv_source = "CCC_Interventions"


    # Create vector stores
    create_ccc_index(df=ccc_diag_df, emb_model=emb, save_path=ccc_diag_faiss_index_path, source=ccc_diag_source)
    create_ccc_index(df=ccc_interv_df, emb_model=emb, save_path=ccc_interv_faiss_index_path, source=ccc_interv_source)

    # Load existing vector stores
    ccc_diag_vector_store = load_index(filepath=ccc_diag_faiss_index_path, emb_model=emb)
    ccc_interv_vector_store = load_index(filepath=ccc_interv_faiss_index_path, emb_model=emb)

    query = "Patient is simply lazy"
    tok_k = 5
    print(f'\nTesting CCC semantic search. Query: "{query}"')
    print(f'Search in CCC_Diagnosis, top {tok_k}:')
    results_ccc_diag = query_index(ccc_diag_vector_store, query_str=query, top_k=tok_k, do_print=True)
    print(f'Search in CCC_Interventions, top {tok_k}:')
    results_ccc_interv = query_index(ccc_interv_vector_store, query_str=query, top_k=tok_k, do_print=True)
