from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from dotenv import dotenv_values
from pathlib import Path
import os
import pandas as pd
from src.utils import config



def init_embedding_model(embedding_model_name, api_key):
    embeddings = OpenAIEmbeddings(
        model=embedding_model_name,
        api_key=api_key,
    )
    return embeddings


def create_care_guideline_index(df, embeddings, save_path):
    texts = []
    metadatas = []

    for _, row in df.iterrows():

        identifier = str(row["identifier"]).strip()
        sortkey = str(row["sortkey"]).strip()
        language = str(row["language"]).strip()
        title = str(row["title"]).strip()
        content = str(row["content"]).strip()

        metadata = {
            "source": "Care_Guidelines",
            "identifier": identifier,
            "sortkey": sortkey,
            "language": language,
            "title": title,
            #"content": content,
        }
        """
        # Title only
        texts.append(title)
        metadatas.append({
            **metadata,
            "content_type": "title",
        })
        """
        # Title + content
        texts.append(' '.join(f"{title}:\n\n{content}".split()))
        metadatas.append({
            **metadata,
            "content_type": "title--content",
        })

    vector_store = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )

    vector_store.save_local(save_path)

    return vector_store


def load_index(filepath, embeddings):
    vector_store = FAISS.load_local(
        filepath,
        embeddings,
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

    embeddings = init_embedding_model(embedding_model_name, api_key)

    # Read the care guideline file
    ccg_faiss_index_path = os.path.join(VEC_STORE_DIR, "ccg_faiss_index")

    ccg_df = pd.read_csv(os.path.join(data_resources_dir, "Care_guidelines.csv"), encoding="utf-8-sig")

    # Create Care Guideline vector store
    #ccg_vector_store = create_care_guideline_index(df=ccg_df, embeddings=embeddings, save_path=ccg_faiss_index_path)

    # Load existing vector stores
    ccg_vector_store = load_index(filepath=ccg_faiss_index_path, embeddings=embeddings)


    query = "The patient slipped and needed some help due to fractured toe"
    tok_k = 5
    print(f'\nTesting semantic search. Query: "{query}"')
    print(f'Search in Care Guideline, top {tok_k}:')
    results_icd10 = query_index(ccg_vector_store, query_str=query, top_k=tok_k, do_print=True)

