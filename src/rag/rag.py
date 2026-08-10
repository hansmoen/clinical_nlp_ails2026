from pathlib import Path
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from src.utils import config
from textwrap import dedent


# ============================================================
# Vector stores
# ============================================================

def load_vector_stores(vec_store_dir, embeddings, allow_dangerous_deserialization=True):
    vector_stores = {
        "ccc_diag": FAISS.load_local(
            os.path.join(vec_store_dir, "ccc_diag_faiss_index"),
            embeddings,
            allow_dangerous_deserialization=allow_dangerous_deserialization,
        ),

        "ccc_interv": FAISS.load_local(
            os.path.join(vec_store_dir, "ccc_interv_faiss_index"),
            embeddings,
            allow_dangerous_deserialization=allow_dangerous_deserialization,
        ),

        "ccg": FAISS.load_local(
            os.path.join(vec_store_dir, "ccg_faiss_index"),
            embeddings,
            allow_dangerous_deserialization=allow_dangerous_deserialization,
        ),

        "icd10": FAISS.load_local(
            os.path.join(vec_store_dir, "icd10_faiss_index"),
            embeddings,
            allow_dangerous_deserialization=allow_dangerous_deserialization,
        ),

        "patient_notes": FAISS.load_local(
            os.path.join(vec_store_dir, "pat_all_faiss_index"),
            embeddings,
            allow_dangerous_deserialization=allow_dangerous_deserialization,
        ),
    }

    return vector_stores



# ============================================================
# Query rewriting
# ============================================================

def rewrite_query(query, llm, history=None):
    """
    Rewrite the user's query into a standalone query
    suitable for semantic retrieval.

    If history is provided, it is used to resolve
    references such as "this", "it", "the condition",
    etc.

    The function returns only the rewritten query.
    """

    if history is None:
        history = []

    # --------------------------------------------------------
    # Format conversation history
    # --------------------------------------------------------

    if history:
        history_parts = []

        for message in history:
            role = message["role"]
            content = message["content"]

            history_parts.append(
                f"{role.upper()}: {content}"
            )

        history_text = "\n".join(history_parts)

    else:
        history_text = "(No conversation history.)"

    # --------------------------------------------------------
    # System prompt
    # --------------------------------------------------------

    system_prompt = dedent("""
        You rewrite user questions into standalone search queries
        for a clinical information retrieval system.

        Your task is NOT to answer the question.

        Rewrite the user's latest question so that it can be understood
        without the conversation history.

        Use the conversation history to resolve references such as:

        - "it"
        - "this"
        - "that"
        - "the condition"
        - "the diagnosis"
        - "the patient"
        - "what about..."
        - "what is its code?"

        Preserve the clinical meaning of the user's question.

        Do not add information that is not present in the
        user's question or conversation history.

        Return ONLY the rewritten search query.
    """).strip()

    # --------------------------------------------------------
    # User prompt
    # --------------------------------------------------------

    user_prompt = dedent(f"""
        Conversation history:

        {history_text}

        Latest user question:

        {query}
    """).strip()

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    response = llm.invoke([
        (
            "system",
            system_prompt,
        ),
        (
            "human",
            user_prompt,
        ),
    ])

    return response.content.strip()


# ============================================================
# Retrieval
# ============================================================

def retrieve(query, vector_store_names, vector_stores, top_k_per_store=5):
    """
    Retrieve documents from one or more vector stores.
    """

    results = []

    for store_name in vector_store_names:

        if store_name not in vector_stores:
            raise ValueError(
                f"Unknown vector store: {store_name}"
            )

        vector_store = vector_stores[store_name]

        docs_with_scores = (
            vector_store.similarity_search_with_score(
                query,
                k=top_k_per_store,
            )
        )

        for doc, score in docs_with_scores:

            results.append({
                "store": store_name,
                "score": float(score),
                "document": doc,
            })

    return results


# ============================================================
# Context
# ============================================================

def build_context(results):
    context_parts = []

    for i, result in enumerate(
        results,
        start=1,
    ):
        doc = result["document"]

        context = dedent(f"""
            SOURCE {i}

            Vector store:
            {result["store"]}

            Similarity score:
            {result["score"]:.4f}

            Metadata:
            {doc.metadata}

            Content:
            {doc.page_content}
        """).strip()

        context_parts.append(context)

    return "\n\n".join(context_parts)


# ============================================================
# Answer generation
# ============================================================

def generate_answer(
    query,
    context,
    llm,
    history=None,
):
    """
    Generate an answer using the retrieved context.

    Conversation history is optional and can be included
    to make the response conversational.
    """

    if history is None:
        history = []

    history_text = ""

    if history:
        history_parts = []

        for message in history:

            role = message["role"]
            content = message["content"]

            history_parts.append(
                f"{role.upper()}: {content}"
            )

        history_text = "\n".join(
            history_parts
        )

    else:
        history_text = (
            "(No conversation history.)"
        )

    system_prompt = dedent("""
        You are a clinical information assistant.

        Answer the user's question using the retrieved context.

        The conversation history may be used to understand
        the user's question and maintain conversational continuity,
        but clinical claims must be supported by the retrieved context.

        If the retrieved context does not contain enough information
        to answer the question, say that the information is not
        available in the retrieved context.

        Do not invent diagnoses, codes, treatments, or clinical facts.

        When appropriate, identify the source of your answer.
    """).strip()

    user_prompt = dedent(f"""
        Conversation history:

        {history_text}

        Current question:

        {query}

        Retrieved context:

        {context}
    """).strip()

    response = llm.invoke([
        (
            "system",
            system_prompt,
        ),
        (
            "human",
            user_prompt,
        ),
    ])

    return response.content


# ============================================================
# RAG
# ============================================================

def rag_q_and_a(
    query,
    vector_store_names,
    vector_stores,
    llm,
    top_k=5,
    history=None,
    rewrite_query_for_retrieval=True
):
    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    if history is None:
        history = []

    # --------------------------------------------------------
    # Query rewriting
    # --------------------------------------------------------

    if rewrite_query_for_retrieval:
        rewritten_query = rewrite_query(query=query, llm=llm, history=history)
    else:
        rewritten_query = query

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    results = retrieve(
        query=rewritten_query,
        vector_store_names=vector_store_names,
        vector_stores=vector_stores,
        top_k_per_store=top_k,
    )

    # --------------------------------------------------------
    # Context
    # --------------------------------------------------------

    context = build_context(results)

    # --------------------------------------------------------
    # Answer
    # --------------------------------------------------------

    answer = generate_answer(
        query=query,
        context=context,
        llm=llm,
        history=history,
    )

    return answer, results, rewritten_query


# ============================================================
# Debugging
# ============================================================

def print_results(results):
    for i, result in enumerate(results, start=1):
        doc = result["document"]

        print("=" * 80)
        print(f"Result {i}")
        print(f"Store: {result['store']}")
        print(f"Score: {result['score']}")
        print(f"Metadata: {doc.metadata}")
        print(f"Content:\n{doc.page_content}")



if __name__ == "__main__":
    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    VEC_STORE_DIR = os.path.join(PROJECT_ROOT, "vector_stores")
    SECRETS_FILE = os.path.join(PROJECT_ROOT, "secrets", "keys.env")

    # ---------------------------------------------------------
    # Initialize models
    # ---------------------------------------------------------
    api_key = config.get_api_key()

    embedding_model_name = "text-embedding-3-small"
    llm_name = "gpt-5.6-luna"

    embeddings = OpenAIEmbeddings(
        model=embedding_model_name,
        api_key=api_key,
    )

    llm = ChatOpenAI(
        model=llm_name,
        api_key=api_key,
    )

    # --------------------------------------------------------
    # Load vector stores
    # --------------------------------------------------------
    vector_stores = load_vector_stores(
        vec_store_dir=VEC_STORE_DIR,
        embeddings=embeddings,
    )

    print(
        "Loaded vector stores:",
        list(vector_stores.keys()),
    )


    search_vector_store_names = [
        "patient_notes",
        "icd10",
        "ccc_diag",
    ]
    top_k = 5

    history = []

    query = """
    The patient is having fever, coughing, and yellow mucus.

    What diagnosis could explain the patient's symptoms,
    and which ICD-10 code is most relevant?
    """

    # --------------------------------------------------------
    # Query loop
    # --------------------------------------------------------
    print("\nClinical RAG system")
    print("Type 'exit' to quit.\n")

    while True:
        query = input(">>> Question: ").strip()

        if query.lower() in ["exit", "quit", "q"]:
            print("Stopping.")
            break

        if query.lower() in ["/h", "/hist", "/history"]:
            print("\nChat history:")
            print(history)
            print()
            continue

        if not query:
            continue

        answer, results, rewritten_query = rag_q_and_a(
            query=query,
            vector_store_names=search_vector_store_names,
            vector_stores=vector_stores,
            llm=llm,
            top_k=5,
            history=history,
        )

        print("\nRewritten index search query:")
        print(rewritten_query)

        print("\nAnswer:")
        print(answer)
        print()

        # ----------------------------------------------------
        # Update conversation history
        # ----------------------------------------------------
        history.append({
            "role": "user",
            "content": query,
        })

        history.append({
            "role": "assistant",
            "content": answer,
        })
