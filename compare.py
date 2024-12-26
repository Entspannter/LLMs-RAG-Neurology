from dotenv import load_dotenv
from PyPDF2 import PdfReader
import chromadb
import json
from fastapi.encoders import jsonable_encoder
from pathlib import Path
import openai
import logging
import os
import re
import hydra
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.chains import RetrievalQA
import datetime

os.environ["HYDRA_FULL_ERROR"] = "1"
logging.getLogger().setLevel(logging.INFO)


def load_docs_from_src(directory: Path):
    # Load each PDF document seperately

    docs = {}
    for doc_p in Path(directory).rglob("*.pdf"):
        doc_str = str(doc_p)
        try:
            split = doc_str.rsplit("_")
            (association, _) = split[-2].rsplit("/")[-1], split[-1].split(".")[0]
            assert association in {
                "AAN",
                "DGN",
            }, "The document naming convention has been violated. The expected format is 'ASSOCIATION_ENTITY.pdf'. For example: 'ASCO_CRC.pdf'."
        except Exception as e:
            raise NameError("Invalid document name.") from e

        l = PyPDFLoader(doc_str)
        txt = l.load()

        docs.setdefault(association, []).extend(txt)

    return docs


def get_chunks_per_pdf(doc_dict, chunk_size, overlap):
    # Store document chunks in a dict, where each key is one identifier for 1 PDF
    chunks = {}
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap, length_function=len
    )

    for key, doc in doc_dict.items():
        chunks[key] = text_splitter.split_documents(doc)

    return chunks


def get_vectorstore_per_pdf(chunk_dict, chunk_size, overlap):
    # Store each PDF in a separated Vectorstore object
    vs_dict = {}
    embeddings = OpenAIEmbeddings()

    for key, doc_chunks in chunk_dict.items():
        entity = (
            doc_chunks[0].metadata["source"].split("/")[-1].split(".")[0].split("_")[1]
        )

        valid_entity_names = "Migraine", "PancreaticCancer", "HCC"
        pattern = re.compile(r"^(%s)" % "|".join(valid_entity_names))
        match = pattern.match(entity)
        entity = match.group(1)

        index = Path(f"./chroma_db/{key}/{entity}_{chunk_size}_{overlap}")

        client_settings = chromadb.config.Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(index),
            anonymized_telemetry=False,
        )

        if index.exists():
            try:
                vectorstore = Chroma(
                    persist_directory=index,
                    embedding_function=embeddings,
                    client_settings=client_settings,
                )
                logging.info(f"Loading existing chroma database from {index}.")

            except Exception as e:
                vectorstore = Chroma.from_documents(
                    doc_chunks,
                    embeddings,
                    persist_directory=str(index),
                    client_settings=client_settings,
                )
                vectorstore.persist()
                logging.info(f"Failed loading existing database from {index}.")

        else:
            vectorstore = Chroma.from_documents(
                doc_chunks,
                embeddings,
                persist_directory=str(index),
                client_settings=client_settings,
            )
            vectorstore.persist()
            logging.info(f"Index not existing. Creating new database at {index}.")

        vs_dict[key] = vectorstore

    return vs_dict


def compare(
    vectorstores,
    question,
    model=None,
    use_rag=True,
):
    # Compare the input from 2 or more documents

    llm = ChatOpenAI(temperature=0, model=model)
    human_message_prompt = HumanMessagePromptTemplate.from_template("{question}")
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        """You are an AI medical assistant specializing in neurology. Based on the provided neurology guidelines, provide detailed and truthful information in response to inquiries from a medical doctor. Ensure your responses are:
            - Relevant to the given context.
                For instance, when asked about chemoradiation, do not include information about chemotherapy alone.
            - Presented in concise bullet points.
            - Honest, especially when the answer isn't available in the guidelines. 
            - Include citations and references.
            - As detailed as possible. Include all details regarding patient and disease characteristics like disease type, disease stage and patient age
            - Include references to clinical trials (by name and not by number), survival data, exact treatments, their outcomes and details on the trial design. 

        Context:
        {context}

        Based on the American neurology guidelines, what does the association say about the topic/question presented in the context?
        """
    )

    chain_res = {}
    for key, vectorstore in vectorstores.items():
        if use_rag:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 25})
        else:
            retriever = None

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": ChatPromptTemplate.from_messages(
                    [
                        system_message_prompt,
                        human_message_prompt,
                    ]
                )
            },
        )

        result = qa_chain({"query": question})
        chain_res[key] = result

    def format_dict(data_dict):
        output = []
        for key, value in data_dict.items():
            output.append(f"{key}:\n{value['result']}\n")
        return "\n".join(output)

    response_str = format_dict(chain_res)

    input_prompt = """You are a dedicated AI medical assistant specializing in neurology. Your answers are based on the provided AAN guidelines and are strictly truthful. If the information is not available in the guidelines, state it clearly. Refrain from including irrelevant or out-of-context details.
        Please respond to the following question using information extracted from the AAN guidelines. Focus on providing answers that are:
            - Directly linked to the guidelines, citing specific sections or page numbers when possible (applicable only if RAG is enabled).
            - Presented in concise bullet points.
            - Comprehensive, including all relevant details about disease type, disease stage, patient age, and clinical trials when applicable.
            - Honest, clearly stating if the information is not covered in the guidelines.
            - Including citations to the guidelines or other reputable sources as needed.
        
        Question:
        {question}

        Based on the AAN guidelines, provide a detailed and truthful response that addresses the specifics of the question.
        Ensure all relevant medical and scientific details are included to support your answer."""

    completion = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": input_prompt},
            {"role": "user", "content": response_str},
        ],
    )

    return completion, chain_res


def save_complete(user_question, vectorstores, model_name, chunk_size, overlap):

    completion, chain_res = compare(vectorstores, user_question, model=model_name)
    ai_message = [jsonable_encoder(completion["choices"][0]["message"]["content"])]
    hu_message = [jsonable_encoder(user_question)]
    source_docs = [jsonable_encoder(v["source_documents"] for v in chain_res.values())]

    with open(f"{model_name}_outputs.json", "a") as f:
        json.dump(
            {
                "Human Message": hu_message,
                "AI Response": ai_message,
                "source documents": source_docs,
                "# timestamp": datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                "# chunk_size": chunk_size,
                "# overlap": overlap,
            },
            f,
            indent=4,
        )

    print(completion["choices"][0]["message"]["content"])


@hydra.main(version_base="1.3", config_path="conf", config_name="migraine_config.yaml")
def main(cfg):

    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    documents_dir = Path(cfg.documents_dir)

    if not documents_dir.exists():
        raise NotADirectoryError(f"Directory at {cfg.documents_dir} does not exist.")

    docs_dict = load_docs_from_src(documents_dir)
    chunks_dict = get_chunks_per_pdf(
        docs_dict, chunk_size=cfg.chunk_size, overlap=cfg.overlap
    )
    vs_dict = get_vectorstore_per_pdf(
        chunks_dict, chunk_size=cfg.chunk_size, overlap=cfg.overlap
    )

    # save_complete(cfg.user_question, vs_dict, cfg.model_name)
    save_complete(
        "MSI", vs_dict, cfg.model_name, chunk_size=cfg.chunk_size, overlap=cfg.overlap
    )


if __name__ == "__main__":
    main()
