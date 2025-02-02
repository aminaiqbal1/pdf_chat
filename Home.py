from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_qdrant import QdrantVectorStore
# from langchain_cohere import CohereEmbeddings
from langchain_openai import OpenAIEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st 
import os,uuid
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Conversational AI RAG ChatBot", page_icon=":💬:", layout="wide")
st.header("Conversational AI RAG ChatBot 💬")

# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")


def get_pdf_text(file_path):
    loader = PyMuPDFLoader(file_path=file_path)
    pages = loader.load_and_split()
    return pages

def get_txt_text(file_path):
    loader = TextLoader(file_path)
    splits = loader.load_and_split()
    return splits

with st.sidebar:

    uploaded_files = st.file_uploader("Upload your file", type=['pdf'], accept_multiple_files=True)
    if st.button("Process"):
        Documents = []
        
        if uploaded_files:
            st.write("Files Loaded...")
            for uploaded_file in uploaded_files:
                try:
                    os.makedirs('data', exist_ok=True)
                    file_path = f"data/{uploaded_file.name}{uuid.uuid1()}"
                    with open(file_path,'wb') as fp:
                        fp.write(uploaded_file.read())

                    split_tup = os.path.splitext(uploaded_file.name)
                    file_extension = split_tup[1]

                    if file_extension == ".pdf":
                        Documents.extend(get_pdf_text(file_path))

                    elif file_extension == ".txt":
                        Documents.extend(get_txt_text(file_path))

                except Exception as e:
                    st.error(f"Error processing this file: {uploaded_file.name} {e}")
                finally:
                    os.remove(file_path)
        else:
            st.error("No file uploaded.")

        if Documents:
            st.write("Indexing Please Wait...")
            # st.write(Documents)
            # indexing part goes here
            try:
                url = os.getenv("QDRANT_URL")
                api_key = os.getenv("QDRANT_API_KEY")
                # embeddings = HuggingFaceEmbeddings()
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                qdrant = QdrantVectorStore.from_documents(
                Documents,
                embeddings,
                url=url,
                prefer_grpc=True,
                api_key=api_key,
                collection_name="my_documents1",
                )
                # qdrant = Qdrant.from_documents(
                #     Documents,
                #     embeddings,
                #     url=url,
                #     prefer_grpc=True,
                #     api_key=api_key,
                #     force_recreate=True,
                #     collection_name="my_documents1")
                
                st.write("Indexing Done")
                st.session_state["processtrue"] = True
                if "langchain_messages" in st.session_state:
                    st.session_state["langchain_messages"] = []

            except Exception as e:
                st.error(f"Error indexing: {e}")

if "processtrue" in st.session_state:
    from chat import main
    main()

else:
    st.info("Please Upload Your Files.")