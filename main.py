from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OllamaEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOllama, ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st

st.set_page_config(
    page_title="JEIL DOCS GPT",
    page_icon="📃",
)

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *arlgs, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


# 주석 둘 중 하나만 주석 해제할 것.


# 주석 1
#################
# 만약 chat gpt 유료버전 사용 중이라면 바로 아래 코드블록 주석 해제하고, openai_api_key 입력
#################
# llm = ChatOpenAI()(
#     openai_api_key="sk- api_key_입력",
#     model_name="gpt-3.5-turbo",
#     temperature=0.1,
#     streaming=True,
#     callbacks=[
#         ChatCallbackHandler(),
#     ]
# )
#
# embed = OpenAIEmbeddings(model="gpt-3.5-turbo")


# 주석 2
#################
# chat gpt 유료버전을 사용할 수 없는 상황이라면
# https://ollama.com/download 에서 다운로드 및 설치 후,
# https://ollama.com/library 에서 원하는 모델 (llama3:8b 추천,   만약 vram 8gb 이상의 nvidia 그래픽카드라면 llama3:70b 추천) 설치
# 다 되었다면 아래 코드블록 해제
#################
llm = ChatOllama(
    model = "llama3:8b",      # model = "llama3:70b"
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ]
)

embed = OllamaEmbeddings(model = "llama3:8b")     # (model = "llama3:70b")



@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = embed
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
            
            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)


st.title("hello jeil!")


with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )

if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            chain.invoke(message)


else:
    st.session_state["messages"] = []