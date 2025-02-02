from operator import itemgetter
from typing import List, Tuple
from dotenv import load_dotenv
load_dotenv()
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,)
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.load import dumps,loads
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,)
from langchain_community.vectorstores import Qdrant
# from langchain_cohere import CohereEmbeddings
from langchain_openai import ChatOpenAI
import qdrant_client,os
from langchain_qdrant import Qdrant
url = os.getenv("QDRANT_URL")
api_key = os.getenv("QDRANT_API_KEY")
client = qdrant_client.QdrantClient(
    url,
    api_key=api_key,)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# embeddings = HuggingFaceEmbeddings()

vectorstore = Qdrant(
    client=client, collection_name="my_documents1", 
    embeddings=embeddings,)

retriever = vectorstore.as_retriever(search_kwargs={'k': 15})

# Condense a chat history and follow-up question into a standalone question
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""  # noqa: E501
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

# RAG answer synthesis prompt
template = """Answer the question based only on the following context also maintain friendly tone:
<context>
{context}
</context>"""
ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{question}")])

def _combine_documents(docs):
    return '\n\n'.join(set(dumps(doc) for doc in docs))

def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    # buffer = []
    # for human, ai in chat_history:
    #     buffer.append(HumanMessage(content=human))
    #     buffer.append(AIMessage(content=ai))
    return chat_history

_search_query = RunnableBranch(
    # If input includes chat_history, we condense it with the follow-up question
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
            run_name="HasChatHistoryCheck"
        ),  # Condense follow-up question and chat into a standalone_question
        RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | ChatOpenAI(model="gpt-4o-mini",openai_api_key=os.getenv("OPENAI_API_KEY"),temperature=0) | StrOutputParser(),
    ),
    # Else, we have no chat history, so just pass through the question
    RunnableLambda(itemgetter("question")))

_inputs = RunnableParallel(
    {
        "question": lambda x: x["question"],
        "chat_history": lambda x: _format_chat_history(x["chat_history"]),
        "context": _search_query | retriever | _combine_documents,
    })

chain = _inputs | ANSWER_PROMPT | ChatOpenAI(model="gpt-4o-mini",openai_api_key=os.getenv("OPENAI_API_KEY"),temperature=0) | StrOutputParser()