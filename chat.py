from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,)
from langchain.schema.messages import HumanMessage,AIMessage

def main():
    prompt = ChatPromptTemplate(
        messages=[
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{question}")])

    msgs = StreamlitChatMessageHistory(key="langchain_messages")

    if len(msgs.messages) == 0:
        msgs.add_ai_message("Hello! How can I assist you today?")

    
    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)
    
    if prompt := st.chat_input():
        st.chat_message("human").write(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                from chain import chain
                _chat_history = st.session_state.langchain_messages[1:40]
                response = chain.stream({"question":prompt,"chat_history":_chat_history})

                for res in response:
                    full_response += res or "" 
                    message_placeholder.markdown(full_response + "|")
                    message_placeholder.markdown(full_response)

                msgs.add_user_message(prompt)
                msgs.add_ai_message(full_response)

            except Exception as e:
                st.error(f"An error occured. {e}")