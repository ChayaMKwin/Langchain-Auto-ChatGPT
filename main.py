import os
from apikey import apikey
import streamlit as st
from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

os.environ['OPEN_API_KEY'] = apikey

# App framework
st.title('YouTube GPT creator')
prompt = st.text_input('Add your prompt here')

# Prompt Tempates
title_template = PromptTemplate(
    input_variable=['topic', 'wikipedia_research'],
    template='Write me a YouTube video title about {topic}, while leveraging this Wikipedia research: {wikipedia_research}'
)

script_template = PromptTemplate(
    input_variable=['topic'],
    template='Write me a YouTube video script based on title TITLE: {title}, while leveraging this Wikipedia research: {wikipedia_research}'
)

# Memory
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

wiki = WikipediaAPIWrapper()

# Llms
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)
sequential_chain = SequentialChain(chains=[title_chain, script_chain], input_variables=['topic'], output_variables=['title', 'script'], verbose=True)

# Show in Screen
if prompt:
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title=title, wikipedia_research=wiki_research)
    st.write(title)
    st.write(script)

    with st.expander('Title History'):
        st.info(title_memory.buffer)

    with st.expander('Script History'):
        st.info(script_memory.buffer)

    with st.expander('Wikipedia Research'):
        st.info(wiki_research)
