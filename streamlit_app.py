import streamlit as st
import os
from dotenv import load_dotenv
from langchain import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
load_dotenv()
import time

os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as full sentence structure.
Always say "thanks for asking!" at the end of the answer.

### Content: 
{context}

### Question:
{question}

Helpful Answer:
"""

# App title
st.set_page_config(page_title="BLINKpedia Chatbot", page_icon='BLINKpedia.png')

# Replicate Credentials
with st.sidebar:
    st.image('BLINKpedia.png',)
    st.logo('BLINKpedia_open.png', icon_image='BLINKpedia_close.png')
    st.title('BLINKpedia Chatbot')
    st.subheader('Models and parameters')
    st.markdown('''
                This model is designed to generate text content related to BLACKPINK, a globally renowned K-pop girl group. It leverages state-of-the-art natural language processing techniques to produce coherent and contextually relevant text based on input prompts.
                ## Model Details
                - **Model Name**: [BLINKpedia](https://huggingface.co/la-min/BLINKpedia)
                - **Model Type**: Text Generation
                - **Training Data**: Curated datasets containing information about BLACKPINK, including lyrics, interviews, news articles, and fan content.
                - **Framework**: Hugging Face Transformers
                ## Contributors
                - [La Min Ko Ko](https://www.linkedin.com/in/la-min-ko-ko-907827205/)
                - [Kyu Kyu Swe](https://www.linkedin.com/in/kyu-kyu-swe-533718171/)
                ''')

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar= "🤖" if message["role"] != "user" else "🧠"):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

def BgeEmbedding():
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    hf = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    return hf

def generate_format_prompt(input):
     # Prepare the DB.
    embedding_function = BgeEmbedding()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_relevance_scores(input, k=4)

    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=input)
    
    return prompt
    

def generate_llama2_response(prompt_input):
    format_prompt = generate_format_prompt(prompt_input)
    llm =  HuggingFaceHub(repo_id="unsloth/tinyllama-chat", model_kwargs={"temperature":0.3,})
    output = llm.invoke(format_prompt)

    return output

def response_generator(txt):
    for word in txt.split():
        yield word + " "
        time.sleep(0.05)
        
def dynamic_waiting_message(elapsed_time):
    if elapsed_time <= 5:
        return "Thinking..."
    elif elapsed_time <= 10:
        return "The result is almost here..."
    elif elapsed_time <= 15:
        return "It's really coming out now..."
    else:
        return "Just a little longer..."

st.markdown(
    """
<style>
    .st-emotion-cache-1c7y2kd {
        flex-direction: row-reverse;
        text-align: right;
        background-color: transparent;
    }
    .st-emotion-cache-1v0mbdj img{
        border-radius: 20px;
    }
    .st-emotion-cache-1mi2ry5{
        align-items: center;
    }
</style>
""",
    unsafe_allow_html=True,
)
    
    
    
# Main execution
def main():
    start_time = time.time()
    
    # User-provided prompt
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧠"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner(dynamic_waiting_message(time.time() - start_time)):
                response = generate_llama2_response(prompt)
                answer_response = response.split("Helpful Answer:")[1]
            st.write_stream(response_generator(answer_response))
        message = {"role": "assistant", "content": answer_response}
        st.session_state.messages.append(message)

if __name__ == "__main__":
    main()