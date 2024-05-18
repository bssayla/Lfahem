import streamlit as st
import transformers
from torch import bfloat16, cuda
from langchain.llms import HuggingFacePipeline
from time import time
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from IPython.display import display, Markdown
from transformers import AutoTokenizer
import torch


def load_model():
    model_id = 'meta-llama/Meta-Llama-3-8B-Instruct'
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    # bnb_config = transformers.BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type='nf4',
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=bfloat16
    # )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id,use_auth_token=st.secrets["hf_tkn"])
    return tokenizer, device

def predict(input_text, model, tokenizer, device):
    inputs = tokenizer.encode_plus(input_text, return_tensors='pt').to(device)
    outputs = model(**inputs)
    return outputs

def test_rag(qa, query):

    time_start = time()
    response = qa.run(query)
    time_end = time()
    total_time = f"{round(time_end-time_start, 3)} sec."

    full_response =  f"Question: {query}\nAnswer: {response}\nTotal time: {total_time}"
    return full_response

def main():
    model_id = 'meta-llama/Meta-Llama-3-8B-Instruct'

    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )

    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=True,
        max_new_tokens=1024,
        use_auth_token=st.secrets["hf_tkn"]
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        use_auth_token=st.secrets["hf_tkn"]
    )
    # tokenizer = AutoTokenizer.from_pretrained(model_id,use_auth_token=st.secrets["hf_tkn"])
    tokenizer, device = load_model()
    
    query_pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            max_length=1024,
            device_map="auto",)

    


    
    loader = TextLoader("output.txt")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_splits = text_splitter.split_documents(documents)
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cuda"}

    # try to access the sentence transformers from HuggingFace: https://huggingface.co/api/models/sentence-transformers/all-mpnet-base-v2
    try:
        embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
    except Exception as ex:
        print("Exception: ", ex)
        # alternatively, we will access the embeddings models locally
        local_model_path = "/kaggle/input/sentence-transformers/minilm-l6-v2/all-MiniLM-L6-v2"
        print(f"Use alternative (local) model: {local_model_path}\n")
        embeddings = HuggingFaceEmbeddings(model_name=local_model_path, model_kwargs=model_kwargs)
    vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory="chroma_db")
    retriever = vectordb.as_retriever()


    llm = HuggingFacePipeline(pipeline=query_pipeline)
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        verbose=True
    )
    st.title('My Model App')

    # Get user input
    user_input = st.text_input("Enter some text")

    if st.button('Predict'):
        # prediction = test_rag(qa,user_input)
        time_start = time()
        response = llm(prompt=user_input)
        time_end = time()
        total_time = f"{round(time_end-time_start, 3)} sec."
        full_response =  f"Question: {user_input}\nAnswer: {response}\nTotal time: {total_time}"
        st.write(full_response)

    

    


if __name__ == "__main__":
    main()