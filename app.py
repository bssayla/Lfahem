import streamlit as st
import transformers
from torch import bfloat16, cuda

@st.cache(allow_output_mutation=True)
def load_model():
    model_id = 'meta-llama/Meta-Llama-3-8B-Instruct'
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )
    model = transformers.AutoModel.from_pretrained(model_id, config=bnb_config)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer, device

model, tokenizer, device = load_model()

st.title('My Model App')

# Get user input
user_input = st.text_input("Enter some text")

# Predict function
def predict(input_text):
    inputs = tokenizer.encode_plus(input_text, return_tensors='pt').to(device)
    outputs = model(**inputs)
    return outputs

# Show prediction
if st.button('Predict'):
    prediction = predict(user_input)
    st.write(prediction)