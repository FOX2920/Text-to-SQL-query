import streamlit as st
from langchain import HuggingFacePipeline, PromptTemplate, LLMChain
from transformers import AutoTokenizer
import torch
import transformers

# Import your Hugging Face token from config.py
from config import HUGGING_FACE_TOKEN

# Set up Hugging Face login
from huggingface_hub import login
login(HUGGING_FACE_TOKEN)

# Model name
model = "meta-llama/Llama-2-7b-chat-hf"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model)

# Set up text generation pipeline
pipeline = transformers.pipeline("text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_new_tokens=512,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id
                )

# Custom pipeline for text generation
llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature': 0})

# Define prompt template
template = """
         Create a SQL query snippet using the below text:
          ```{text}```
          Just SQL query:
       """
prompt = PromptTemplate(template=template, input_variables=["text"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

# Streamlit app
def main():
    st.title("Text to SQL Query Generator")

    # Input text area for user input
    text_input = st.text_area("Enter your text here:")

    # Button to generate SQL query
    if st.button("Generate SQL Query"):
        if text_input:
            # Invoke language model to generate SQL query
            sql_query = llm_chain.invoke(text_input)
            st.write("Generated SQL Query:")
            st.code(sql_query)
        else:
            st.warning("Please enter some text to generate SQL query.")

if __name__ == "__main__":
    main()
