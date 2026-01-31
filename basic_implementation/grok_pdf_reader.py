from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader

from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_xai import ChatXAI

load_dotenv()

####################################### Text Loader

def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text

def read_documents_from_directory(directory):
    combined_text = ''
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.endswith('.pdf'):
            combined_text += read_pdf(file_path)

    return combined_text

#######################################

# Resolve path relative to this script so it works from any cwd
_script_dir = os.path.dirname(os.path.abspath(__file__))
train_directory = os.path.join(_script_dir, 'train_files')
text = read_documents_from_directory(train_directory)

# Chunking
char_text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200, length_function=len)

text_chunks = char_text_splitter.split_text(text)

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

docsearch = FAISS.from_texts(text_chunks, embeddings)

# Grok (xAI)
llm = ChatXAI(
    model="grok-4",
    temperature=0,
)

chain = load_qa_chain(llm, chain_type='stuff')

#######################################

query = "ENTER YOUR QUERY HERE"

docs = docsearch.similarity_search(query)

response = chain.run(input_documents = docs, question=query)
print(' ')
print(query)
print(response)
