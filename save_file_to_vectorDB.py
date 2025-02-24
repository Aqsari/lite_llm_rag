import PyPDF2
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Set your Hugging Face token
os.environ['HUGGINGFACE_TOKEN'] = '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'

def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

def create_vector(text):
    model = SentenceTransformer('all-MiniLM-L6-v2', use_auth_token=os.getenv('HUGGINGFACE_TOKEN'))
    vector = model.encode(text)
    return vector

def save_vector_to_faiss(vector, index_path='faiss_index.index'):
    vector_np = np.array([vector], dtype='float32')
    dimension = vector_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vector_np)
    faiss.write_index(index, index_path)

# Contoh penggunaan
file_path = '2409.14924v1.pdf'
text = read_pdf(file_path)
vector = create_vector(text)
save_vector_to_faiss(vector)


def load_faiss_index(index_path='faiss_index.index'):
    # Load the FAISS index from the specified file
    index = faiss.read_index(index_path)
    return index

def check_index(index):
    # Check the number of vectors in the index
    num_vectors = index.ntotal
    print(f"Number of vectors in the index: {num_vectors}")
    return num_vectors

def retrieve_vector(index, vector_id):
    # Retrieve a specific vector by its ID
    vector = index.reconstruct(vector_id)
    return vector

# Example usage
index_path = 'faiss_index.index'
index = load_faiss_index(index_path)

# Check the number of vectors in the index
num_vectors = check_index(index)

# If there are vectors, retrieve and print the first vector
if num_vectors > 0:
    first_vector = retrieve_vector(index, 0)  # Retrieve the first vector
    print("First vector:", first_vector)