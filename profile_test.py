import logging
import sys, json, os
import pathlib

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

os.environ["HF_TOKEN"] = "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"

from huggingface_hub import login
login(os.environ["HF_TOKEN"])

import faiss

# dimensions of "all-MiniLM-L6-v2"
d = 384
faiss_index = faiss.IndexFlatL2(d)

from llama_index.core import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
    ServiceContext,
    get_response_synthesizer,
    PromptTemplate,
)
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Set the model to use
from llama_index.core import Settings
from llama_index.llms.vllm import Vllm

llm = Vllm(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    dtype="float16",
    tensor_parallel_size=1,
    temperature=0,
    max_new_tokens=100,
    vllm_kwargs={
        "swap_space": 1,
        "gpu_memory_utilization": 0.8,
        "max_model_len": 4096,
    },
)

# Core settings 
Settings.embed_model = HuggingFaceEmbedding(
    model_name="all-MiniLM-L6-v2")
Settings.llm = llm
Settings.chunk_size = 512
Settings.chunk_overlap = 0

# Prompt template

new_qa_tmpl_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and the query below, "
    "answer the query in NOT MORE than 5 words. DO NOT OUTPUT "
    " UNNCESSARY WORDS AND DON'T REPEAT THE QUERY \n"
    "Query: {query_str}\n"
    "Answer: "
)
new_qa_tmpl = PromptTemplate(new_qa_tmpl_str)

def convert_to_txt_documents(json_file, save_path):
    with open(json_file, 'r') as f:
        data = json.load(f)

    count = 0
    texts = []
    for doc in data['context']:
        # join all doc['content'] into one string
        texts.append((doc['content']))
    
    file_name = json_file.split('.')[0] + '.txt'
    file_name = file_name.replace('contexts_200', 'documents')
    with open(file_name, 'a') as f:
        full_text = '\n'.join(texts)
        f.write(full_text)

def get_query(query_dir):
    count = 0
    query_list = []
    for i in range(count, 200):
        file = str(count) + '.json'
        with open(query_dir + file, 'r') as f:
            query = f.read()
            count += 1
            query_list.append(query)
    return query_list

def get_answers(answers_dir):
    count = 0
    answers_list = []
    for i in range(count, 200):
        file = str(count) + '.json'
        with open(answers_dir + file, 'r') as f:
            answers = f.read()
            count += 1
            answers_list.append(answers)
    return answers_list
        
def main():

    query_path = 'data/musique/queries_200/'
    answers_path = 'data/musique/answers_200/'
    contexts_path = 'data/musique/contexts_200/'

    save_path = 'data/musique/documents/'

    # Check if save_path is empty or does not exist
    if not os.path.exists(save_path) or not os.listdir(save_path):
        os.makedirs(save_path, exist_ok=True)
        for file in os.listdir(contexts_path):
            convert_to_txt_documents(contexts_path + file, save_path)

    documents = SimpleDirectoryReader("data/musique/documents/").load_data()

    logging.info("Loaded %d documents", len(documents))

    if not os.path.exists("./storage"):
        os.makedirs("./storage")
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context,
        )
        logging.info("Index built")

        index.storage_context.persist()
    
    else:
        vector_store = FaissVectorStore.from_persist_dir("./storage")
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir="./storage", 
        )
        index = load_index_from_storage(storage_context=storage_context,    
                        similarity_top_k=10)  

    response_synthesizer = get_response_synthesizer(response_mode="compact")
    query_engine = index.as_query_engine(llm = llm, response_synthesizer=response_synthesizer)

    prompts_dict_old = query_engine.get_prompts()
    logging.info("Old prompt: %s", prompts_dict_old)

    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": new_qa_tmpl}
    )

    prompts_dict_new = query_engine.get_prompts()
    logging.info("New prompt: %s", prompts_dict_new)

    query_list = get_query(query_path)
    answers_list = get_answers(answers_path)

    assert len(query_list) == len(answers_list)

    for idx, query in enumerate(query_list):
        response = query_engine.query(query)
        with open('outputs/musique.csv', 'a') as f:
            answer = answers_list[idx]
            answer = json.loads(answer)['answer']
            f.write(f"{str(response)};{answer}\n")
           

if __name__ == '__main__':
    main()
    



