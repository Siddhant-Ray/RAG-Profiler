import logging
import sys, json, os
import pathlib
import yaml
import time
from datasets import load_dataset
import requests
from huggingface_hub import login
import faiss
import pandas as pd, numpy as np
from llama_index.core import PromptHelper
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# Set the model to use
from llama_index.core import Settings
from llama_index.llms.vllm import Vllm
from utils import Scorer
### ONLINE MODEL
from llama_index.llms.vllm import VllmServer
from llama_index.core.llms import ChatMessage
import json
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
    ServiceContext,
    get_response_synthesizer,
    PromptTemplate,
    Document
)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
login(os.environ["HF_TOKEN"])
# Test yaml

# Refactor later
config_path = "configs/central.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# dimensions of "all-MiniLM-L6-v2"
d = config['embedding_dim']
faiss_index = faiss.IndexFlatL2(d)
llm = OpenAILike(model="mistralai/Mistral-7B-Instruct-v0.3", 
            api_base=f"http://localhost:{config['port']}/v1", api_key="fake")

# Core settings 
Settings.embed_model = HuggingFaceEmbedding(
    model_name=config['embed_model'])
Settings.llm = llm
Settings.chunk_size = config['chunk_size']
Settings.chunk_overlap = config['chunk_overlap']
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

def parser_answer(text):
    import re
    # Keep only the last line 
    text = text.split('\n')[-1]
    # Keep only text after "Answer: "
    text = re.sub(r'Answer: ', '', text).strip()
    return text    

def get_ttft_from_query_engine(query_engine, query):
    start_time = time.time()
    # Variable to store the time of the first token
    first_token_time = None
    # Execute the query and process the output stream
    response = query_engine.query(query)
    for token in response.response_gen:
        # Check if the token is not empty
        if token.strip() and first_token_time is None:
            first_token_time = time.time()  # Capture time of the first token
    # Calculate TTFT
    ttft = first_token_time - start_time if first_token_time else 0
    return ttft
    
def main():
    ds = load_dataset("deepmind/narrativeqa", cache_dir = config["cache_dir"])
    num_docs = config["num_docs"]

    documents = []
    for document_idx in range(num_docs):
        document_txt = requests.get(ds["train"][document_idx]["document"]["url"]).text
        documents.append(Document(text = document_txt))

    logging.info("Loaded %d documents", len(documents))
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, metric="euclidean",
        )
    prompt_helper = PromptHelper(context_window=32786,) # for Mistral as default is 3900
    response_synthesizer = get_response_synthesizer(response_mode="compact", prompt_helper=prompt_helper,
                                    streaming=True,)
    query_engine = index.as_query_engine(llm = llm, response_synthesizer=response_synthesizer, 
                                        similarity_top_k= config['similarity_top_k'],
                                        streaming=True,)
    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": new_qa_tmpl}
    )
    # prompts_dict_new = query_engine.get_prompts()
    # logging.debug("New prompt: %s", prompts_dict_new)
    query_list = []
    answers_list = []
    for document_idx in range(num_docs):
        # Add query for specific document_idx
        query_list.append(ds["train"][document_idx]["question"]["text"])
        # Add list of possible answers specific to document_idx
        cur_query_ans_list = [answer["text"] for answer in ds["train"][document_idx]["answers"]]
        answers_list.append(cur_query_ans_list)

    assert len(query_list) == len(answers_list)

    lambda_poisson = 20
    num_events = len(query_list)

    total_ttft = 0
    total_valid_queries = 0
    F1scorer = Scorer(metric='f1')
    scores = []

    for idx, event in enumerate(range(num_events)):
        time_interval = np.random.exponential(1 / lambda_poisson) 
        # Get ttft separately, may be zero if no ttft found
        ttft = get_ttft_from_query_engine(query_engine, query_list[idx])
        start = time.time()
        response = query_engine.query(query_list[idx])
        response_txt = ""
        for text in response.response_gen:
            response_txt += text
        end = time.time()
        # logging.info(f"Response was {response_txt}")
        # logging.info(f"Time taken: {end - start} seconds")
        # logging.info(f"TTFT: {ttft} seconds")

        total_ttft += ttft
        total_valid_queries += (ttft > 0)

        with open('outputs/narrative_qa.csv', 'a') as f:
            ground_truths = answers_list[idx]
            response_txt = parser_answer(str(response_txt))
            score = max(F1scorer.compute_f1(response_txt, ground_truth) for ground_truth in ground_truths)
            scores.append(score)
            f.write(f"{response_txt};{ground_truths};{end-start};{score}\n")
        time.sleep(time_interval)

    # logging.info("Done")
    # F1scorer = Scorer(metric='f1')

    # df = pd.read_csv('outputs/narrative_qa.csv', sep=';', header=None)
    # df.columns = ['response', 'answer', 'time', 'score']

    # gt = df['answer'].values
    # pred = df['response'].values

    # assert len(gt) == len(pred) and len(gt) == len(query_list)

    # scores = []
    # for i in range(len(gt)):
    #     score = max(F1scorer.compute_f1(pred[i], answer) for answer in answers_list[i])
    #     logging.info(f"pred[i] was {pred[i]} for answers {answers_list[i]}, yielding score {score}")
    #     # score = F1scorer.compute_f1(pred[i], gt[i])
    #     scores.append(score)
    
    logging.info(f"F1 Score: {np.mean(scores)}")
    logging.info(f"Average TTFT: {total_ttft / total_valid_queries}")
    # logging.info(f"count was {count}")

if __name__ == '__main__':
    main()
    



