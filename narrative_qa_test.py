import logging
import sys, json, os
import pathlib
import utils
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
# from transformers import AutoTokenizer
import tiktoken

# GPT tokenizer used temporarily for now
tokenizer = tiktoken.get_encoding("cl100k_base")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
login(os.environ["HF_TOKEN"])

# HF Tokenizer NOT WORKING currently; gated repo
# tokenizer = AutoTokenizer.from_pretrained(
#     "mistralai/Mistral-7B-Instruct-v0.1",
# )

# Refactor later
config_path = "configs/central.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Create faiss index
faiss_index = faiss.IndexFlatL2(config['embedding_dim'])
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
    
def main():
    # We use the dataset to grab queries and answers, but context is stored through the persisted index
    num_queries = config["num_queries"]
    ds = load_dataset("deepmind/narrativeqa", cache_dir = config["cache_dir"])
    if not os.path.exists("./storage_qa"):
        # Assemble the documents; note that some models will be limited in num_docs context
        documents = []
        seen_doc_ids = set()
        for query_idx in range(num_queries):
            if ds["train"][query_idx]["document"]["id"] not in seen_doc_ids:
                document_txt = requests.get(ds["train"][query_idx]["document"]["url"]).text
                documents.append(Document(text = document_txt))
                seen_doc_ids.add(ds["train"][query_idx]["document"]["id"])

        logging.info("Loaded %d documents", len(documents))
        os.makedirs("./storage_qa")
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, metric="euclidean",
        )
        logging.info("Index built")

        index.storage_context.persist(persist_dir = "./storage_qa/persist_dir")
    
    else:
        vector_store = FaissVectorStore.from_persist_dir("./storage_qa/persist_dir")
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir="./storage_qa/persist_dir", 
        )
        index = load_index_from_storage(storage_context=storage_context,    
                        similarity_top_k= config['similarity_top_k'], metric="euclidean",
                        )
        logging.info("Persisted index")
        
    prompt_helper = PromptHelper(context_window=32786,) # for Mistral as default is 3900
    response_synthesizer = get_response_synthesizer(response_mode="compact", prompt_helper=prompt_helper,
                                    streaming=True,)
    query_engine = index.as_query_engine(llm = llm, response_synthesizer=response_synthesizer, 
                                        similarity_top_k= config['similarity_top_k'],
                                        streaming=True,)
    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": new_qa_tmpl}
    )

    query_list, answers_list = [], []

    # Amass queries and answers (list of lists)
    for query_idx in range(num_queries):
        query_list.append(ds["train"][query_idx]["question"]["text"])
        cur_query_ans_list = [answer["text"] for answer in ds["train"][query_idx]["answers"]]
        answers_list.append(cur_query_ans_list)

    scores_list, ttft_list, total_response_time_list = [], [], []
    F1scorer = Scorer(metric='f1')

    with open('outputs/narrative_qa.csv', 'a') as f:
        for query_idx in range(num_queries):
            # Get ttft separately, may be zero if no ttft found
            ttft = utils.get_ttft_from_query_engine(query_engine, query_list[query_idx])
            # ttft = 1

            # Time the total response
            start = time.time()
            response = query_engine.query(query_list[query_idx])
            response_txt = ""
            for text in response.response_gen:
                response_txt += text
            end = time.time()

            # Print out the token counts for the input chunks
            # utils.print_chunk_token_counts(response)
            utils.print_total_input_token_count(response, new_qa_tmpl_str, query_list[query_idx])

            response_txt = utils.parser_answer(str(response_txt))
            # Here, answer_list[query_idx] is a list of ground truths
            score = max(F1scorer.compute_f1(response_txt, ground_truth) for ground_truth in answers_list[query_idx])
            
            # Track metrics
            scores_list.append(score)
            ttft_list.append(ttft)
            total_response_time_list.append(end - start)

            # Write the response, truths, total time and F1 score to the csv
            f.write(f"{response_txt};{answers_list[query_idx]};{end-start};{score}\n")
    
    logging.info(f"F1 Score: {np.mean(scores_list)}")
    logging.info(f"Average TTFT: {np.mean(ttft_list)}")
    logging.info(f'Average total response time: {np.mean(total_response_time_list)}')

if __name__ == '__main__':
    main()
    



