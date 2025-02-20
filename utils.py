import pandas as pd
import numpy as np
import argparse
from rouge_score import rouge_scorer
# from transformers import AutoTokenizer
import tiktoken
import time
import logging, sys

# GPT tokenizer used temporarily for now
tokenizer = tiktoken.get_encoding("cl100k_base")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

class Scorer:
    def __init__(self, metric="f1"):
        self.metric = metric

    def normalize_text(self, s):
        """Removing articles and punctuation, and standardizing 
        whitespace are all typical text processing steps."""
        import string, re

        def remove_articles(text):
            regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
            return re.sub(regex, " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))
 
    def compute_f1(self, prediction, truth):
        pred_tokens = self.normalize_text(prediction).split()
        truth_tokens = self.normalize_text(truth).split()

        # # Import tokenizer from transformers
        # from transformers import AutoTokenizer
        # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # # Tokenize the prediction and truth
        # pred_tokens = tokenizer.encode(prediction, add_special_tokens=False)
        # truth_tokens = tokenizer.encode(truth, add_special_tokens=False)
        
        # if either the prediction or the truth is no-answer 
        # then f1 = 1 if they agree, 0 otherwise
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return int(pred_tokens == truth_tokens)
        
        common_tokens = set(pred_tokens) & set(truth_tokens)
        
        # if there are no common tokens then f1 = 0
        if len(common_tokens) == 0:
            return 0
        
        prec = len(common_tokens) / len(pred_tokens)
        rec = len(common_tokens) / len(truth_tokens)
        
        return 2 * (prec * rec) / (prec + rec)
    
    def compute_accuracy(self, prediction, truth):
        pred_tokens = self.normalize_text(prediction).split()
        truth_tokens = self.normalize_text(truth).split()

        # If truth_tokens in pred_tokens, return 1, else 0
        return int(set(truth_tokens) <= set(pred_tokens))

def compute_score(metric, answer, ground_truth):
    """Compute the score for the given metric."""
    F1Scorer = Scorer(metric='f1')
    if metric == "rouge":
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    elif metric == "f1":
        scorer = F1Scorer()
    elif metric == "accuracy":
        scorer = F1Scorer()
    else:
        raise ValueError(f"Metric not specified: {metric}")

    assert len(answer) == len(ground_truth)
    scores = []
    for i in range(len(answer)):
        if type(answer[i]) == float or type(ground_truth[i]) == float:
            scores.append(0)
            continue
        if metric == "rouge":
            score = scorer.score(answer[i], ground_truth[i])
            fmeasure = score['rouge1'].fmeasure
            scores.append(fmeasure)
        elif metric == "f1":
            score = scorer.compute_f1(answer[i], ground_truth[i])
            scores.append(score)
        elif metric == "accuracy":
            score = scorer.compute_accuracy(answer[i], ground_truth[i])
            scores.append(score)
    return scores

def parser_answer(text):
    import re
    # Keep only the last line 
    text = text.split('\n')[-1]
    # Keep only text after "Answer: "
    text = re.sub(r'Answer: ', '', text).strip()
    return text

def print_chunk_token_counts(response):
    """Prints token counts for each retrieved chunk."""
    for idx, node in enumerate(response.source_nodes):
        chunk_text = node.text
        token_count = len(tokenizer.encode(chunk_text))
        logging.info(f"Chunk {idx + 1}: {token_count} tokens")

def print_total_input_token_count(response, template_str, query):
    query_tokens = len(tokenizer.encode(query))
    context_tokens = sum(len(tokenizer.encode(node.text)) for node in response.source_nodes)
    template_tokens = len(tokenizer.encode(template_str))
    logging.info(f"total token input is {query_tokens + context_tokens + template_tokens}")

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