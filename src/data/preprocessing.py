import re
import hashlib
from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np
from collections import defaultdict


def preprocess_text(text: str, min_length: int = 10) -> str:
    text = text.strip()
    if len(text) < min_length:
        return ""
    
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?;:\-()\[\]{}"\']', '', text)
    
    return text


def compute_minhash(text: str, num_perm: int = 128) -> np.ndarray:
    shingles = set()
    words = text.lower().split()
    
    for i in range(len(words) - 2):
        shingle = ' '.join(words[i:i+3])
        shingles.add(shingle)
    
    minhash = np.full(num_perm, np.inf)
    
    for shingle in shingles:
        hash_val = int(hashlib.md5(shingle.encode()).hexdigest(), 16)
        for i in range(num_perm):
            hash_val = (hash_val * 1103515245 + 12345) & 0x7fffffff
            minhash[i] = min(minhash[i], hash_val)
    
    return minhash


def compute_jaccard_similarity(minhash1: np.ndarray, minhash2: np.ndarray) -> float:
    return np.sum(minhash1 == minhash2) / len(minhash1)


def deduplicate_dataset(
    texts: List[str],
    threshold: float = 0.95,
    num_perm: int = 128,
) -> List[str]:
    if len(texts) == 0:
        return []
    
    minhashes = [compute_minhash(text, num_perm) for text in texts]
    seen = set()
    unique_texts = []
    
    for i, text in enumerate(texts):
        is_duplicate = False
        minhash = minhashes[i]
        
        for j in seen:
            similarity = compute_jaccard_similarity(minhash, minhashes[j])
            if similarity >= threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            seen.add(i)
            unique_texts.append(text)
    
    return unique_texts


def compute_perplexity_estimate(text: str, unigram_counts: Dict[str, int], total_words: int) -> float:
    words = text.lower().split()
    if len(words) == 0:
        return float('inf')
    
    log_prob_sum = 0.0
    for word in words:
        count = unigram_counts.get(word, 0)
        prob = (count + 1) / (total_words + len(unigram_counts))
        log_prob_sum += np.log(prob + 1e-10)
    
    avg_log_prob = log_prob_sum / len(words)
    perplexity = np.exp(-avg_log_prob)
    
    return perplexity


def filter_quality(
    texts: List[str],
    perplexity_threshold: float = 100.0,
    min_length: int = 128,
    max_length: int = 8192,
) -> List[str]:
    if len(texts) == 0:
        return []
    
    unigram_counts = defaultdict(int)
    total_words = 0
    
    for text in texts:
        words = text.lower().split()
        for word in words:
            unigram_counts[word] += 1
            total_words += 1
    
    filtered_texts = []
    
    for text in texts:
        if len(text) < min_length or len(text) > max_length:
            continue
        
        perplexity = compute_perplexity_estimate(text, unigram_counts, total_words)
        
        if perplexity < perplexity_threshold:
            filtered_texts.append(text)
    
    return filtered_texts

