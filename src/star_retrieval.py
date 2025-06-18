from typing import Dict, List, Tuple
import numpy as np
from scipy.spatial.distance import cosine, cdist
from scipy.sparse import csr_matrix
from tqdm import tqdm
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class STARRetrieval:
    def __init__(self, 
                 semantic_weight: float = 0.5,    
                 temporal_decay: float = 0.7,     
                 history_length: int = 3,
                 gpt2_model_path: str = "D:/surf-RAG/GPT-2"):  # Load the GPT-2 model through the path
        self.semantic_weight = semantic_weight
        self.temporal_decay = temporal_decay
        self.history_length = history_length

        self.semantic_matrix = None
        self.collaborative_matrix = None
        self.item_to_idx = {}
        self.idx_to_item = {}

        # Load the GPT-2 model and Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_path)
        self.model = GPT2LMHeadModel.from_pretrained(gpt2_model_path)

    def compute_semantic_relationships(self, 
                                    item_embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute semantic similarity matrix from item embeddings"""
        print("\nComputing semantic relationships...")

        sorted_items = sorted(item_embeddings.keys())
        self.item_to_idx = {item: idx for idx, item in enumerate(sorted_items)}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        n_items = len(self.item_to_idx)
        
        # Convert embeddings to array and normalize
        embeddings_array = np.zeros((n_items, next(iter(item_embeddings.values())).shape[0]))
        for item_id, embedding in item_embeddings.items():
            embeddings_array[self.item_to_idx[item_id]] = embedding
            
        # L2 normalize embeddings
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        embeddings_array = embeddings_array / norms
        
        # Compute similarities using cosine distance
        semantic_matrix = 1 - cdist(embeddings_array, embeddings_array, metric='cosine')
        np.fill_diagonal(semantic_matrix, 0)
        
        self.semantic_matrix = np.maximum(0, semantic_matrix)
        return self.semantic_matrix

    def generate_gpt2_prompt(self, user_history: List[str], candidate_item: str) -> str:
        """Create a prompt for GPT-2 based on the user's history and the candidate item"""
        # Example prompt for GPT-2: "Given the items the user has liked: ..., how would you rate this item?"
        history_str = ", ".join(user_history)
        prompt = f"Given the items the user has liked: {history_str}. How would you rate this item: {candidate_item}?"
        return prompt

    def get_gpt2_score(self, prompt: str) -> float:
        """Get GPT-2's rating or score for the candidate item based on the generated prompt"""
        # Encode the prompt text and generate GPT-2 output
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(inputs)
            logits = outputs.logits
        # Return the score by taking the last token's probability
        # You can modify this part for more sophisticated scoring if needed
        score = logits[0, -1, self.tokenizer.encode(prompt)[-1]].item()  
        return score

    def score_candidates(self,
                        user_history: List[str],
                        ratings: List[float],
                        candidate_items: List[str] = None,
                        top_k: int = None) -> List[Tuple[str, float]]:
        if self.collaborative_matrix is None:
            raise ValueError("Collaborative matrix not set. Run compute_collaborative_relationships first.")
        
        """Score candidate items based on user history"""
        if len(user_history) > self.history_length:
            user_history = user_history[-self.history_length:]
            ratings = ratings[-self.history_length:]
        
        if candidate_items is None:
            candidate_items = [item for item in self.item_to_idx.keys() 
                             if item not in set(user_history)]
        
        scores = {}
        n = len(user_history)

        # Generate scores for the candidates based on semantic similarity and GPT-2 output
        for candidate in candidate_items:
            if candidate not in self.item_to_idx or candidate in user_history:
                continue
            
            cand_idx = self.item_to_idx[candidate]
            score = 0.0
            
            # Generate the GPT-2 prompt and get a rating score from GPT-2
            prompt = self.generate_gpt2_prompt(user_history, candidate)
            gpt2_score = self.get_gpt2_score(prompt)

            for t, (hist_item, rating) in enumerate(zip(reversed(user_history), 
                                                      reversed(ratings))):
                if hist_item not in self.item_to_idx:
                    continue
                    
                hist_idx = self.item_to_idx[hist_item]
                sem_sim = self.semantic_matrix[cand_idx, hist_idx]
                collab_sim = self.collaborative_matrix[cand_idx, hist_idx]
                
                combined_sim = (self.semantic_weight * sem_sim + 
                              (1 - self.semantic_weight) * collab_sim)
                
                score += (1/n) * rating * (self.temporal_decay ** t) * combined_sim
            
            # Incorporate GPT-2 score into final score 
            score += 0.1 * gpt2_score  # GPT-2's output is added with a small weight
            
            scores[candidate] = score

        # Sort candidates based on the computed scores
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if top_k:
            sorted_items = sorted_items[:top_k]
            
        return sorted_items

    def get_similar_items(self, item_id: str, k: int = 10, use_collaborative: bool = True) -> List[Tuple[str, float]]:
        """Get most similar items using combined similarity"""
        if item_id not in self.item_to_idx:
            return []
            
        idx = self.item_to_idx[item_id]
        similarities = np.zeros(len(self.item_to_idx))
        
        for j in range(len(self.item_to_idx)):
            if j != idx:
                sem_sim = self.semantic_matrix[idx, j]
                collab_sim = self.collaborative_matrix[idx, j] if use_collaborative else 0
                similarities[j] = (
                    self.semantic_weight * sem_sim + 
                    (1 - self.semantic_weight) * collab_sim
                )
        
        # Get top-k similar items
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        return [
            (self.idx_to_item[j], float(similarities[j]))
            for j in top_indices
        ]
