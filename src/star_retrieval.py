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
                 history_length: int = 3):        
        self.semantic_weight = semantic_weight
        self.temporal_decay = temporal_decay
        self.history_length = history_length
        
        self.semantic_matrix = None
        self.collaborative_matrix = None
        self.item_to_idx = {}
        self.idx_to_item = {}

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
        
        for candidate in candidate_items:
            if candidate not in self.item_to_idx or candidate in user_history:
                continue
                
            cand_idx = self.item_to_idx[candidate]
            score = 0.0
            
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
            
            scores[candidate] = score
        
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

class STARRetrieval_gpt2:
    def __init__(self,
                 semantic_weight: float = 0.5,
                 temporal_decay: float = 0.7,
                 history_length: int = 3,
                 gpt2_model_path: str = "D:/STAR-main/GPT-2/GPT-2"):  # Load GPT-2 model via path
        self.semantic_weight = semantic_weight
        self.temporal_decay = temporal_decay
        self.history_length = history_length

        self.semantic_matrix = None
        self.collaborative_matrix = None
        self.item_to_idx = {}
        self.idx_to_item = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load GPT-2 model and Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_path)
        self.model = GPT2LMHeadModel.from_pretrained(gpt2_model_path).to(self.device)

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
        history_str = ", ".join(user_history)
        prompt = f"Given the items the user has liked: {history_str}. How would you rate this item: {candidate_item}?"
        return prompt

    def get_gpt2_score(self, prompt: str) -> float:
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)  # Move to GPU
        with torch.no_grad():
            outputs = self.model(inputs)
            logits = outputs.logits
        score = logits[0, -1, inputs[0][-1]].item()  # Avoid redundant encode
        return score

    def score_candidates(self,
                         user_history: List[str],
                         ratings: List[float],
                         candidate_items: List[str] = None,
                         top_k: int = None) -> List[Tuple[str, float]]:
        """
        Score candidate items based on user history, supporting staged recall and scoring

        Parameters:
        - user_history: List of user's interaction history items
        - ratings: Corresponding rating list
        - candidate_items: List of candidate items, default is None (use all non-interacted items)
        - top_k: Number of top-rated items to return, default is None (returns all items)
        """
        if self.collaborative_matrix is None:
            raise ValueError("Collaborative matrix not set. Run compute_collaborative_relationships first.")

        # Truncate history
        if len(user_history) > self.history_length:
            user_history = user_history[-self.history_length:]
            ratings = ratings[-self.history_length:]

        # Get candidate items
        if candidate_items is None:
            candidate_items = [item for item in self.item_to_idx.keys()
                               if item not in set(user_history)]

        # Stage 1: Recall (use collaborative filtering and semantic similarity)

        # Recall using only collaborative filtering and semantic similarity
        recall_scores = {}
        n = len(user_history)

        for candidate in candidate_items:
            if candidate not in self.item_to_idx or candidate in user_history:
                continue

            cand_idx = self.item_to_idx[candidate]
            score = 0.0

            for t, (hist_item, rating) in enumerate(zip(reversed(user_history),
                                                            reversed(ratings))):
                if hist_item not in self.item_to_idx:
                    continue

                hist_idx = self.item_to_idx[hist_item]
                sem_sim = self.semantic_matrix[cand_idx, hist_idx]
                collab_sim = self.collaborative_matrix[cand_idx, hist_idx]

                combined_sim = (self.semantic_weight * sem_sim +
                                (1 - self.semantic_weight) * collab_sim)

                score += (1 / n) * rating * (self.temporal_decay ** t) * combined_sim

            recall_scores[candidate] = score

            # Recall top-k items
        recalled_items = [item for item, _ in sorted(recall_scores.items(),
                                                         key=lambda x: x[1], reverse=True)[:top_k]]
        candidate_items = recalled_items


        # Stage 2: Scoring (use collaborative filtering, semantic similarity, and GPT-2)
        final_scores = {}
        n = len(user_history)

        for candidate in candidate_items:
            if candidate not in self.item_to_idx or candidate in user_history:
                continue

            cand_idx = self.item_to_idx[candidate]
            score = 0.0

            # Compute weighted similarity score for history items
            for t, (hist_item, rating) in enumerate(zip(reversed(user_history),
                                                        reversed(ratings))):
                if hist_item not in self.item_to_idx:
                    continue

                hist_idx = self.item_to_idx[hist_item]
                sem_sim = self.semantic_matrix[cand_idx, hist_idx]
                collab_sim = self.collaborative_matrix[cand_idx, hist_idx]

                combined_sim = (self.semantic_weight * sem_sim +
                                (1 - self.semantic_weight) * collab_sim)

                score += (1 / n) * rating * (self.temporal_decay ** t) * combined_sim

            # Add GPT-2 score when needed
            prompt = self.generate_gpt2_prompt(user_history, candidate)
            gpt2_score = self.get_gpt2_score(prompt)
            score += 0.1 * gpt2_score  # GPT-2 output weight is 0.1

            final_scores[candidate] = score

        # Sort by score and return top_k
        sorted_items = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
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
