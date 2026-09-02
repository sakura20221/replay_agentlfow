from sentence_transformers import SentenceTransformer
import torch

def get_sentence_embedding(sentence):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(sentence)
    return torch.tensor(embeddings)

class SentenceEncoder(torch.nn.Module):
    def __init__(self,device=None):
        super().__init__()
        self.device = device if device else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device=self.device)
        
    def forward(self, sentence):
        if len(sentence) == 0:
            return torch.tensor([]).to(self.device)
        embeddings = self.model.encode(sentence,convert_to_tensor=True,device=self.device)
        # --- shared-layer shim ---
        # sentence-transformers >= 3 encodes under torch.inference_mode(), and an
        # inference tensor cannot be saved for backward. These embeddings feed the
        # router's trainable heads, so cloning here (outside the inference context)
        # is required to make them ordinary tensors. The encoder remains frozen.
        return embeddings.clone()