
import torch
import torch.nn as nn

class TypeEncoding(nn.Module):
    def __init__(self, num_types, channels, existing_embedding=None):
        super().__init__()
        self.type_embedding = nn.Embedding(num_types, channels)
        self.existing_embedding = existing_embedding

    def forward(self, types, x=None):
        assert (self.existing_embedding is None) == (x is None), f"This model expects precomputed {self.existing_embedding} embeddings. Please pass them for correct computation."

        type_embedding = self.type_embedding(types)
        if self.existing_embedding:
            return x + type_embedding
        else:
            return type_embedding