import torch
import torch.nn as nn
import math

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim = embed_size, num_heads=num_heads, batch_first=True)
        
        self.ln1 = nn.LayerNorm(embed_size)
        self.dropout1 = nn.Dropout(dropout)
        
        # feed forward sub block
        self.ff = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.GELU(),
            nn.Linear(4 * embed_size, embed_size), 
        )
        self.ln2 = nn.LayerNorm(embed_size)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attn_mask, padding_mask):
        # TODO: Implement this method
        """
        Args:
            x: (batch_size, seq_len, embed_size)
            attn_mask: (seq_len, seq_len) causal mask (True where we CANNOT attend)
            padding_mask: (batch_size, seq_len) True at PAD positions
        Returns:
            (batch_size, seq_len, embed_size)
        """

        # Multi-head self-attention
        # attn_mask shape: (seq_len, seq_len)
        # padding_mask shape: (batch_size, seq_len)
        attn_out, _ = self.mha (
            query = x,
            key = x,
            value = x,
            attn_mask = attn_mask, 
            key_padding_mask = padding_mask
        )
        # residual + layernorm 
        x = self.ln1(x + self.dropout1(attn_out))
        
        #feed forward
        ff_out = self.ff(x)
        
        # residual + layer norm
        x = self.ln2(x + self.dropout2(ff_out))
        
        return x


class PositionalEncoding(nn.Module):
    """
    Positional encoding module: adds positional information to the input embeddings.
    """
    def __init__(self, embed_size, max_len):
        """
    positional encoding:
      pe[pos, 2i]   = sin(pos / (10000^(2i/embed_size)))
      pe[pos, 2i+1] = cos(pos / (10000^(2i/embed_size)))
    Stored in a buffer, not learnable.
    """
        super().__init__()
        # TODO: Implement this method
        # Use self.register_bufffer("positional_encoding", positional_encoding) to store the positional encoding (not a parameter)
        # Create a (max_len, embed_size) tensor
        
        pe = torch.zeros(max_len, embed_size)
        position =  torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Shape -> (1, max_len, embed_size)
        pe = pe.unsqueeze(0)
        
         # register as a buffer so it's moved with the model but not trained
        self.register_buffer("positional_encoding", pe)
        

    def forward(self, x):
        # TODO: Implement this method
        # Remember to slice the positional encoding to match the length of the input sequence
        # and to move the positional encoding to the device of the input tensor
        
        seq_len = x.size(1)
        
        pe_slice = self.positional_encoding[:, :seq_len, :].to(x.device)
        return x + pe_slice
        

class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_size = config.embed_size
        self.num_layers = config.num_layers 
        self.vocab_size = config.vocab_size
        self.max_len = config.max_len
        self.dropout_p = config.dropout_p
        self.num_heads = config.num_heads
        self.device = config.device
      

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.pos_encoder = PositionalEncoding(self.embed_size, self.max_len)

        self.layers = nn.ModuleList([DecoderBlock(self.embed_size, self.num_heads, self.dropout_p) for _ in range(self.num_layers)])
        self.fc_out = nn.Linear(self.embed_size, self.vocab_size)

        # Precompute the causal mask and positional encoding
        self.register_buffer("causal_mask", self.generate_causal_mask(self.max_len))

    def forward(self, x, padding_mask=None):
        batch_size, seq_len = x.shape

        # Use the precomputed causal mask (trim to match seq_len)
        attn_mask = self.causal_mask[:seq_len, :seq_len]

        # Embed and add positional encoding
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, attn_mask, padding_mask)
        
        logits = self.fc_out(x)
        return logits    

        # return self.fc_out(x)

    def generate_causal_mask(self, seq_len):
        """
        Generates an upper triangular mask to prevent attending to future tokens.
        """
        # TODO: Implement this method
        # You can use torch.ones and torch.triu to generate the mask and cast it to a boolean tensor with .bool()
        
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        return mask.bool()


if __name__ == "__main__":
    from tokenizers import Tokenizer
    from torch.nn.functional import cross_entropy

    from config import config
    from utils import get_num_params
    from dataset import QADataset

    model = TransformerModel(config)
    print(f"Number of parameters in the model: {get_num_params(model):,}")

    # Simple forward pass for sanity checking
    tokenizer = Tokenizer.from_file(config.tokenizer_filename)
    dataset = QADataset(config, tokenizer)
    source = dataset[0]["source_sequence"].unsqueeze(0)
    target = dataset[0]["target_sequence"].unsqueeze(0)
    padding_mask = dataset[0]["key_padding_mask"].unsqueeze(0)

    # Forward pass
    out = model(source, padding_mask)
    print("Output shape:", out.shape)
    print("Target shape:", target.shape)
    print("Loss mask shape:", padding_mask.shape)

    # Calculate loss
    loss = cross_entropy(out.transpose(1, 2), target)
    print("Loss:", loss.item())

