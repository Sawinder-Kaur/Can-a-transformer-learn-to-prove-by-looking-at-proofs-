import torch
import numpy as np
import matplotlib.pyplot as plt

class Prooformer(torch.nn.Module):
    def __init__(self, d_model, max_len, num_layers, num_tokens, device):
        super().__init__()
        self.max_len = max_len
        self.pos_enc = PositionalEncoding(d_model, max_len, 10000)#.to(device)
        self.embedding = torch.nn.Embedding(num_embeddings=num_tokens, embedding_dim=d_model)
        self.trf = torch.nn.Transformer(d_model, nhead=8, num_encoder_layers=num_layers, num_decoder_layers=num_layers, batch_first=True)
        self.generator = torch.nn.Linear(d_model, num_tokens)
        self.device = device
        self.softmax = torch.nn.Softmax(dim = 2)

    def forward(self, src, tgt):
        """
        proofs[b,t]: tth token of bth proof in the batch
        goals[b,s]: sth token of bth goal in the batch
        logits[b,t,r]: logit for label r at tth step of bth example of the batch
        """
        
        # embed and position encode
        #proofs = self.pos_enc(self.proof_embedding(proofs[:,-self.max_len:]))
        #goals = self.pos_enc(self.goal_embedding(goals[:,-self.max_len:]))

        #src = self.pos_enc(self.embedding(src[:,-self.max_len:])).to(self.device)
        #tgt = self.pos_enc(self.embedding(tgt[:,-self.max_len:])).to(self.device)
        
        src = self.pos_enc(self.embedding(src))#.to(self.device)
        tgt = self.pos_enc(self.embedding(tgt))#.to(self.device)

        # transformer
        mask = causal_mask(tgt.shape[1]).to(self.device)
        result = self.trf(src, tgt, tgt_mask=mask)

        # get logits for next proof label
        """
        readout = self.embedding.weight
        logits = result @ readout.t()
        #"""
        logits = self.softmax(self.generator(result))
        return logits

# causal mask based on
# https://github.com/pytorch/examples/blob/main/word_language_model/model.py
def causal_mask(sz):
    return torch.log(torch.tril(torch.ones(sz,sz)))
"""
mask = causal_mask(10)
plt.imshow(mask)
plt.show()
plt.savefig('causal_mask.png')
"""

"""
Position encoding adapted from

https://github.com/pytorch/examples/blob/main/word_language_model/model.py

This version
- does not use dropout
- expects input shape (batch size, sequence length, embedding dimension)

embedding dimension must be even for sin and cos
"""
    
def pe_tensor(d_model, max_len, base):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(base) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, max_len, base):
        super(PositionalEncoding, self).__init__()
        self.max_len = max_len
        self.register_buffer('pe', pe_tensor(d_model, max_len, base))

    def forward(self, x):
        seq_len = min(self.max_len, x.shape[1])
        x = x[:,-seq_len:] + self.pe[:seq_len, :]
        return x
    

"""
pe = pe_tensor(d_model=256, max_len=128, base=10000)
plt.imshow(pe @ pe.t())
plt.savefig('position_embedding.png')

x = torch.randn(2, 3, 6)
x = PositionalEncoding(6,3,10)(x)
print(x)
    
model = Prooformer(d_model=64, max_len=100, num_layers=2, num_goal_tokens=10, num_proof_tokens=20)

proofs = torch.randint(20, (2,200))
goals = torch.randint(10, (2,50))
logits = model(proofs, goals)
print(logits)
print(logits.shape)
"""