import torch
import torch.nn as nn
from .utils import PositionalEncoding
from .selfattn import DFSCodeEncoder

class DFSCodeVariationalAutoencoder(nn.Module):
    def __init__(self, atom_embedding, bond_embedding, 
                 n_atoms, n_bonds, emb_dim=120, nhead=12, 
                 nlayers=6, n_memory_blocks=1, dim_feedforward=2048, 
                 max_nodes=250, max_edges=500, dropout=0.1, 
                 missing_value=None):
        super().__init__()
        self.ninp = emb_dim * 5
        self.n_memory_blocks = n_memory_blocks # length of the bottleneck sequence
        self.encoder = DFSCodeEncoder(atom_embedding, bond_embedding, 
                                      emb_dim=emb_dim, nhead=nhead, nlayers=nlayers, 
                                      dim_feedforward=dim_feedforward,
                                      max_nodes=max_nodes, max_edges=max_edges, 
                                      dropout=dropout, missing_value=missing_value)
        self.bottleneck_mean = nn.MultiheadAttention(self.ninp, nhead)
        self.bottleneck_log_var = nn.MultiheadAttention(self.ninp, nhead)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=self.ninp, 
                                                                    nhead=nhead,
                                                                    dim_feedforward=dim_feedforward,
                                                                    dropout=dropout), nlayers)
        self.fc_dfs_idx1 = nn.Linear(self.ninp, max_nodes)
        self.fc_dfs_idx2 = nn.Linear(self.ninp, max_nodes)
        self.fc_atom1 = nn.Linear(self.ninp, n_atoms)
        self.fc_atom2 = nn.Linear(self.ninp, n_atoms)
        self.fc_bond = nn.Linear(self.ninp, n_bonds)
        
        self.memory_query = nn.Parameter(torch.empty(n_memory_blocks, 1, self.ninp), requires_grad=True)
        pos_emb = PositionalEncoding(self.ninp, dropout=0, max_len=max_edges)(torch.zeros((max_edges, 1, self.ninp)))
        pos_emb = torch.squeeze(pos_emb)
        self.register_buffer('output_query', pos_emb)
        nn.init.normal_(self.memory_query, mean=.0, std=.5)
        

    def forward(self, C, N, E):
        self_attn, src_key_padding_mask = self.encoder(C, N, E) # seq x batch x feat
        query = torch.tile(self.memory_query, [1, self_attn.shape[1], 1]) # n_memory_blocks x batch x feat
        memory_mean, _ = self.bottleneck_mean(query, self_attn, self_attn,
                                                           key_padding_mask=src_key_padding_mask) # n_memory_blocks x batch x feat
        memory_log_var, _ = self.bottleneck_log_var(query, self_attn, self_attn,
                                                           key_padding_mask=src_key_padding_mask) # n_memory_blocks x batch x feat
        memory_std = torch.exp(0.5*memory_log_var)
        memory = memory_mean + torch.randn_like(memory_mean)*memory_std
        
        tgt = torch.tile(self.output_query[:self_attn.shape[0]].unsqueeze(1), [1, self_attn.shape[1], 1]) # seq x batch x feat
        cross_attn = self.decoder(tgt, memory, tgt_key_padding_mask=src_key_padding_mask) # seq x batch x feat
        batch = cross_attn.permute(1,0,2) # batch x seq x feat
        dfs_idx1_logits = self.fc_dfs_idx1(batch).permute(1,0,2) #seq x batch x feat
        dfs_idx2_logits = self.fc_dfs_idx2(batch).permute(1,0,2)
        atom1_logits = self.fc_atom1(batch).permute(1,0,2)
        atom2_logits = self.fc_atom2(batch).permute(1,0,2)
        bond_logits = self.fc_bond(batch).permute(1,0,2)
        return dfs_idx1_logits, dfs_idx2_logits, atom1_logits, atom2_logits, bond_logits
    
    
    def encode(self, C, N, E):
  
        self_attn, src_key_padding_mask = self.encoder(C, N, E) # seq x batch x feat
        query = torch.tile(self.memory_query, [1, self_attn.shape[1], 1]) # n_memory_blocks x batch x feat
        memory_mean, _ = self.bottleneck_mean(query, self_attn, self_attn,
                                                           key_padding_mask=src_key_padding_mask) # n_memory_blocks x batch x feat
        memory_log_var, _ = self.bottleneck_log_var(query, self_attn, self_attn,
                                                           key_padding_mask=src_key_padding_mask) # n_memory_blocks x batch x feat
        return memory_mean, memory_log_var