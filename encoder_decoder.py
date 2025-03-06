import torch
from torch import nn


        
class Encoder_decoder(nn.Module):
    def __init__(self, h_dim, embed_dim, n_head, n_layer, vocab_size, max_seq_len):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(batch_first= True, d_model=h_dim, nhead=n_head, dropout = .3, dim_feedforward = h_dim, norm_first = True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layer)
        self.decoder_layer = nn.TransformerDecoderLayer(batch_first= True, d_model=h_dim, nhead=n_head, dropout = .3,  dim_feedforward = h_dim, norm_first = True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=n_layer)
        # Since x and y are both english text, I think its fine for them to share embeddings
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim = embed_dim, padding_idx = 0) # zero is our pad index. Always and forever 🕊️
        self.position_embedding_table = nn.Embedding(max_seq_len, embed_dim)
    def forward(self, x, y):
        # where x is the src, and y is the tgt
        x = x.long().to_device()
        y = y.long().to_device()
        batch_size, seq_length_x = x.shape
        _, seq_length_y = y.shape

        # encoding x (src)
        token_embeddings_x = self.embedding(x)
        possitional_embeddings_x = self.position_embedding_table(torch.arange(seq_length_x, device=device))
        x = token_embeddings_x + possitional_embeddings_x # batch, seq, embed_dim

        # encoding y (tgt
        token_embeddings_y = self.embedding(y)
        possitional_embeddings_y = self.position_embedding_table(torch.arange(seq_length_y, device=device))
        y = token_embeddings_y + possitional_embeddings_y  # batch, seq, embed_dim

        # making all required masks
        pad_mask_x = x == 0 # zero is our pad index. Always and forever 🕊️
        pad_mask_y = y == 0  # zero is our pad index. Always and forever 🕊️
        autoregressive_mask = torch.triu(torch.ones(y.shape[1], y.shape[1], device=device).bool(), diagonal=1)



        encoder_output = self.encoder(x, src_key_padding_mask=pad_mask)
        decoder_output = self.decoder(tgt = y, # the summary inputs (used in self and cross)
                                      memory=encoder_output,  # encoder outputs (used in cross attention)
                                      tgt_mask = autoregressive_mask, # mask for summary inputs (used in self attention)
                                      memory_key_padding_mask=pad_mask_x, # our pad mask for encoder outputs
                                      tgt_key_padding_mask=pad_mask_y # our pad mask for decoder inputs
                                      )
        return decoder_output

    

