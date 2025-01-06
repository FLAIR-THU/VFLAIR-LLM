import torch
from torch import nn

# from models.layers import *

import math
# from torch.distributions.gamma import Gamma
# from util.utils import get_token_embedding
# from data.load_data import sample_noise_Gaussian, sample_noise_Chi
# from util.globals import *
# from baseline.ks_dist import ddKS
# from transformers import BartForConditionalGeneration, AutoTokenizer


class denoiseModelv3_2slice(nn.Module):
    def __init__(self, d_model, d_out, args, decoder=False): #change some attributes
        super(denoiseModelv3_2slice, self).__init__()
        self.d_model = d_model
        self.d_out = d_out
        
        # self.n_emb_block = args.n_emb_block
        # self.n_noise_block = args.n_noise_block
        self.num_heads = args.num_heads
        self.num_layers = args.num_layers
        self.d_ff = args.d_ff
        self.dim_head = args.dim_head
        self.dropout = args.dropout
        
        if self.d_out != self.d_model:
            #[bs, dim, d_model] -- [bs, dim, d_out]
            self.match_transformer_num_layers = 2
            self.match_transformer = TransformerNopool(d_model, d_out, \
                self.num_heads, self.dim_head, self.match_transformer_num_layers, self.d_ff, self.dropout, decoder)
            # self.linear_match = nn.Linear(d_model, d_out)
            
        self.transformer = TransformerAP(d_out, d_out, self.num_heads, self.dim_head, self.num_layers, self.d_ff, self.dropout, decoder)

    def forward(self, init_embedding, noise, output, attention_mask=None):
        '''
        init_embedding noise: [bs, seq_len, d_model]
        output : [bs, d_out]  num_classes or vocab_dim
        '''
        print('init_embedding:',init_embedding.shape,' noise:',noise.shape,'  output:',output.shape)
        noise_info = torch.cat([init_embedding, noise], dim = 1) # [bs, seq_len+seq_len, hidden_dim]
        if self.d_out != self.d_model:
            noise_info.to(self.match_transformer.fc.weight.device)
            noise_info = self.match_transformer(noise_info)
        noise_info = noise_info.to(output.device)
        print('noise_info:',noise_info.shape)
        print(output.unsqueeze(dim=1).shape)
        output = torch.cat([output, noise_info], dim = 1) # bs, 2seq_len+1, d_out
        output.to(self.transformer.fc.weight.device)
        
        if attention_mask is not None:
            batch_size = output.shape[0]
            ones_ts = torch.ones(batch_size, 1, device = attention_mask.device)
            attention_mask = torch.cat([ones_ts, attention_mask, attention_mask], dim=1)
            attention_mask.to(self.transformer.fc.weight.device)
        
        output = self.transformer(output, mask=attention_mask) # [bs, 2seqlen+1, d_out] --> [bs, d_out]
        # print('transformer output:',output.shape)
        return output
    

class denoiseModelv3_3slice(nn.Module):
    def __init__(self, d_model, d_out, args, decoder=False): #change some attributes
        super(denoiseModelv3_3slice, self).__init__()
        self.d_model = d_model
        self.d_out = d_out
        self.num_heads = args.num_heads
        self.num_layers = args.num_layers
        self.d_ff = args.d_ff
        self.dim_head = args.dim_head
        self.dropout = args.dropout
        # self.comb = args.comb
        
        self.linear_input_first = nn.Linear(2*d_out, d_out)
        self.activation = nn.Tanh()
        self.linear_input_sec = nn.Linear(d_out, d_out)
        self.transformer = Transformer(d_model, d_out, self.num_heads, self.dim_head, self.num_layers, self.d_ff, self.dropout, decoder)
        
    def forward(self, init_embedding, noise, output, attention_mask=None):
        '''
        init_embedding noise: [bs, seq_len, d_model]
        output : [bs, seq_len, d_model] 
        '''
        transformed_output = []
        # print('init_embedding:',init_embedding.shape,' noise:',noise.shape,'  output:',output.shape)
        # print('self.d_model:',self.d_model,' self.d_out:',self.d_out)
        for idx in range(output.shape[1]):
            output_slice  = output[:,idx,:]
            output_slice = torch.cat([output_slice.unsqueeze(1), init_embedding, noise], dim = 1)
            if attention_mask is not None:
                batch_size = output.shape[0]
                ones_ts = torch.ones(batch_size, 1, device = attention_mask.device)
                mask = torch.cat([ones_ts, attention_mask, attention_mask], dim=1)
            output_slice = self.transformer(output_slice, mask=mask)
            transformed_output.append(output_slice)
        transformed_output = torch.stack(transformed_output,dim=1)
        # print('transformed_output:',transformed_output.shape)
        return transformed_output


class denoiseModelv3(nn.Module):
    def __init__(self, d_model, d_out, args, decoder=False): #change some attributes
        super(denoiseModelv3, self).__init__()
        self.d_model = d_model
        self.d_out = d_out
        
        # self.n_emb_block = args.n_emb_block
        # self.n_noise_block = args.n_noise_block
        self.num_heads = args.num_heads
        self.num_layers = args.num_layers
        self.d_ff = args.d_ff
        self.dim_head = args.dim_head
        self.dropout = args.dropout
        # self.comb = args.comb
        
        self.linear_input_first = nn.Linear(2*d_out, d_out)
        self.activation = nn.Tanh()
        self.linear_input_sec = nn.Linear(d_out, d_out)
        self.transformer = Transformer(d_model, d_out, self.num_heads, self.dim_head, self.num_layers, self.d_ff, self.dropout, decoder)

    def forward(self, init_embedding, noise, output, attention_mask=None):
        print('init_embedding:',init_embedding.shape,' noise:',noise.shape,'  output:',output.shape)
        print('self.d_model:',self.d_model,' self.d_out:',self.d_out)
        output = torch.cat([output.unsqueeze(dim=1), init_embedding, noise], dim = 1)
        # output = torch.cat([output, init_embedding, noise], dim = 1)
        
        if attention_mask is not None:
            batch_size = output.shape[0]
            ones_ts = torch.ones(batch_size, 1, device = attention_mask.device)
            attention_mask = torch.cat([ones_ts, attention_mask, attention_mask], dim=1)
        output = self.transformer(output, mask=attention_mask)
        # print('transformer output:',output.shape)
        return output



########### denoise layers

class TransformerNopool(nn.Module):
    def __init__(self, d_model, d_out, num_heads, dim_head, num_layers, d_ff, dropout, decoder=False):
        # super(Transformer, self).__init__()
        super().__init__()
        self.decoder = decoder

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dim_head, d_ff, dropout) for _ in range(num_layers)])
        if self.decoder == True:
            self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dim_head, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src_embed, tgt_embed=None, mask=None):

        enc_output = src_embed
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, mask)

        if self.decoder == True:
            dec_output = tgt_embed
            for dec_layer in self.decoder_layers:
                dec_output = dec_layer(dec_output, enc_output)
            output = self.fc(dec_output)
        else:
            output = self.fc(enc_output)
        # output : [bs, channel_num, d_out]
        return output
    
class TransformerAP(nn.Module):
    def __init__(self, d_model, d_out, num_heads, num_layers, dim_head, d_ff, dropout, decoder=False):
        # super(Transformer, self).__init__()
        super().__init__()
        self.decoder = decoder

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dim_head, d_ff, dropout) for _ in range(num_layers)])
        if self.decoder == True:
            self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dim_head, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, d_out)
        self.dropout = nn.Dropout(dropout)
        self.attn_pool = AttentionPooling(d_out, d_out, dropout)

    def forward(self, src_embed, tgt_embed=None, mask=None):
        enc_output = src_embed
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, mask)

        if self.decoder == True:
            dec_output = tgt_embed
            for dec_layer in self.decoder_layers:
                dec_output = dec_layer(dec_output, enc_output)
            output = self.fc(dec_output)
        else:
            output = self.fc(enc_output) # [bs, 2seq_len+1, d_out]
        
        output = self.attn_pool(output, mask) # [bs, d_out]
        return output

class Transformer(nn.Module):
    def __init__(self, d_model, d_out, num_heads, dim_head, num_layers, d_ff, dropout, decoder=False):
        # super(Transformer, self).__init__()
        super().__init__()
        self.decoder = decoder

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dim_head, d_ff, dropout) for _ in range(num_layers)])
        if self.decoder == True:
            self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dim_head, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src_embed, tgt_embed=None, mask=None):
        '''
        src_embed [bs, dim, d_model]
        mask [bs, dim]
        output [bs, dim, d_out]
        '''
        enc_output = src_embed
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, mask)

        if self.decoder == True:
            dec_output = tgt_embed
            for dec_layer in self.decoder_layers:
                dec_output = dec_layer(dec_output, enc_output)
            output = self.fc(dec_output)
        else:
            output = self.fc(enc_output)
        return output[:, 0, :]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dim_head):
        # super(MultiHeadAttention, self).__init__()
        super().__init__()
        if dim_head is None:
            assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        if dim_head is None:
            self.d_k = d_model // num_heads
        else:
            self.d_k = dim_head
        
        self.W_q = nn.Linear(d_model, self.d_k * self.num_heads)
        self.W_k = nn.Linear(d_model, self.d_k * self.num_heads)
        self.W_v = nn.Linear(d_model, self.d_k * self.num_heads)
        self.W_o = nn.Linear(self.d_k * self.num_heads, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        batch_size, _, _, k_length = attn_scores.size()
        if mask is not None:
            mask_reshp = (batch_size, 1, 1, k_length)
            mask = (mask == 0).view(mask_reshp).expand_as(attn_scores)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_k * self.num_heads)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        # print(attn_output[0][0])
        # print(attn_output[0][0][:,-1])
        output = self.W_o(self.combine_heads(attn_output))
        return output

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        # super(PositionWiseFeedForward, self).__init__()
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_head, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dim_head)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None): #cancel the mask here 
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_head, d_ff, dropout):
        # super(DecoderLayer, self).__init__()
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dim_head)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dim_head)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class AttentionPooling(nn.Module):
    def __init__(self, d_h, hidden_size, drop_rate):
        # super(AttentionPooling, self).__init__()
        super().__init__()
        self.att_fc1 = nn.Linear(d_h, hidden_size // 2)
        self.att_fc2 = nn.Linear(hidden_size // 2, 1)
        self.drop_layer = nn.Dropout(p=drop_rate)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.zeros_(m.bias)

    def forward(self, x, attn_mask=None):

        bz = x.shape[0]
        e = self.att_fc1(x)  # (bz, sequence length, hidden_size // 2)
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e)  # (bz, sequence length, 1)

        alpha = torch.exp(alpha)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8) # (batch size, sequence length, 1)
        x = torch.bmm(x.permute(0, 2, 1), alpha) # (bz, dim, 1)
        x = torch.reshape(x, (bz, -1))  # (bz, dim)
        return x

class CrsTransformer(nn.Module):
    def __init__(self, d_model, d_out, num_heads, dim_head, num_layers, d_ff, dropout):
        # super(Transformer, self).__init__()
        super().__init__()

        self.encoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dim_head, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dim_head, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, d_out)
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
    
    def forward(self, src_embed, tgt_embed, mask=None):

        prev_enc_output = src_embed
        dec_output = tgt_embed
        for i in range(self.num_layers):
            enc_output = self.encoder_layers[i](prev_enc_output, dec_output, mask, mask)
            dec_output = self.decoder_layers[i](dec_output, prev_enc_output, mask, mask)
            prev_enc_output = enc_output
            
        return enc_output, dec_output
    
class denoiseModelv3_clm_2slice(nn.Module):
    def __init__(self, d_model, d_out, args, decoder=False): #change some attributes
        super(denoiseModelv3_clm_2slice, self).__init__()
        self.d_model = d_model
        self.d_out = d_out
        
        self.num_heads = args.num_heads
        self.num_layers = args.num_layers
        self.d_ff = args.d_ff
        self.dim_head = args.dim_head
        self.dropout = args.dropout
        
        if self.d_out != self.d_model:
            self.match_transformer_num_layers = 1
            self.match_transformer = TransformerNopool(d_model, d_out, \
                self.num_heads, self.dim_head, self.match_transformer_num_layers, self.d_ff, self.dropout, decoder)
            
        self.transformer = TransformerAP(d_out, d_out, self.num_heads, self.dim_head, self.num_layers, self.d_ff, self.dropout, decoder)

    def forward(self, init_embedding, noise, output, attention_mask=None):
        '''
        init_embedding noise: [bs, seq_len, d_model]
        output : [bs, d_out=vocab_dim] 
        '''
        print('init_embedding:',init_embedding.shape,' noise:',noise.shape,'  output:',output.shape)
        noise_info = torch.cat([init_embedding, noise], dim = 1) # [bs, seq_len+seq_len, hidden_dim]
        if self.d_out != self.d_model:
            noise_info = self.match_transformer(noise_info)
        output = torch.cat([output.unsqueeze(dim=1), noise_info], dim = 1) # bs, 2seq_len+1, d_out
        output.to(self.transformer.fc.weight.device)
        
        if attention_mask is not None:
            batch_size = output.shape[0]
            ones_ts = torch.ones(batch_size, 1, device = attention_mask.device)
            attention_mask = torch.cat([ones_ts, attention_mask, attention_mask], dim=1)
        output = self.transformer(output, mask=attention_mask) # [bs, 2seqlen+1, d_out] --> [bs, d_out]
        print('transformer output:',output.shape)
        return output


class denoiseModelv3_clm(nn.Module):
    def __init__(self, d_model, d_out, args, decoder=False): #change some attributes
        super(denoiseModelv3_clm, self).__init__()
        self.d_model = d_model
        self.d_out = d_out
        
        self.num_heads = args.num_heads
        self.num_layers = args.num_layers
        self.d_ff = args.d_ff
        self.dim_head = args.dim_head
        self.dropout = args.dropout
        # self.comb = args.comb
        
        self.linear_input_first = nn.Linear(2*d_out, d_out)
        self.activation = nn.Tanh()
        self.linear_input_sec = nn.Linear(d_out, d_out)
        self.transformer = Transformer(d_model, d_out, self.num_heads, self.dim_head, self.num_layers, self.d_ff, self.dropout, decoder)

    def forward(self, init_embedding, noise, output, attention_mask=None):
        '''
        init_embedding noise: [bs, seq_len, d_model]
        output : [bs, d_out=vocab_dim]  vocab_dim
        '''
        print('init_embedding:',init_embedding.shape,' noise:',noise.shape,'  output:',output.shape)
        print('self.d_model:',self.d_model,' self.d_out:',self.d_out)
        output = torch.cat([output.unsqueeze(dim=1), init_embedding, noise], dim = 1)
        # output = torch.cat([output, init_embedding, noise], dim = 1)
        
        if attention_mask is not None:
            batch_size = output.shape[0]
            ones_ts = torch.ones(batch_size, 1, device = attention_mask.device)
            attention_mask = torch.cat([ones_ts, attention_mask, attention_mask], dim=1)
        output = self.transformer(output, mask=attention_mask)
        # print('transformer output:',output.shape)
        return output

