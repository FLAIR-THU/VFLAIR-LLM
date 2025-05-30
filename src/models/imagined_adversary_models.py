import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import math

class ImaginedAdversary_MLP3_noflatten(nn.Module):
    '''
    input --- intermediate : bs, seq_length, 768(embed_dim)
    output --- embedding : bs, seq_length, 768(embed_dim)
    '''

    def __init__(self, seq_length, embed_dim , hidden_size=80):
        super(ImaginedAdversary_MLP3_noflatten, self).__init__()
        # print('Adversarial_MLP init:',seq_length, embed_dim)
        self.seq_length = seq_length
        self.embed_dim = embed_dim

        self.net1 = nn.Sequential(
            nn.Linear(embed_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

        self.net2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )

        self.net3 = nn.Sequential(
            nn.Linear(hidden_size, embed_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        origin_shape = x.shape 
        origin_dtype = x.dtype
        # x = torch.tensor(x, dtype=torch.float32)

        if x.shape[1] != self.seq_length:
            x = x.permute(0, 2, 1)  

        x1 = self.net1(x)

        x2 = self.net2(x1)

        x3 = self.net3(x2)
        

        return x3

class ImaginedAdversary_MLP3(nn.Module):
    '''
    input --- intermediate : bs, seq_length, 768(embed_dim)
    output --- embedding : bs, seq_length, 768(embed_dim)
    '''
    
    def __init__(self, seq_length, embed_dim , hidden_size=80):
        super(ImaginedAdversary_MLP3, self).__init__()
        # print('Adversarial_MLP init:',seq_length, embed_dim)
        self.seq_length = seq_length
        self.embed_dim = embed_dim

        self.net1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_length * embed_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

        self.net2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )

        self.net3 = nn.Sequential(
            nn.Linear(hidden_size, seq_length * embed_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        origin_shape = x.shape 
        origin_dtype = x.dtype
        # x = torch.tensor(x, dtype=torch.float32)
        

        if x.shape[1] != self.seq_length:
            x = x.permute(0, 2, 1)  

        x1 = self.net1(x)

        x2 = self.net2(x1)

        x3 = self.net3(x2)
        
        
        x3 = x3.reshape(origin_shape)

        return x3
    
class ImaginedAdversary_Tail_MLP3_for_generation(nn.Module):
    def __init__(self, embed_dim, vocab_size, \
            hidden_dim=200, num_layers=1, num_heads=12,  dropout=0.1, max_seq_len=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        # Transformer decoder layers
        decoder_layers = TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_layers)
        
        # MLP3 tail (3-layer MLP for output projection)
        self.mlp_tail = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights with reasonable defaults"""
        for layer in self.mlp_tail:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.zero_()
    
    def forward(self, x, attention_mask=None):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
            attention_mask: Optional mask for attention (1 for valid, 0 for masked)
                           Should include causal mask for autoregressive generation
        
        Returns:
            Output tensor of shape [batch_size, seq_len, vocab_size]
        """
       
        # Pass through transformer decoder
        # For decoder-only, we use x as both memory and tgt
        x = self.transformer_decoder(
            tgt=x,
            memory=x,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None
        )
        
        # Pass through MLP tail
        output = self.mlp_tail(x)
        
        return output


class ImaginedAdversary_Tail_MLP3(nn.Module):
    '''
    input --- intermediate : bs, seq_length, 768(embed_dim)
    output --- embedding : bs, label_size/vocab_size
    '''

    def __init__(self, seq_length, embed_dim, label_size, hidden_size=80):
        super(ImaginedAdversary_Tail_MLP3, self).__init__()
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        self.label_size = label_size

        # 对每个 token 的 embed_dim 进行变换
        self.net1 = nn.Sequential(
            nn.Linear(embed_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

        # 进一步变换 hidden_size 维度
        self.net2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )

        # 对 seq_len 维度进行聚合（取均值）
        self.pool = nn.AdaptiveAvgPool1d(1)  # 将 seq_len 维度压缩为 1

        # 将 hidden_size 映射到 label_size
        self.net3 = nn.Sequential(
            nn.Linear(hidden_size, self.label_size),
            nn.Softmax(dim=1)  # 在 label_size 维度上计算 Softmax
        )

    def forward(self, x):
        # 输入形状: [bs, seq_len, embed_dim]
        if x.shape[1] != self.seq_length:
            x = x.permute(0, 2, 1)  # bs, seq_len, embed_dim

        bs, seq_len, embed_dim = x.shape

        x1 = self.net1(x)  

        x2 = self.net2(x1)  

        x2 = x2.permute(0, 2, 1)  
        x2 = self.pool(x2) 
        x2 = x2.squeeze(-1)  

        x3 = self.net3(x2)  

        return x3

# class ImaginedAdversary_Tail_MLP3(nn.Module):
#     '''
#     input --- intermediate : bs, seq_length, 768(embed_dim)
#     output --- embedding : bs, seq_length, 768(embed_dim)
#     '''

#     def __init__(self, seq_length, embed_dim ,label_size, hidden_size=80 ):
#         super(ImaginedAdversary_Tail_MLP3, self).__init__()
#         self.seq_length = seq_length
#         self.embed_dim = embed_dim
#         self.label_size = label_size

#         self.net1 = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(seq_length * embed_dim, hidden_size),
#             nn.LayerNorm(hidden_size),
#             nn.ReLU(),
#         )

#         self.net2 = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size),
#             nn.LayerNorm(hidden_size),
#             nn.ReLU()
#         )

#         self.net3 = nn.Sequential(
#             nn.Linear(hidden_size, self.label_size),
#             nn.Softmax()
#         )

#     def forward(self, x):
#         origin_shape = x.shape 
#         origin_dtype = x.dtype


#         if x.shape[1] != self.seq_length:
#             x = x.permute(0, 2, 1)  # bs, seq_len, embed_dim

#         x1 = self.net1(x)

#         # print('x1:',x1.requires_grad) # = True
#         x2 = self.net2(x1)
#         # print(are_tensors_connected(x1, x2)) # True

#         x3 = self.net3(x2)

#         return x3

def are_tensors_connected(tensor1, tensor2):
    """
    判断两个 Tensor 是否通过计算图连接。
    """
    
    def get_grad_fn_chain(grad_fn):
        """递归获取 grad_fn 链中的所有节点"""
        chain = set()
        stack = [grad_fn]
        while stack:
            node = stack.pop()
            if node is None:
                continue
            if node in chain:
                continue
            chain.add(node)
            for next_fn, _ in node.next_functions:
                stack.append(next_fn)
        return chain

    # 获取两个张量的 grad_fn 链
    chain1 = get_grad_fn_chain(tensor1.grad_fn)
    chain2 = get_grad_fn_chain(tensor2.grad_fn)

    # 检查是否有交集
    return len(chain1.intersection(chain2)) > 0

class ImaginedAdversary_MLP3_on_grad(nn.Module):
    '''
    input --- intermediate : bs, seq_length, 768(embed_dim)
    output --- embedding : bs, seq_length, 768(embed_dim)
    '''

    def __init__(self, seq_length, embed_dim , label_size=2, hidden_size=80):
        super(ImaginedAdversary_MLP3_on_grad, self).__init__()
        # print('Adversarial_MLP init:',seq_length, embed_dim)
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        self.label_size = label_size

        self.net1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_length * embed_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

        self.net2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )

        self.net3 = nn.Sequential(
            nn.Linear(hidden_size, label_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        origin_shape = x.shape 
        origin_dtype = x.dtype

        print('=== ia model ===')
        print('x raw:',x.shape,x.dtype)
        print('origin_shape:',origin_shape)

        if origin_shape[1] != self.seq_length:
            x = x.transpose(0,1) # should be [bs, seq_len, embed_dim]
            print('x after:',x.shape,x.dtype)

        x = torch.tensor(x, dtype=torch.float32)
        x1 = self.net1(x)
        print('x1:',x1.shape)

        x2 = self.net2(x1)
        print('x2:',x2.shape)

        x3 = self.net3(x2)
        print('x3:',x3.shape)

        # x3 = x3.reshape(origin_shape)

        # if not self.batch_first:
        #     x3 = x3.transpose(0,1) # should be [bs, seq_len, embed_dim]

        # print('final x3:',x3.shape,x3.dtype)
        print('=== ia model ===')
        return x3

