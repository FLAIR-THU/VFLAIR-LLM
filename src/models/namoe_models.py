import dataclasses
import math

import torch
from torch import Tensor, nn
from torch.nn import ModuleList
from transformers import PretrainedConfig, PreTrainedModel

# from sfl.utils.model import get_embed_size, sentence_score_tokens
# from trl import DPOTrainer
def sentence_score_tokens(sent, model):
    model.train(False)
    padded = sent.to(model.device).long()
    stride = 16
    scoress = []
    for i in range(int(np.ceil(len(padded) / stride))):
        outputs = model(padded[i * stride: min((i + 1) * stride, len(padded))])
        lsm = -outputs[0].log_softmax(2)
        preds = torch.zeros_like(lsm)
        preds[:, 1:] = lsm[:, :-1]
        wordscores = (
            preds.gather(
                2, padded[i * stride: min((i + 1) * stride, len(padded))].unsqueeze(2)
            )
            .squeeze(2)
            .detach()
        )
        scores = wordscores.sum(1) / wordscores.shape[1]
        scoress.append(scores)
    # wordscores = torch.cat(wordscoress)
    score = torch.cat(scoress)
    model.train(True)
    # DPOTrainer
    return score

def get_embed_size(target_config: PretrainedConfig):
    n_embed = 0
    if hasattr(target_config, 'n_embd'):
        n_embed = target_config.n_embd
    elif hasattr(target_config, 'hidden_size'):
        n_embed = target_config.hidden_size
    elif hasattr(target_config, 'd_model'):
        n_embed = target_config.d_model
    return n_embed

def get_inverter_class(attack_model):
    attacker_cls = LSTMDRInverter
    if attack_model == 'lstm':
        attacker_cls = LSTMDRInverter
    elif attack_model in ['gru', 'gru-bi']:
        attacker_cls = GRUDRInverter
    elif attack_model == 'linear':
        attacker_cls = LinearSIPInverter
    elif attack_model == 'dec':
        attacker_cls = DecoderSIPInverter
    elif attack_model == 'moe' or attack_model == 'moe2':
        attacker_cls = MOEDRInverter
    elif attack_model == 'vit':
        attacker_cls = ViTDRAttacker
    elif attack_model == 'attngru':
        attacker_cls = AttnGRUDRInverter
    elif attack_model == 'gruattn':
        attacker_cls = GRUAttnSIPInverter
    elif attack_model == 'attn':
        attacker_cls = AttnSIPInverter
    return attacker_cls

def get_inverter_with_config(model_name):
    inverter_clz = get_inverter_class(model_name)
    kwargs = {}
    if model_name == 'gru_bi':
        kwargs['bidirectional'] = True
    cfg = inverter_clz.config_class(**kwargs)
    return inverter_clz, cfg
    # if model_name == 'lstm':
    #     inverter_clz = LSTMDRInverter
    #     cfg = LSTMDRAttackerConfig()
    # elif model_name == 'gru':
    #     inverter_clz = GRUDRInverter
    #     cfg = LSTMDRAttackerConfig()
    # elif model_name == 'gru-bi':
    #     inverter_clz = GRUDRInverter
    #     cfg = LSTMDRAttackerConfig(bidirectional=True)
    # elif model_name == 'linear':
    #     inverter_clz = LinearSIPInverter
    #     cfg = SIPInverterConfig()
    # elif model_name == 'dec':
    #     inverter_clz = DecoderSIPInverter
    #     cfg = TransformerSIPInverterConfig(num_layers=2)
    # elif model_name == 'attngru':
    #     inverter_clz = AttnGRUDRInverter
    #     cfg = TransformerGRUDRAttackerConfig()
    # elif model_name == 'gruattn':
    #     inverter_clz = GRUAttnSIPInverter
    #     cfg = TransformerGRUDRAttackerConfig()
    # elif model_name == 'attn':
    #     cfg = TransformerSIPInverterConfig()
    #     inverter_clz = AttnSIPInverter
    # return inverter_clz, cfg


@dataclasses.dataclass
class SIPInverterConfig(PretrainedConfig):
    vfl_args : dict = None
    model_name: str = None
    target_model: str = None
    vocab_size: int = 0
    n_embed: int = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class SIPInverter(nn.Module):
    """
    SIP Inversion Model
    """

    # config_class = SIPInverterConfig

    def __init__(self, vfl_args,reduce_dim=None,**kwargs):
        super().__init__()
        # if target_config:
        self.attack_config = vfl_args.attack_configs
        self.args = vfl_args
        self.target_config = self.args.model_config #target_config
        self.n_embed = get_embed_size(self.target_config)
        self.vocab_size = self.target_config.vocab_size
        if reduce_dim:
            self.n_embed = reduce_dim
        self.target_model = self.args.model_type
        
        # name_or_path = target_config.name_or_path
        # # if it is a path, use the last dir name
        # if '/' in name_or_path:
        #     if name_or_path.endswith('/'):
        #         name_or_path = name_or_path[:-1]
        #     name_or_path = name_or_path.split('/')[-1]
        # self.config.target_model = name_or_path

    def forward(self, x) -> Tensor:
        if 'chatglm' in self.target_model:
            x = x.permute(1, 0, 2)
        if x.dtype == torch.float16:
            x = x.float()
        return x

    def search(self, x, base_model, beam_size=6):
        logits = self.forward(x.to(self.device))
        batch_size, seq_len, vocab_size = logits.shape
        beams = [(None, [0] * batch_size)] * beam_size
        for step in range(seq_len):
            candidates = []
            for sentence_batch, sent_score_batch in beams:
                last_token_logits = logits[:, step, :]
                topk_probs, topk_indices = torch.topk(last_token_logits, beam_size)
                topk_probs = torch.softmax(topk_probs, dim=-1)
                for k in range(beam_size):
                    prob, token = topk_probs[:, k].unsqueeze(-1), topk_indices[:, k].unsqueeze(-1)  # (batch_size, 1)
                    sents = torch.cat([sentence_batch, token],
                                      dim=1) if sentence_batch is not None else token  # (batch_size, seq++)
                    candidate_score = sentence_score_tokens(sents, base_model).unsqueeze(-1)  # (batch_size, 1)
                    score = prob * 5 - candidate_score
                    # print(prob.shape, candidate_score.shape, score.shape)
                    candidates.append((sents, score))
            new_list = []
            for batch in range(batch_size):
                # print(candidates)
                candidates_batch = [(c[batch, :].unsqueeze(0), score[batch, :].unsqueeze(0)) for c, score in
                                    candidates]
                # print(candidates_batch)
                candidates_batch = sorted(candidates_batch, key=lambda x: x[-1], reverse=True)
                if len(new_list) == 0:
                    new_list = candidates_batch
                else:
                    nl = []
                    for (sent, score), (sent2, score2) in zip(new_list, candidates_batch):
                        nl.append((torch.concat([sent, sent2], dim=0), torch.concat([score, score2], dim=0)))
                    new_list = nl
            beams = new_list[:beam_size]
        return beams[0][0]


class LinearSIPInverter(SIPInverter):
    # config_class = SIPInverterConfig

    def __init__(self, *args, **kwargs):
        # config.model_name = 'linear'
        super().__init__(*args, **kwargs)
        print(f'--init LinearSIPInverter {self.n_embed} --> {self.vocab_size}')
        self.mlp = nn.Linear(self.n_embed, self.vocab_size)
        self.attacker_model_name = 'linear'

    def forward(self, x):
        x = super().forward(x)
        # x[batch_size, seq_len, n_embed]
        # output [batch_size,seq_len, vocab_size]
        return self.mlp(x)


# @dataclasses.dataclass
# class MOEDRAttackerConfig(SIPInverterConfig):
#     dropout: float = 0.1
#     hidden_size: int = 256
#     expert_scales: list[float] = None

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.model_name = 'moe'
#         if self.expert_scales is None:
#             self.expert_scales = [0, 10.0, 7.5, 5.0]


class MOEDRInverter(SIPInverter):

    def __init__(self, vfl_args, expert_scales = [0, 10.0, 7.5, 5.0], hidden_size=256,  dropout=0.1, **kwargs):
        super().__init__(vfl_args, **kwargs)
        
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.expert_scales = expert_scales

        self.experts = ModuleList(
            [nn.GRU(input_size=self.n_embed, hidden_size=self.hidden_size, batch_first=True)
             for _ in self.expert_scales])

        self.gating_mlp = nn.Linear(self.n_embed, self.hidden_size)
        self.gating_mlp2 = nn.Linear(self.hidden_size, len(self.expert_scales))
        self.gating_attn = nn.MultiheadAttention(self.hidden_size, 4, dropout=self.dropout)
      
        self.mlp = nn.Linear(self.hidden_size, self.vocab_size)
        self.attacker_model_name = 'moe'

    def train_exp_forward(self, inters: list):
        assert self.training
        assert len(inters) == len(self.experts)
        outputs = []
        for inter, exp in zip(inters, self.experts):
            # print(f'exp:{next(exp.parameters()).device} inter:{inter.device}')
            inter.to(next(exp.parameters()).device)
            # print(f'aftre inter:{inter.device}')
            if inter is None:
                outputs.append(None)
                continue
            if 'chatglm' in self.target_model:
                inter = inter.permute(1, 0, 2)
            if inter.dtype == torch.float16:
                inter = inter.float()
            hidden = torch.dropout(exp(inter.to(next(exp.parameters()).device))[0], p=self.dropout, train=self.training)
            outputs.append(self.mlp(hidden.to(next(self.mlp.parameters()).device)))
        return outputs

    def freeze_parts(self, experts=False, freeze=True):
        if experts:
            # self.mlp.requires_grad_(not freeze)
            for expert in self.experts:
                expert.requires_grad_(not freeze)
        else:
            self.gating_attn.requires_grad_(not freeze)
            self.gating_mlp.requires_grad_(not freeze)
            self.gating_mlp2.requires_grad_(not freeze)

    def forward(self, x) -> Tensor:
        # x.to(self.mlp.weight.device)
        # x.to(next(self.experts[0].parameters()).device)
        
        x = super().forward(x)
        exp_outputs = [torch.dropout(exp(x.to(next(exp.parameters()).device))[0], p=self.dropout, train=self.training) for exp in self.experts]
        exp_outputs = torch.stack(exp_outputs, dim=1)  # [batch_size, len(experts), seq_len, hidden_size]
        qkv = self.gating_mlp(x.to(next(self.gating_mlp.parameters()).device)).to(next(self.gating_attn.parameters()).device)
        gating_hidden, _ = self.gating_attn(qkv, qkv, qkv)  # [batch_size, seq_len, hidden_size]
        gating_hidden = torch.mean(self.gating_mlp2(gating_hidden), dim=1)  # [batch_size, hidden_size]
        weights = torch.softmax(gating_hidden, dim=-1)  # [batch_size, len(experts)]
        output = torch.einsum('besh,be->bsh', exp_outputs, weights)  # [batch_size, seq_len, hidden_size]
        return self.mlp(output.to(next(self.mlp.parameters()).device))  # [batch_size, seq_len, vocab_size]

