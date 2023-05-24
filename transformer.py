import torch.nn as nn
import torch
from transformers import RobertaModel
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import math
import copy

class Beam(object):
    def __init__(self, size, sos, eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.LongTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                           .fill_(sos)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished = []
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i))
            unfinished.sort(key=lambda a: -a[0])
            self.finished += unfinished[:self.size - len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps = []
        for _, timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j + 1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps

    def buildTargetTokens(self, preds):
        sentence = []
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok == self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence

class myTransformerDecoderLayer(nn.Module):
    from typing import Optional, Union, Callable
    from torch import Tensor
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(myTransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            if activation == "relu":
                self.activation = F.relu
            elif activation == "gelu":
                self.activation = F.gelu
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(myTransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class myTransformerDecoder(nn.Module):
    from typing import Optional
    from torch import Tensor
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(myTransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.2, max_len=256):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.transpose(0, 1)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x).transpose(0, 1)

class Token_Encoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward=2048, dropout=0.2):
        super(Token_Encoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                        nhead=nhead,
                                                        dim_feedforward=dim_feedforward,
                                                        dropout=dropout,
                                                        activation='gelu',
                                                        batch_first=True)
        self.token_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, src, mask):
        return self.token_encoder(src, src_key_padding_mask=mask)  # [batch_size, seq_len, emb_size]

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward=2048, dropout=0.2):
        super(Decoder, self).__init__()
        self.decoder_layer = myTransformerDecoderLayer(d_model=d_model,
                                                       nhead=nhead,
                                                       dim_feedforward=dim_feedforward,
                                                       dropout=dropout,
                                                       activation='gelu',
                                                       batch_first=True)
        self.transformer_decoder = myTransformerDecoder(self.decoder_layer, num_layers=num_layers)
    def forward(self, tgt, memory, memory_key_padding_mask, tgt_mask=None, tgt_key_padding_mask=None):
        if tgt_mask is not None and tgt_key_padding_mask is not None :
            return self.transformer_decoder(tgt=tgt,
                                            memory=memory,
                                            tgt_mask=tgt_mask,
                                            tgt_key_padding_mask=tgt_key_padding_mask,
                                            memory_key_padding_mask=memory_key_padding_mask)
        else:
            return self.transformer_decoder(tgt=tgt, memory=memory, memory_key_padding_mask=memory_key_padding_mask)

class my_transformer_model(nn.Module):
    def __init__(self, vacab_size, emb_size, hidden_size, nhead, beam_size=None, max_length=None, sos_id=None, eos_id=None):
        super(my_transformer_model, self).__init__()
        self.hidden_size = hidden_size
        self.nhead = nhead
        self.embedding = nn.Embedding(vacab_size, emb_size)
        self.pos_encoder = PositionalEncoding(emb_size, dropout=0.2)
        self.token_encoder = Token_Encoder(6, emb_size, self.nhead)
        self.decoder = Decoder(6, emb_size, self.nhead)
        self.register_buffer(
            "bias", torch.tril(torch.ones((1024, 1024), dtype=torch.uint8)).view(1, 1024, 1024)
        )

        self.lm_head = nn.Linear(self.hidden_size, vacab_size, bias=False)
        # self.lm_head.weight = self.encoder.embeddings.word_embeddings.weight
        self.lsm = nn.LogSoftmax(dim=-1)
        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id


    def forward(self, source_ids, path_ids, target_ids=None, temperature=1.0, tasks='gst'):
        if target_ids is None:
            return self.generate(source_ids, path_ids)

        source_mask = source_ids.eq(1)
        tgt_pad_mask = target_ids.eq(1)

        source_embedding = self.embedding(source_ids)  # [batch_size, seq_len, emb_size]
        target_embedding = self.embedding(target_ids)  # [batch_size, tgt_seq_len, emb_size]
        if tasks == 'gst':
            source_embedding = self.pos_encoder(source_embedding)
            target_embedding = self.pos_encoder(target_embedding)
            token_encoder_output = self.token_encoder(source_embedding, source_mask)  # [batch_size, seq_len, emb_size]
            attn_mask = torch.triu(torch.full((target_ids.shape[1], target_ids.shape[1]), bool('False')), diagonal=1).cuda()
            token_decoder_output = self.decoder(target_embedding,
                                                token_encoder_output,
                                                tgt_mask=attn_mask,
                                                tgt_key_padding_mask=tgt_pad_mask,
                                                memory_key_padding_mask=source_mask)
            lm_logits = self.lm_head(token_decoder_output)  # [batch_size, seq_len, vacab_size]
            # Shift so that tokens < n predict n
            active_loss = target_ids[..., 1:].ne(1).view(-1)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                            shift_labels.view(-1)[active_loss])

            outputs = loss, loss * active_loss.sum(), active_loss.sum()
            # outputs = loss
            return outputs

    def generate(self, source_ids, path_ids):
        source_mask = source_ids.eq(1)
        source_embedding = self.embedding(source_ids)
        source_embedding = self.pos_encoder(source_embedding)
        token_encoder_output = self.token_encoder(source_embedding, source_mask)
        preds = []
        zero = torch.cuda.LongTensor(1).fill_(0)
        # source_len = list(source_ids.ne(1).sum(-1).cpu().numpy())
        for i in range(source_ids.shape[0]):
            beam = Beam(self.beam_size, self.sos_id, self.eos_id)
            input_ids = beam.getCurrentState()
            memory = token_encoder_output[i:i+1].repeat(self.beam_size, 1, 1)  # [beam_size, seq_len, emb_size]
            memory_pad_mask = source_mask[i:i+1].repeat(self.beam_size, 1)  # [beam_size, seq_len]
            # context_ids = source_ids[i:i + 1, :source_len[i]].repeat(self.beam_size, 1)
            for _ in range(self.max_length):
                if beam.done():
                    break
                # ids = torch.cat((context_ids, input_ids), -1)
                input_embedding = self.embedding(input_ids)  # [beam_size, tgt_len, emb_size]
                input_embedding = self.pos_encoder(input_embedding)
                out = self.decoder(input_embedding, memory=memory, memory_key_padding_mask=memory_pad_mask)  # [beam_size, seq_len, emb_size]
                hidden_states = out[:, -1, :]
                out = self.lsm(self.lm_head(hidden_states)).data
                beam.advance(out)
                input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                input_ids = torch.cat((input_ids, beam.getCurrentState()), -1)
            hyp = beam.getHyp(beam.getFinal())
            pred = beam.buildTargetTokens(hyp)[:self.beam_size]
            pred = [torch.cat([x.view(-1) for x in p] + [zero] * (self.max_length - len(p))).view(1, -1) for p in pred]
            preds.append(torch.cat(pred, 0).unsqueeze(0))

        preds = torch.cat(preds, 0)

        return preds


