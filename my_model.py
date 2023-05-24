import torch.nn as nn
import torch
from transformers import RobertaModel
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import math
import copy
from point_transformer import Decoder

def plot_embedding(path_data, target_data):
    tsne = TSNE(n_components=2, init='pca', random_state=501)
    path_data = tsne.fit_transform(path_data)
    label = np.arange(path_data.shape[0])
    x_min, x_max = np.min(path_data, 0), np.max(path_data, 0)
    path_data = (path_data - x_min) / (x_max - x_min)
    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(path_data.shape[0]):
        plt.text(path_data[i, 0], path_data[i, 1], str(label[i]),
                 color='r',
                 fontdict={'weight': 'bold', 'size': 9})
    tsne = TSNE(n_components=2, init='pca', random_state=501)
    target_data = tsne.fit_transform(target_data)
    x_min, x_max = np.min(target_data, 0), np.max(target_data, 0)
    target_data = (target_data - x_min) / (x_max - x_min)
    for i in range(target_data.shape[0]):
        plt.text(target_data[i, 0], target_data[i, 1], str(label[i]),
                 color='b',
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.show()

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
    def new_getHyp(self, beam_res):
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
        scores = [x[0].item() for x in beam_res]
        return scores, hyps

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

# class myTransformerDecoderLayer(nn.Module):
#     from typing import Optional, Union, Callable
#     from torch import Tensor
#     __constants__ = ['batch_first', 'norm_first']
#
#     def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
#                  activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
#                  layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
#                  device=None, dtype=None) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super(myTransformerDecoderLayer, self).__init__()
#         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
#                                             **factory_kwargs)
#         self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
#                                                  **factory_kwargs)
#         # Implementation of Feedforward model
#         self.proj = nn.Linear(d_model*2, d_model, **factory_kwargs)
#         self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
#
#         self.norm_first = norm_first
#         self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
#         self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
#         self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.dropout3 = nn.Dropout(dropout)
#
#         # Legacy string support for activation function.
#         if isinstance(activation, str):
#             if activation == "relu":
#                 self.activation = F.relu
#             elif activation == "gelu":
#                 self.activation = F.gelu
#         else:
#             self.activation = activation
#
#     def __setstate__(self, state):
#         if 'activation' not in state:
#             state['activation'] = F.relu
#         super(myTransformerDecoderLayer, self).__setstate__(state)
#
#     def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
#                 tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None, semantic: Optional[Tensor] = None) -> Tensor:
#
#         x = tgt
#         if self.norm_first:
#             x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
#             # semantic = semantic.unsqueeze(1).repeat(1, x.shape[1], 1)  # [batch_size, seq_len, emb_size]
#             # x = x.cat(semantic, dim=-1)  # [batch_size, seq_len, emb_size*2]
#             # x = self.proj(x)  # [batch_size, seq_len, emb_size]
#             x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
#             x = x + self._ff_block(self.norm3(x))
#         else:
#             x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
#             # semantic = semantic.unsqueeze(1).repeat(1, x.shape[1], 1)  # [batch_size, seq_len, emb_size]
#             # x = torch.cat((x, semantic), dim=-1)  # [batch_size, seq_len, emb_size*2]
#             # x = self.proj(x)  # [batch_size, seq_len, emb_size]
#             x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
#             x = self.norm3(x + self._ff_block(x))
#
#         return x
#
#     # self-attention block
#     def _sa_block(self, x: Tensor,
#                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
#         x = self.self_attn(x, x, x,
#                            attn_mask=attn_mask,
#                            key_padding_mask=key_padding_mask,
#                            need_weights=False)[0]
#         return self.dropout1(x)
#
#     # multihead attention block
#     def _mha_block(self, x: Tensor, mem: Tensor,
#                    attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
#         x = self.multihead_attn(x, mem, mem,
#                                 attn_mask=attn_mask,
#                                 key_padding_mask=key_padding_mask,
#                                 need_weights=False)[0]
#         return self.dropout2(x)
#
#     # feed forward block
#     def _ff_block(self, x: Tensor) -> Tensor:
#         x = self.linear2(self.dropout(self.activation(self.linear1(x))))
#         return self.dropout3(x)
#
# def _get_clones(module, N):
#     return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
#
# class myTransformerDecoder(nn.Module):
#     from typing import Optional
#     from torch import Tensor
#     __constants__ = ['norm']
#
#     def __init__(self, decoder_layer, num_layers, norm=None):
#         super(myTransformerDecoder, self).__init__()
#         self.layers = _get_clones(decoder_layer, num_layers)
#         self.num_layers = num_layers
#         self.norm = norm
#
#     def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
#                 memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
#                 memory_key_padding_mask: Optional[Tensor] = None, semantic = None) -> Tensor:
#         output = tgt
#
#         for mod in self.layers:
#             output = mod(output, memory, tgt_mask=tgt_mask,
#                          memory_mask=memory_mask,
#                          tgt_key_padding_mask=tgt_key_padding_mask,
#                          memory_key_padding_mask=memory_key_padding_mask,
#                          semantic=semantic)
#
#         if self.norm is not None:
#             output = self.norm(output)
#
#         return output

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

class Path_Encoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward=2048, dropout=0.2):
        super(Path_Encoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                        nhead=nhead,
                                                        dim_feedforward=dim_feedforward,
                                                        dropout=dropout,
                                                        activation='gelu',
                                                        batch_first=True)
        self.path_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
    def forward(self, src, mask=None):
        if mask == None:
            return self.path_encoder(src)  # [node_num, path_len, emb_size]
        else:
            return self.path_encoder(src, src_key_padding_mask=mask)  # [node_num, path_len, emb_size]

class Gru_Path_Encoder(nn.Module):
    def __init__(self, emb_dim, layer_num=2):
        super(Gru_Path_Encoder, self).__init__()

        self.layer_num = layer_num
        self.emb_dim = emb_dim
        self.hid_dim = emb_dim

        self.rnn = nn.GRU(input_size=emb_dim,
                          hidden_size=emb_dim,
                          num_layers=self.layer_num,
                          batch_first=True,
                          bidirectional=True)

        self.fc = nn.Linear(2 * emb_dim, emb_dim)

    def forward(self, src):
        outputs, _ = self.rnn(src)
        outputs = self.fc(outputs)
        return outputs

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

# class Nl_Encoder(nn.Module):
#     def __init__(self, hid_dim):
#         super(Nl_Encoder, self).__init__()
#         self.nl_encoder = RobertaModel.from_pretrained('microsoft/unixcoder-base')
#         self.top = nn.Linear(768, hid_dim)
#         self.dropout = nn.Dropout(0.2)
#         # for p in self.parameters():
#         #     p.requires_grad = False
#
#     def forward(self, ids):
#         mask = ids.ne(1)[:, None, :] * ids.ne(1)[:, :, None]
#         x = self.nl_encoder(ids, mask)[1]  # [batch_size, 768]
#         x = self.top(x)  # [batch_size, hid_dim]
#         x = self.dropout(x)
#         return x

# class Decoder(nn.Module):
#     def __init__(self, emb_dim, hid_dim, dropout):
#         super(Decoder, self).__init__()
#
#         self.emb_dim = emb_dim
#         self.hid_dim = hid_dim
#         self.dropout = dropout
#
#         self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, input, hidden):
#
#         # embedded = self.dropout(input)
#
#         output, hidden = self.rnn(input, hidden)
#
#         return output, hidden

# class Decoder(nn.Module):
#     def __init__(self, num_layers, d_model, nhead, dim_feedforward=2048, dropout=0.2):
#         super(Decoder, self).__init__()
#         self.decoder_layer = myTransformerDecoderLayer(d_model=d_model,
#                                                        nhead=nhead,
#                                                        dim_feedforward=dim_feedforward,
#                                                        dropout=dropout,
#                                                        activation='gelu',
#                                                        batch_first=True)
#         self.transformer_decoder = myTransformerDecoder(self.decoder_layer, num_layers=num_layers)
#     def forward(self, tgt, memory, memory_key_padding_mask, tgt_mask=None, tgt_key_padding_mask=None, semantic=None):
#         if tgt_mask is not None and tgt_key_padding_mask is not None :
#             return self.transformer_decoder(tgt=tgt,
#                                             memory=memory,
#                                             tgt_mask=tgt_mask,
#                                             tgt_key_padding_mask=tgt_key_padding_mask,
#                                             memory_key_padding_mask=memory_key_padding_mask,
#                                             semantic=semantic)
#         else:
#             return self.transformer_decoder(tgt=tgt, memory=memory, memory_key_padding_mask=memory_key_padding_mask, semantic=semantic)



class my_model(nn.Module):
    def __init__(self, vocab_size, max_source_len, d_model, hidden_size, nhead, num_layers, gamma, beam_size=None, max_length=None, sos_id=None, eos_id=None):
        super(my_model, self).__init__()
        self.hidden_size = hidden_size
        self.nhead = nhead
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.num_layers = num_layers
        self.pos_encoder = PositionalEncoding(d_model, dropout=0.2)
        # self.path_encoder = Path_Encoder(4, emb_size, self.nhead)
        self.path_encoder = Gru_Path_Encoder(d_model)
        self.token_encoder = Token_Encoder(self.num_layers, d_model, self.nhead)
        self.proj = torch.nn.Parameter(torch.rand(d_model, d_model))
        self.decoder = Decoder(self.num_layers, d_model, self.nhead)
        # self.nl_encoder = Nl_Encoder(hid_dim=emb_size)
        # self.attn = nn.MultiheadAttention(embed_dim=emb_size, num_heads=self.nhead, batch_first=True)
        self.alpha = torch.nn.Parameter(torch.rand(1, 1))
        self.gamma = gamma
        # self.register_buffer(
        #     "bias", torch.tril(torch.ones((1024, 1024), dtype=torch.uint8)).view(1, 1024, 1024)
        # )

        # self.lm_head = nn.Linear(self.hidden_size, vacab_size, bias=False)
        # self.lm_head.weight = self.encoder.embeddings.word_embeddings.weight
        # self.lsm = nn.LogSoftmax(dim=-1)

        self.CBOW_linear1 = nn.Linear(d_model, d_model)
        self.CBOW_activate1 = nn.GELU()
        self.CBOW_linear2 = nn.Linear(d_model, vocab_size)
        self.CBOW_activate2 = nn.LogSoftmax(dim=-1)

        self.p_vocab = nn.Sequential(
            nn.Linear(d_model, vocab_size),
            nn.Softmax(dim=-1))
        self.p_gen = nn.Sequential(
            nn.Linear(d_model * 3, 1),
            nn.Sigmoid())
        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id

    def get_ctl_logits(self, query, keys, batch_size):
        expand_query = query.repeat((1, batch_size, 1))
        expand_keys = torch.transpose(keys, 0, 1).repeat((batch_size, 1, 1))
        d_pos = F.pairwise_distance(query, keys, p=2)
        d_pos = d_pos.repeat(1, batch_size)
        d_neg = F.pairwise_distance(expand_query, expand_keys, p=2)
        lambda_coefficient = (d_pos / d_neg)

        hardness_masks = torch.gt(d_neg, d_pos).int()
        hard_keys = (expand_query + lambda_coefficient.unsqueeze(2) * (expand_keys - expand_query)) * \
                    hardness_masks.unsqueeze(2) + expand_keys * (1.0 - hardness_masks).unsqueeze(2)

        logits = torch.matmul(torch.matmul(query, self.proj),
                              hard_keys.transpose(-1, -2))  # [batch_size, 1, batch_size]
        return logits


    def soft_copy(self, source_ids, decoder_output, attention, hidden_states, target_embedding):
        p_vocab = self.p_vocab(decoder_output)
        context_vectors = torch.matmul(attention, hidden_states)
        total_states = torch.cat((context_vectors, decoder_output, target_embedding), dim=-1)
        p_gen = self.p_gen(total_states)
        p_copy = 1 - p_gen
        one_hot = torch.zeros(source_ids.size(0), source_ids.size(1), self.vocab_size, device=source_ids.device)
        one_hot = one_hot.scatter_(dim=-1, index=source_ids.unsqueeze(-1), value=1)
        # p_copy from source is sum over all attention weights for each token in source
        p_copy_src_vocab = torch.matmul(attention, one_hot)
        # convert representation of token from src vocab to tgt vocab
        # Compute final probability
        p = torch.add(p_vocab * p_gen, p_copy_src_vocab * p_copy)

        lm_logits = torch.log(p)
        return lm_logits

    def get_ctl_loss_v1(self, src_embedding, trg_embedding, dynamic_coefficient=1, normalize=True, temperature=0.33):
        # src_embedding: [batch_size, hidden_size]
        # trg_embedding: [batch_size, hidden_size]
        src_embedding = src_embedding.unsqueeze(1)  # src_embedding: [batch_size, 1, hidden_size]
        trg_embedding = trg_embedding.unsqueeze(1)  # trg_embedding: [batch_size, 1, hidden_size]
        batch_size = src_embedding.shape[0]
        if normalize:
            src_embedding = F.normalize(src_embedding, p=2, dim=2)
            trg_embedding = F.normalize(trg_embedding, p=2, dim=2)

        def get_ctl_logits(query, keys):
            # expand_query: [batch_size, batch_size, hidden_size]
            # expand_keys: [batch_size, batch_size, hidden_size]
            # the current ref is the positive key, while others in the training batch are negative ones
            expand_query = query.repeat((1, batch_size, 1))
            expand_keys = torch.transpose(keys, 0, 1).repeat((batch_size, 1, 1))

            # distances between queries and positive keys
            d_pos = F.pairwise_distance(query, keys, p=2)
            d_pos = d_pos.repeat(1, batch_size)
            d_neg = F.pairwise_distance(expand_query, expand_keys, p=2)
            # d_pos = tf.sqrt(tf.reduce_sum(input_tensor=tf.pow(query - keys, 2.0), axis=-1))  # [batch_size, 1]
            # d_pos = tf.tile(d_pos, [1, batch_size])  # [batch_size, batch_size]
            # d_neg = tf.sqrt(
            #     tf.reduce_sum(input_tensor=tf.pow(expand_query - expand_keys, 2.0), axis=-1))  # [batch_size, batch_size]

            # lambda_coefficient = (d_pos / d_neg) ** dynamic_coefficient
            lambda_coefficient = (d_pos / d_neg)

            hardness_masks = torch.gt(d_neg, d_pos).int()
            # hardness_masks = tf.cast(tf.greater(d_neg, d_pos), dtype=tf.float32)

            # hard_keys = (expand_query + tf.expand_dims(lambda_coefficient, axis=2) * (
            #             expand_keys - expand_query)) * tf.expand_dims(hardness_masks, axis=2) + \
            #             expand_keys * tf.expand_dims(1.0 - hardness_masks, axis=2)

            hard_keys = (expand_query + lambda_coefficient.unsqueeze(2) * (expand_keys - expand_query)) * \
                        hardness_masks.unsqueeze(2) + expand_keys * (1.0 - hardness_masks).unsqueeze(2)

            logits = torch.matmul(torch.matmul(query, self.proj), hard_keys.transpose(-1, -2))  # [batch_size, 1, batch_size]
            # logits = torch.matmul(query, hard_keys.transpose(-1, -2)) / temperature  # [batch_size, 1, batch_size]

            # logits = tf.matmul(query, hard_keys, transpose_b=True) / temperature  # [batch_size, 1, batch_size]
            return logits

        # logits_src_src = src_embedding.squeeze(1).mm(src_embedding.squeeze(1).transpose(0, 1)).unsqueeze(1) - torch.eye(batch_size).unsqueeze(1).cuda()  # [batch_size, 1, batch_size]
        logits_src_trg = get_ctl_logits(src_embedding, trg_embedding)  # [batch_size, 1, batch_size]
        # logits_src_trg = torch.cosine_similarity(src_embedding.squeeze(1), trg_embedding.squeeze(1).transpose(0, 1)).unsqueeze(1)
        # logits_trg_src = get_ctl_logits(trg_embedding, src_embedding) + \
        #                  tf.expand_dims(tf.linalg.band_part(tf.ones([3, 3]), 0, 0) * -1e9, axis=1)
        logits_trg_src = get_ctl_logits(trg_embedding, src_embedding) + \
                         (torch.eye(batch_size).cuda() * -1e9).unsqueeze(1)
        # logits_trg_src = torch.cosine_similarity(trg_embedding.squeeze(1), src_embedding.squeeze(1).transpose(0, 1)).unsqueeze(1) + \
        #                  (torch.eye(batch_size).cuda() * -1e9).unsqueeze(1)
        # logits_trg_src = torch.matmul(trg_embedding, src_embedding) + (torch.eye(batch_size).cuda() * -1e9).unsqueeze(1)
        # logits_trg_src = get_ctl_logits(trg_embedding, src_embedding)

        # logits = tf.concat([logits_src_trg, logits_trg_src], axis=2)  # [batch_size, 1, 2*batch_size]

        logits = logits_src_trg
        # logits = torch.concat((logits_src_trg, logits_trg_src), dim=2)  # [batch_size, 1, 2*batch_size]

        # labels = tf.expand_dims(tf.range(batch_size, dtype=tf.int32), axis=1)
        # one_hot_labels = tf.one_hot(labels, depth=2 * batch_size, on_value=1.0, off_value=0.0)

        # one_hot_labels = F.one_hot(torch.arange(0, batch_size), num_classes=2 * batch_size).unsqueeze(1)

        # loss = tf.reduce_mean(
        #     input_tensor=tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.stop_gradient(one_hot_labels)))
        loss = F.cross_entropy(logits.squeeze(1), torch.arange(0, batch_size).cuda())

        # contrast_acc = tf.reduce_mean(
        #     input_tensor=tf.cast(tf.equal(tf.argmax(input=logits, axis=2), tf.cast(labels, tf.int64)), tf.float32))

        contrast_acc = torch.mean(
            torch.eq(torch.argmax(logits.squeeze(), dim=-1), torch.arange(0, batch_size).cuda()).float())

        return loss, contrast_acc

    def get_ctl_loss_v2(self, src_embedding, trg_embedding, dynamic_coefficient=1, normalize=True, temperature=0.02):
        # src_embedding: [batch_size, hidden_size]
        # trg_embedding: [batch_size, hidden_size]
        # src_embedding = src_embedding.unsqueeze(1)  # src_embedding: [batch_size, 1, hidden_size]
        # trg_embedding = trg_embedding.unsqueeze(1)  # trg_embedding: [batch_size, 1, hidden_size]
        batch_size = src_embedding.shape[0]
        if normalize:
            src_embedding = F.normalize(src_embedding, p=2, dim=1)
            trg_embedding = F.normalize(trg_embedding, p=2, dim=1)
        anchor_dot_contrast = torch.div(torch.matmul(torch.matmul(src_embedding, self.proj), trg_embedding.T), temperature)
        loss = F.cross_entropy(anchor_dot_contrast, torch.arange(0, batch_size).cuda())
        contrast_acc = torch.mean(
            torch.eq(torch.argmax(anchor_dot_contrast, dim=-1), torch.arange(0, batch_size).cuda()).float())

        return loss, contrast_acc

    def get_semantic_embedding(self, path_ids):
        # path_ids = [batch_size, node_len, seq_len]
        # target_output = [batch_size, tgt_len, emb_size]
        # tgt_mask = [batch_size, tgt_len]
        node_lens = list(path_ids.ne(1)[:, :, 2].sum(1).cpu().numpy())  # [batch_size]
        batch_path_representation = []
        for batch_idx, node_len in enumerate(node_lens):
            # nl_semantic = nl_semantic_output[batch_idx]  # [emb_size]
            # target = target_output[batch_idx:batch_idx+1].repeat(node_len, 1, 1)  #[node_len, tgt_len, emb_size]
            # mask = tgt_mask[batch_idx:batch_idx+1].repeat(node_len, 1)
            # sample_path_ids = path_ids[batch_idx][:node_len]  # [node_len, path_len]
            sample_path_ids = path_ids[batch_idx]
            sample_path_embedding = self.embedding(sample_path_ids)  # [node_len, path_len, emb_size]
            sample_path_mask = sample_path_ids.eq(1)  # [node_len, path_len]
            # sample_output = self.path_encoder(sample_path_embedding, sample_path_mask)  # [node_len, path_len, emb_size]
            sample_output, _ = self.path_encoder(sample_path_embedding)
            # attn_out = self.attn(query=sample_output, key=target, value=target, key_padding_mask=mask, need_weights=False)[0]  # [node_len, path_len, emb_size]
            # sample_output = self.nl_encoder(sample_path_embedding, sample_path_mask)  # [node_len, path_len, emb_size]
            sample_path_mask = ~sample_path_mask.unsqueeze(-1)  # [node_len, path_len]
            # path_representation = attn_out.mul(sample_path_mask).sum(1).div(sample_path_mask.sum(dim=1)).mean(dim=0)  # [emb_size]
            path_representation = sample_output.mul(sample_path_mask).sum(1).div(sample_path_mask.sum(dim=1))  # [node_len, emb_size]
            # weight = F.softmax(path_representation.mm(nl_semantic.unsqueeze(-1)).squeeze(1), dim=0)
            # path_representation = path_representation.mul(weight.unsqueeze(-1)).sum(0)  # [node_len]
            batch_path_representation.append(path_representation)
        return torch.stack(batch_path_representation).cuda()  # [batch_size, node_len, emb_size]

    def get_cooc_loss(self, path_embedding, source_ids, mask):
        path_embedding = self.CBOW_activate1(self.CBOW_linear1(path_embedding))  # [B, S, E]
        path_embedding = self.CBOW_activate2(self.CBOW_linear2(path_embedding))  # [B, S, V]
        path_embedding = path_embedding.view(-1, path_embedding.size(-1))[mask.view(-1)]
        cop_loss = F.nll_loss(path_embedding, source_ids.view(-1)[mask.view(-1)])
        return cop_loss

    def cal_dis(self, path_ids, target_ids=None, temperature=1.0, tasks='gst'):
        path_mask = path_ids.ne(1).unsqueeze(-1)
        source_ids = path_ids[:, :, 0]
        source_mask = source_ids.eq(1)
        semantic_encoder_output = self.embedding(path_ids)
        semantic_encoder_output = semantic_encoder_output.mul(path_mask).sum(2).div(path_mask.sum(dim=2))
        self.eval()
        semantic_encoder_output = self.path_encoder(semantic_encoder_output)
        semantic_encoder_output = semantic_encoder_output.mul((~source_mask).unsqueeze(-1)).sum(1).div(
            (~source_mask).unsqueeze(-1).sum(dim=1))
        target_embedding = self.embedding(target_ids)
        target_embedding = self.pos_encoder(target_embedding)
        tgt_pad_mask = target_ids.eq(1)
        nl_semantic_output = target_embedding.mul((~tgt_pad_mask).unsqueeze(-1)).sum(1).div(
            (~tgt_pad_mask).unsqueeze(-1).sum(dim=1))
        src_embedding = semantic_encoder_output
        trg_embedding = nl_semantic_output
        src_embedding = src_embedding.unsqueeze(1)  # src_embedding: [batch_size, 1, hidden_size]
        trg_embedding = trg_embedding.unsqueeze(1)  # trg_embedding: [batch_size, 1, hidden_size]
        batch_size = src_embedding.shape[0]
        src_embedding = F.normalize(src_embedding, p=2, dim=2)
        trg_embedding = F.normalize(trg_embedding, p=2, dim=2)

        logits_src_trg = self.get_ctl_logits(src_embedding, trg_embedding, batch_size)  # [batch_size, 1, batch_size]
        logits = logits_src_trg.squeeze(1)
        sim_scores = F.softmax(logits, dim=-1)
        loss = F.cross_entropy(logits.squeeze(1), torch.arange(0, batch_size).cuda())

        contrast_acc = torch.mean(
            torch.eq(torch.argmax(logits.squeeze(), dim=-1), torch.arange(0, batch_size).cuda()).float())

        sim_scores = sim_scores.detach().cpu().numpy()
        # print(sim_scores)
        pos = np.diagonal(sim_scores).copy()
        np.random.shuffle(sim_scores)
        neg = np.diagonal(sim_scores)
        return pos, neg
    def cal_dis_base(self, path_ids, target_ids=None, temperature=1.0, tasks='gst'):
        source_ids = path_ids[:, :, 0]
        source_mask = source_ids.eq(1)
        semantic_encoder_output = self.embedding(source_ids)
        semantic_encoder_output = semantic_encoder_output.mul((~source_mask).unsqueeze(-1)).sum(1).div(
            (~source_mask).unsqueeze(-1).sum(dim=1))
        tgt_pad_mask = target_ids.eq(1)
        target_embedding = self.embedding(target_ids)
        target_embedding = self.pos_encoder(target_embedding)
        nl_semantic_output = target_embedding.mul((~tgt_pad_mask).unsqueeze(-1)).sum(1).div(
            (~tgt_pad_mask).unsqueeze(-1).sum(dim=1))
        src_embedding = semantic_encoder_output
        trg_embedding = nl_semantic_output
        # src_embedding = src_embedding.unsqueeze(1)  # src_embedding: [batch_size, 1, hidden_size]
        # trg_embedding = trg_embedding.unsqueeze(1)  # trg_embedding: [batch_size, 1, hidden_size]
        batch_size = src_embedding.shape[0]
        src_embedding = F.normalize(src_embedding, p=2, dim=1)
        trg_embedding = F.normalize(trg_embedding, p=2, dim=1)
        logits = torch.matmul(src_embedding, trg_embedding.T)
        # logits_src_trg = self.get_ctl_logits(src_embedding, trg_embedding)  # [batch_size, 1, batch_size]
        # logits = logits_src_trg.squeeze(1)
        sim_scores = F.softmax(logits, dim=-1)
        loss = F.cross_entropy(logits.squeeze(1), torch.arange(0, batch_size).cuda())

        contrast_acc = torch.mean(
            torch.eq(torch.argmax(logits.squeeze(), dim=-1), torch.arange(0, batch_size).cuda()).float())

        sim_scores = sim_scores.detach().cpu().numpy()
        # print(sim_scores)
        pos = np.diagonal(sim_scores).copy()
        np.random.shuffle(sim_scores)
        neg = np.diagonal(sim_scores)
        return pos, neg


    def forward(self, path_ids, target_ids=None, temperature=1.0, tasks='gst'):
        if target_ids is None:
            return self.generate(path_ids)

        # source_embedding=0
        # source_mask=0
        # token_encoder_output = self.token_encoder(source_embedding, source_mask)  # [batch_size, seq_len, emb_size]
        # semantic_encoder_output = source_embedding.unsqueeze(0)
        # semantic_encoder_output = self.path_encoder(semantic_encoder_output)


        # mask = source_ids.ne(1)[:, None, :] * source_ids.ne(1)[:, :, None]
        # target_mask = target_ids.ne(1)[:, None, :] * target_ids.ne(1)[:, :, None]
        # path_mask = path_ids.ne(1)
        tgt_pad_mask = target_ids.eq(1)
        # path_mask = path_ids[:,:,0].eq(1)  # [batch_size, seq_len]
        # nl_semantic_output = self.nl_encoder(target_ids)  # [batch_size, emb_size]
        source_ids = path_ids[:,:,0]
        source_mask = source_ids.eq(1)
        source_embedding = self.embedding(source_ids)  # [batch_size, seq_len, emb_size]
        target_embedding = self.embedding(target_ids)  # [batch_size, target_len, emb_size]
        # nl_semantic_output = self.path_encoder(target_embedding, tgt_pad_mask)  # [batch_size, seq_len, emb_size]
        # semantic_encoder_output = self.get_semantic_embedding(path_ids)  # [batch_size, seq_len, emb_size]

        # path_mask = path_ids.ne(1).unsqueeze(-1)  # [B, S, path_len, 1]
        # semantic_encoder_output = self.embedding(path_ids)  # [B, S, path_len, E]
        # semantic_encoder_output = semantic_encoder_output.mul(path_mask).sum(2).div(path_mask.sum(dim=2))  # [B, S, E]
        # cop_loss = self.get_cooc_loss(semantic_encoder_output, source_ids, ~source_mask)

        # semantic_encoder_output = semantic_encoder_output.mul(path_mask).sum(2).div(path_mask.sum(dim=2))
        if tasks == 'clt':
            nl_semantic_output = nl_semantic_output.mul((~tgt_pad_mask).unsqueeze(-1)).sum(1).div((~tgt_pad_mask).unsqueeze(-1).sum(dim=1))  # [batch_size, emb_size]
            clt_loss, contrast_acc = get_ctl_loss(semantic_encoder_output, nl_semantic_output, temperature=temperature)
            # plot_embedding(semantic_encoder_output.cpu().detach().numpy(), nl_semantic_output.cpu().detach().numpy())
            return clt_loss, contrast_acc
        if tasks == 'gst':
            # source_embedding = self.pos_encoder(source_embedding)
            source_embedding = self.pos_encoder(source_embedding)
            target_embedding = self.pos_encoder(target_embedding)
            token_encoder_output = self.token_encoder(source_embedding, source_mask)  # [batch_size, seq_len, emb_size]
            # token_encoder_output = token_encoder_output * self.alpha + semantic_encoder_output * (1 - self.alpha)


            # clt_loss
            # nl_semantic_output = target_embedding.mul((~tgt_pad_mask).unsqueeze(-1)).sum(1).div((~tgt_pad_mask).unsqueeze(-1).sum(dim=1))  # [batch_size, emb_size]
            # semantic_encoder_output = self.path_encoder(semantic_encoder_output)  # [B, S, E]
            # semantic_encoder_output = semantic_encoder_output.mul((~source_mask).unsqueeze(-1)).sum(1).div((~source_mask).unsqueeze(-1).sum(dim=1))  # [batch_size, emb_size]
            # # clt_loss, contrast_acc = self.get_ctl_loss_v2(semantic_encoder_output, nl_semantic_output,
            # #                                               temperature=temperature)
            # clt_loss, contrast_acc = self.get_ctl_loss_v1(semantic_encoder_output, nl_semantic_output, temperature=temperature)

            # clt_loss, contrast_acc = get_ctl_loss(semantic_encoder_output, nl_semantic_output, temperature=temperature)
            # token_encoder_output = torch.add(token_encoder_output, semantic_encoder_output, alpha=0.2)
            # context = torch.add(token_encoder_output, torch.nan_to_num(semantic_encoder_output), alpha=0.1)
            # context = torch.concat((token_encoder_output, semantic_encoder_output), dim=1).transpose(1, 2)  # [batch_size, E, seq_len+node_len]
            # decoder_input = self.attn(query=semantic_encoder_output,
            #                           key=token_encoder_output,
            #                           value=token_encoder_output,
            #                           key_padding_mask=source_mask,
            #                           need_weights=False)[0]
            # token_encoder_output_weitht = F.softmax((semantic_encoder_output.unsqueeze(1)).bmm(token_encoder_output.transpose(1, 2)), dim=-1).transpose(1, 2)
            # token_encoder_output = token_encoder_output_weitht * token_encoder_output
            # token_encoder_output = token_encoder_output.add(semantic_encoder_output.unsqueeze(1))

            # token_decoder_output = self.decoder(target_embedding,
            #                                     decoder_input,
            #                                     tgt_mask=attn_mask,
            #                                     tgt_key_padding_mask=tgt_pad_mask,
            #                                     memory_key_padding_mask=source_mask,
            #                                     semantic=semantic_encoder_output)
            attn_mask = torch.triu(torch.full((target_ids.shape[1], target_ids.shape[1]), bool('False')),
                                   diagonal=1).cuda()
            decoder_output, attention = self.decoder(target_embedding,
                                                     token_encoder_output,
                                                     tgt_mask=attn_mask,
                                                     tgt_key_padding_mask=tgt_pad_mask,
                                                     memory_key_padding_mask=source_mask)
            lm_logits = self.soft_copy(source_ids, decoder_output, attention, token_encoder_output, target_embedding)  # [batch_size, seq_len, vacab_size]
            # lm_logits = self.lm_head(token_decoder_output)  # [batch_size, seq_len, vacab_size]
            # Shift so that tokens < n predict n
            active_loss = target_ids[..., 1:].ne(1).view(-1)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                            shift_labels.view(-1)[active_loss])
            # outputs = 0.1 * cop_loss, loss, loss * active_loss.sum(), active_loss.sum()
            # outputs = cop_loss, loss, clt_loss, contrast_acc, loss * active_loss.sum(), active_loss.sum()
            outputs = loss, loss * active_loss.sum(), active_loss.sum()
            # outputs = loss
            return outputs
        # path_representation = self.path_encoder(path_embedding, path_mask)  # [batch_size, node_num, path_len, emb_size]
        # node_mask = path_ids.ne(1)[:, :, 0].unsqueeze(-1)
        # path_representation = self.proj(path_embedding)  # [batch_size, node_len, seq_len, hidden_dim]
        # path_representation = path_representation.mul(node_mask).sum(1).div(node_mask.sum(dim=1))  # [batch_size, seq_len, emb_size]

        # path_representation = path_representation.mean(dim=1)  # [batch_size, hidden_dim]

        # if tasks == 'clt':
        #     target_representation = self.nl_encoder(target_ids)  # [batch_size, hidden_dim]
        #     clt_loss, contrast_acc = get_ctl_loss(path_representation, target_representation)
        #     return clt_loss, contrast_acc
        #
        # if tasks == 'csg':
        #     ids = torch.cat((source_ids, target_ids), -1)
        #     mask = self.bias[:, source_ids.size(-1):ids.size(-1), :ids.size(-1)].bool()
        #     mask = mask & ids[:, None, :].ne(1)
        #
        #     input_feature = torch.cat((token_hn, path_representation))
        #     state, _ = self.decoder(target_embedding, token_hn)
        #     lm_logits = self.lm_head(state)
        #     # Shift so that tokens < n predict n
        #     active_loss = target_ids[..., 1:].ne(1).view(-1)
        #     shift_logits = lm_logits[..., :-1, :].contiguous()
        #     shift_labels = target_ids[..., 1:].contiguous()
        #     # Flatten the tokens
        #     loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        #     loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
        #                     shift_labels.view(-1)[active_loss])
        #
        #     outputs = loss, loss * active_loss.sum(), active_loss.sum()
        #     # outputs = loss
        #     return outputs

    def generate(self, path_ids):
        source_ids = path_ids[:,:,0]
        source_mask = source_ids.eq(1)

        source_embedding = self.embedding(source_ids)
        source_embedding = self.pos_encoder(source_embedding)
        token_encoder_output = self.token_encoder(source_embedding, source_mask)
        # semantic_encoder_output = self.get_semantic_embedding(path_ids)

        path_mask = path_ids.ne(1).unsqueeze(-1)
        semantic_encoder_output = self.embedding(path_ids)
        semantic_encoder_output = semantic_encoder_output.mul(path_mask).sum(2).div(path_mask.sum(dim=2))
        token_encoder_output = token_encoder_output * self.alpha + semantic_encoder_output * (1 - self.alpha)
        semantic_encoder_output = self.path_encoder(semantic_encoder_output)
        semantic_encoder_output = semantic_encoder_output.mul((~source_mask).unsqueeze(-1)).sum(1).div((~source_mask).unsqueeze(-1).sum(dim=1))

        # token_encoder_output = self.token_encoder(semantic_encoder_output, source_mask)

        # token_encoder_output = torch.add(token_encoder_output, semantic_encoder_output, alpha=0.2)
        # semantic_encoder_output = self.get_semantic_embedding(path_ids)
        preds = []
        zero = torch.cuda.LongTensor(1).fill_(0)
        # source_len = list(source_ids.ne(1).sum(-1).cpu().numpy())
        for i in range(source_ids.shape[0]):
            beam = Beam(self.beam_size, self.sos_id, self.eos_id)
            input_ids = beam.getCurrentState()
            memory = token_encoder_output[i:i+1].repeat(self.beam_size, 1, 1)  # [beam_size, seq_len, emb_size]
            memory_pad_mask = source_mask[i:i+1].repeat(self.beam_size, 1)  # [beam_size, seq_len]
            ids = source_ids[i:i+1].repeat(self.beam_size, 1)
            # semantic = semantic_encoder_output[i:i+1].repeat(self.beam_size, 1, 1)  # [beam_size, seq_len, emb_size]
            # memory = torch.add(memory, torch.nan_to_num(semantic), alpha=self.alpha.data)
            # semantic = semantic_encoder_output[i:i+1].repeat(self.beam_size, 1)  # [beam_size, emb_size]
            # context_ids = source_ids[i:i + 1, :source_len[i]].repeat(self.beam_size, 1)
            for j in range(self.max_length):
                if beam.done():
                    break
                # ids = torch.cat((context_ids, input_ids), -1)
                input_embedding = self.embedding(input_ids)  # [beam_size, tgt_len, emb_size]
                input_embedding = self.pos_encoder(input_embedding)  # [beam_size, tgt_len, emb_size]
                # target_output = self.path_encoder(input_embedding)  # [beam_size, tgt_len, emb_size]
                # path_id = path_ids[i:i+1].repeat(self.beam_size, 1, 1)  # [beam_size, node_num, seq_len]
                # semantic_encoder_output = self.get_semantic_embedding(path_id, target_output, input_ids.eq(1))
                # memory.add(semantic_encoder_output.unsqueeze(1))
                # out = self.decoder(input_embedding,
                #                    memory=memory,
                #                    memory_key_padding_mask=memory_pad_mask,
                #                    semantic=semantic_encoder_output)  # [beam_size, seq_len, emb_size]
                decoder_output, attention = self.decoder(input_embedding,
                                                         memory=memory,
                                                         memory_key_padding_mask=memory_pad_mask)
                p_vocab = self.p_vocab(decoder_output)
                context_vectors = torch.matmul(attention, memory)
                total_states = torch.cat((context_vectors, decoder_output, input_embedding), dim=-1)
                p_gen = self.p_gen(total_states)
                p_copy = 1 - p_gen
                # input_ids = source_ids[i:i+1].repeat(self.beam_size, 1)
                one_hot = torch.zeros(ids.size(0), ids.size(1), self.vocab_size, device=ids.device)
                one_hot = one_hot.scatter_(dim=-1, index=ids.unsqueeze(-1), value=1)
                # p_copy from source is sum over all attention weights for each token in source
                p_copy_src_vocab = torch.matmul(attention, one_hot)
                # convert representation of token from src vocab to tgt vocab
                # p_copy_tgt_vocab = torch.matmul(p_copy_src_vocab, self.src_to_tgt_vocab_conversion_matrix).transpose(0, 1)
                # Compute final probability
                p = torch.add(p_vocab * p_gen, p_copy_src_vocab * p_copy)

                # Change back batch and sequence dimensions, from (T, N, tgt_vocab_size) -> (N, T, tgt_vocab_size)
                # return torch.log(p.transpose(0, 1))
                out = torch.log(p[:, -1, :]).data
                # hidden_states = out[:, -1, :]
                # out = self.lsm(self.lm_head(hidden_states)).data
                beam.advance(out)
                input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                input_ids = torch.cat((input_ids, beam.getCurrentState()), -1)

            # hyps = beam.getHyp(beam.getFinal())

            scores, hyps = beam.new_getHyp(beam.getFinal())
            all_nl_semantic_output = []
            for hyp in hyps:
                target_ids = torch.Tensor(hyp).long().cuda()
                target_emb = self.embedding(target_ids)
                nl_semantic_output = target_emb.mean(dim=0)
                all_nl_semantic_output.append(nl_semantic_output)
            all_nl_semantic_output = torch.stack(all_nl_semantic_output)
            x_semantic = semantic_encoder_output[i:i+1]
            src_embedding = x_semantic.unsqueeze(1)  # src_embedding: [batch_size, 1, hidden_size]
            trg_embedding = all_nl_semantic_output.unsqueeze(1)  # trg_embedding: [batch_size, 1, hidden_size]
            src_embedding = F.normalize(src_embedding, p=2, dim=2)
            trg_embedding = F.normalize(trg_embedding, p=2, dim=2)
            batch_size = all_nl_semantic_output.shape[0]
            src_embedding = src_embedding.repeat(batch_size, 1, 1)
            logits_src_trg = self.get_ctl_logits(src_embedding, trg_embedding, batch_size)  # [batch_size, 1, batch_size]
            logits = logits_src_trg.squeeze(1)
            sim_scores = F.softmax(logits, dim=-1).cpu().numpy()
            sim_scores = sim_scores.diagonal().copy().tolist()
            # sim_scores = F.softmax(torch.matmul(torch.matmul(x_semantic, self.proj), all_nl_semantic_output.transpose(-1, -2)), dim=1).squeeze().tolist()
            # sim_scores = F.softmax(
            #     torch.matmul(x_semantic, all_nl_semantic_output.transpose(-1, -2)),
            #     dim=1).squeeze().tolist()
            final_scores = list(map(lambda x: (1-self.gamma)*x[0] + self.gamma*x[1], zip(scores, sim_scores)))
            resort = list(zip(final_scores, hyps))
            resort.sort(key=lambda a:-a[0])
            resort_hyps = [a[1] for a in resort]
            pred = beam.buildTargetTokens(resort_hyps)[:self.beam_size]

            # pred = beam.buildTargetTokens(hyps)[:self.beam_size]
            pred = [torch.cat([x.view(-1) for x in p] + [zero] * (self.max_length - len(p))).view(1, -1) for p in pred]
            preds.append(torch.cat(pred, 0).unsqueeze(0))

        preds = torch.cat(preds, 0)

        return preds
