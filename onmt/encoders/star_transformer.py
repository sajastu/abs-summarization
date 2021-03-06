"""
Implementation of "Attention is All You Need"
"""
import torch
from torch import nn
from torch.nn import functional as F
import numpy as NP
from torch.autograd import Variable


from onmt.encoders.encoder import EncoderBase

class StarTransformerEncoder(EncoderBase):
    """The Transformer encoder from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O

    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings

    Returns:
        (torch.FloatTensor, torch.FloatTensor):

        * embeddings ``(src_len, batch_size, model_dim)``
        * memory_bank ``(src_len, batch_size, model_dim)``
    """

    def __init__(self, num_layers, d_model, heads, d_ff, dropout, embeddings,
                 max_relative_positions=0):
        super(StarTransformerEncoder, self).__init__()

        super(StarTransformerEncoder, self).__init__()
        self.iters = num_layers
        self.embeddings = embeddings
        self.norm = nn.ModuleList([nn.LayerNorm(d_model, eps=1e-6) for _ in range(self.iters)])
        self.ring_att = nn.ModuleList(
            [MSA1(d_model, nhead=heads, dropout=dropout)
             for _ in range(self.iters)])
        self.star_att = nn.ModuleList(
            [MSA2(d_model, nhead=heads, dropout=dropout)
             for _ in range(self.iters)])

        if max_relative_positions != 0:
            self.pos_emb = self.pos_emb = nn.Embedding(max_relative_positions, d_model)
        else:
            self.pos_emb = None

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.dropout,
            embeddings,
            opt.max_relative_positions)

    def forward(self, src, lengths=None):
        """See :func:`EncoderBase.forward()`
            src: LongTensor: (len, batch, features)
        """
        def norm_func(f, x):
            # B, H, L, 1
            return f(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        self._check_args(src, lengths)
        emb = self.embeddings(src)

        out = emb.transpose(0, 1).contiguous()  # data: (B, L, H)
        # words = src[:, :, 0].transpose(0, 1)  # words: (B, L)
        # w_batch, w_len = words.size()
        # padding_idx = self.embeddings.word_padding_idx
        # mask = words.data.eq(padding_idx).byte()  # [B, L]

        mask = seq_len_to_byte_mask(lengths)
        B, L, H = out.size()  # (B, L, H)
        mask = (mask == 0) # flip the mask for masked _fill_

        smask = torch.cat([torch.zeros(B, 1, ).byte().to(mask), mask], 1)

        embs = out.permute(0, 2, 1)[:, :, :, None]  # B H L 1

        if self.pos_emb:
            print('Hey! I\'m in pos emb!!!')
            P = self.pos_emb(torch.arange(L, dtype=torch.long, device=embs.device) \
                             .view(1, L)).permute(0, 2, 1).contiguous()[:, :, :, None]  # 1 H L 1
            embs = embs + P

        nodes = embs  # H^0 = E, [B, H, L, 1]

        relay = embs.mean(2, keepdim=True)  # s^0 = average(E), [B, H, 1, 1]

        ex_mask = mask[:, None, :, None].expand(B, H, L, 1)
        # import pdb; pdb.set_trace()

        r_embs = embs.view(B, H, 1, L)
        for i in range(self.iters):
            ax = torch.cat([r_embs, relay.expand(B, H, 1, L)], 2)  # context vector
            nodes = nodes + F.leaky_relu(self.ring_att[i](norm_func(self.norm[i], nodes), ax=ax))
            relay = F.leaky_relu(self.star_att[i](relay, torch.cat([relay, nodes], 2), smask))
            nodes = nodes.masked_fill_(ex_mask, 0)
        # import pdb; pdb.set_trace()
        nodes = nodes.view(B, H, L).permute(0, 2, 1) # B L H
        # return self.embedding(data), nodes, relay.view(B, H)
        # out should be L B H
        return emb, nodes.transpose(0, 1).contiguous(), lengths


class MSA1(nn.Module):
    def __init__(self, model_dim, nhead=10, dropout=0.1):
        super(MSA1, self).__init__()
        assert model_dim % nhead == 0
        head_dim = model_dim // nhead

        # Multi-head Self Attention Case 1, doing self-attention for small regions
        # Due to the architecture of GPU, using hadamard production and summation are faster than dot production when unfold_size is very small
        self.WQ = nn.Conv2d(model_dim, nhead * head_dim, 1)
        self.WK = nn.Conv2d(model_dim, nhead * head_dim, 1)
        self.WV = nn.Conv2d(model_dim, nhead * head_dim, 1)
        self.WO = nn.Conv2d(nhead * head_dim, model_dim, 1)

        self.drop = nn.Dropout(dropout)

        # print('NUM_HEAD', nhead, 'DIM_HEAD', head_dim)
        self.model_dim, self.nhead, self.head_dim, self.unfold_size = model_dim, nhead, head_dim, 3

    def forward(self, x, ax=None):
        # x: B, H, L, 1, ax : B, H, X, L append features
        model_dim, nhead, head_dim, unfold_size = self.model_dim, self.nhead, self.head_dim, self.unfold_size
        B, H, L, _ = x.shape

        q, k, v = self.WQ(x), self.WK(x), self.WV(x)  # x: (B,H,L,1)

        if ax is not None:
            aL = ax.shape[2]
            ak = self.WK(ax).view(B, nhead, head_dim, aL, L)
            av = self.WV(ax).view(B, nhead, head_dim, aL, L)
        q = q.view(B, nhead, head_dim, 1, L)
        k = F.unfold(k.view(B, nhead * head_dim, L, 1), (unfold_size, 1), padding=(unfold_size // 2, 0)) \
            .view(B, nhead, head_dim, unfold_size, L)
        v = F.unfold(v.view(B, nhead * head_dim, L, 1), (unfold_size, 1), padding=(unfold_size // 2, 0)) \
            .view(B, nhead, head_dim, unfold_size, L)
        if ax is not None:
            k = torch.cat([k, ak], 3)
            v = torch.cat([v, av], 3)

        alphas = self.drop(F.softmax((q * k).sum(2, keepdim=True) / NP.sqrt(head_dim), 3))  # B N L 1 U
        att = (alphas * v).sum(3).view(B, nhead * head_dim, L, 1)

        ret = self.WO(att)

        return ret


class MSA2(nn.Module):
    def __init__(self, nhid, nhead=10, dropout=0.1):
        # Multi-head Self Attention Case 2, a broadcastable query for a sequence key and value
        super(MSA2, self).__init__()
        assert nhid % nhead == 0
        head_dim = nhid // nhead
        self.WQ = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WK = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WV = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WO = nn.Conv2d(nhead * head_dim, nhid, 1)

        self.drop = nn.Dropout(dropout)

        print('NUM_HEAD', nhead, 'DIM_HEAD', head_dim)
        self.nhid, self.nhead, self.head_dim, self.unfold_size = nhid, nhead, head_dim, 3

    def forward(self, x, y, mask=None):
        # x: B, H, 1, 1, 1 y: B H L 1
        nhid, nhead, head_dim, unfold_size = self.nhid, self.nhead, self.head_dim, self.unfold_size
        B, H, L, _ = y.shape

        q, k, v = self.WQ(x), self.WK(y), self.WV(y)

        q = q.view(B, nhead, 1, head_dim)  # B, H, 1, 1 -> B, N, 1, h
        k = k.view(B, nhead, head_dim, L)  # B, H, L, 1 -> B, N, h, L
        v = v.view(B, nhead, head_dim, L).permute(0, 1, 3, 2)  # B, H, L, 1 -> B, N, L, h
        pre_a = torch.matmul(q, k) / NP.sqrt(head_dim)
        if mask is not None:
            pre_a = pre_a.masked_fill(mask[:, None, None, :], -1e18)
        alphas = self.drop(F.softmax(pre_a, 3))  # B, N, 1, L
        att = torch.matmul(alphas, v).view(B, -1, 1, 1)  # B, N, 1, h -> B, N*h, 1, 1
        return self.WO(att)

def seq_len_to_byte_mask(seq_lens):
    # usually seq_lens: LongTensor, batch_size
    # return value: ByteTensor, batch_size x max_len
    batch_size = seq_lens.size(0)
    max_len = seq_lens.max()
    broadcast_arange = torch.arange(max_len).view(1, -1).repeat(batch_size, 1).to(seq_lens.device)
    mask = broadcast_arange.float().lt(seq_lens.float().view(-1, 1))
    return mask