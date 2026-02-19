__all__ = ['PatchTST_backbone']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

#from collections import OrderedDict
from layers.PatchTST_layers import *
from layers.RevIN import RevIN
class MultiScalePatchTST_backbone(nn.Module):
    def __init__(self, c_in:int, context_window:int, target_window:int, patch_lens:list, strides:list, max_seq_len:Optional[int]=1024, 
                 n_layers:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
                 pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False,
                 verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # 1. RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        
        # 2. Multi-Scale Config
        self.patch_lens = patch_lens  # 예: [16, 32, 48]
        if strides is None:
            self.strides = [pl // 2 for pl in patch_lens]
        elif isinstance(strides, int):
            self.strides = [strides for _ in patch_lens]
        else:
            self.strides = list(strides)
        if len(self.strides) != len(self.patch_lens):
            raise ValueError("len(strides) must match len(patch_lens)")

        self.base_stride = self.strides[0]
        if self.base_stride <= 0:
            raise ValueError("stride must be a positive integer")
        if any(s <= 0 for s in self.strides):
            raise ValueError("all strides must be positive integers")
        self.padding_patch = padding_patch
        
        # 기준이 되는 가장 작은 patch length (Center Alignment의 기준점)
        base_patch_len = patch_lens[0] 
        self.base_patch_len = base_patch_len

        # Stage-2: scale마다 patch 개수를 별도로 계산해 패딩 토큰 폭증을 방지
        self.patch_nums = [
            self._compute_patch_num(context_window, p_len, stride)
            for p_len, stride in zip(self.patch_lens, self.strides)
        ]
        self.base_patch_num = self.patch_nums[0]
        self.max_patch_num = max(self.patch_nums)
        self.total_patch_num = sum(self.patch_nums)
            
        # 3. Projections & Embeddings
        # 각 스케일별로 다른 Projection Layer를 가짐 (길이가 달라도 d_model로 통일)
        self.W_P = nn.ModuleList([
            nn.Linear(pl, d_model) for pl in patch_lens
        ])
        
        # Scale Embedding: 각 스케일 ID에 대한 임베딩 (0: scale1, 1: scale2 ...)
        self.scale_embedding = nn.Embedding(len(patch_lens), d_model)
        # Continuous patch-length embedding (e.g. log2 ratio to base patch length)
        self.patch_len_proj = nn.Linear(1, d_model)
        self.len_alpha = nn.Parameter(torch.zeros(1))
        patch_len_feats = [[np.log2(pl / self.base_patch_len)] for pl in self.patch_lens]
        self.register_buffer(
            "patch_len_features", torch.tensor(patch_len_feats, dtype=torch.float32), persistent=False
        )
        # Ablation switches (toggle by uncommenting one line)
        self.use_scale_embedding = True
        # self.use_scale_embedding = False  # disable scale-id embedding
        self.use_len_gate = True
        # self.use_len_gate = False         # use len_emb without len_alpha gate
        self.use_cross_gate = True
        # self.use_cross_gate = False       # use cross_pos_emb without cross_alpha gate

        # Cross-scale relative position embedding:
        # maps each scale token to its base-scale center index.
        self.cross_pos_embedding = nn.Embedding(self.base_patch_num, d_model)
        self.cross_alpha = nn.Parameter(torch.zeros(1))

        # 4. Backbone (Shared Transformer Encoder)
        # Encoder는 가변 토큰 길이를 처리하므로, PE 테이블은 최대 patch_num 기준으로 생성한다.
        self.backbone = TSTiEncoder(c_in, patch_num=self.max_patch_num, patch_len=base_patch_len, max_seq_len=max_seq_len,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        # 5. Head
        # Flatten Head의 입력 차원은 (sum(scale별 patch_num) * d_model)
        self.head_nf = d_model * self.total_patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head: 
            self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout)
        elif head_type == 'flatten': 
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)
        
    def _compute_patch_num(self, seq_len:int, patch_len:int, stride:int) -> int:
        # Center alignment를 위해 좌측 패딩을 고려한 scale별 토큰 개수 계산
        pad_left = (patch_len - self.base_patch_len) // 2
        pad_right = stride if self.padding_patch == 'end' else 0
        padded_len = seq_len + pad_left + pad_right
        if padded_len < patch_len:
            return 1
        return int((padded_len - patch_len) / stride + 1)
    
    
    def forward(self, z):                                                                   # z: [bs x nvars x seq_len]
        # norm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)
            
        # Multi-Scale Patching & Projection
        bs, n_vars, seq_len = z.shape
        scale_tokens = []
        
        for i, (p_len, stride, patch_num, proj) in enumerate(zip(self.patch_lens, self.strides, self.patch_nums, self.W_P)):
            # -----------------------------------------------------------
            # Center Alignment Logic
            # -----------------------------------------------------------
            # 목표: Scale이 커져도 i번째 패치의 중심(Center)이 Scale 0의 i번째 패치 중심과 같아야 함.
            # 공식: Pad_Left = (Current_Len - Base_Len) // 2
            # 이렇게 하면 더 긴 패치가 왼쪽으로 확장되어 중심을 유지함.
            pad_left = (p_len - self.base_patch_len) // 2
            
            # unfold가 정상 작동하기 위한 총 필요 길이 계산 (scale별 patch_num 사용)
            req_len = (patch_num - 1) * stride + p_len
            pad_total = req_len - seq_len
            pad_right = pad_total - pad_left
            pad_right = max(0, pad_right)
            
            # 패딩 수행 (좌/우 비대칭 패딩일 수 있음)
            # F.pad는 (Left, Right) 순서
            z_padded = F.pad(z, (pad_left, pad_right)) # [bs, nvars, padded_len]
            
            # Patching
            patches = z_padded.unfold(dimension=-1, size=p_len, step=stride) # [bs, nvars, patch_num_i, p_len]
            patches = patches[:, :, :patch_num, :]
            patches = patches.permute(0,1,3,2)  # [bs, nvars, p_len, patch_num_i]
            
            # Projection
            patches = patches.permute(0,1,3,2)   # [bs, nvars, patch_num_i, p_len]
            emb = proj(patches)                  # [bs, nvars, patch_num_i, d_model]
            
            # Add Encodings
            # 1. Positional Encoding (Time): backbone.W_pos 사용 (모든 스케일이 시간 위치는 공유)
            # 2. Scale Encoding (Resolution): i번째 스케일 임베딩 더하기
            
            # backbone.W_pos shape: [max_patch_num, d_model]
            pos_emb = self.backbone.W_pos[:patch_num]
            if self.use_scale_embedding:
                scale_emb = self.scale_embedding.weight[i] # [d_model]
            else:
                scale_emb = 0.0
            len_feat = self.patch_len_features[i:i+1].to(dtype=emb.dtype)  # [1, 1]
            len_emb = self.patch_len_proj(len_feat).squeeze(0)              # [d_model]
            center_idx = torch.arange(patch_num, device=z.device, dtype=torch.float32)
            center_idx = torch.round(center_idx * (stride / self.base_stride)).long()
            valid = center_idx < self.base_patch_num
            center_idx = center_idx.clamp(min=0, max=self.base_patch_num - 1)
            cross_pos_emb = self.cross_pos_embedding(center_idx)  # [patch_num_i, d_model]
            valid = valid.unsqueeze(-1).to(cross_pos_emb.dtype)
            cross_pos_emb = cross_pos_emb * valid
            if self.use_len_gate:
                len_term = self.len_alpha * len_emb
            else:
                len_term = len_emb
            if self.use_cross_gate:
                cross_term = self.cross_alpha * cross_pos_emb
            else:
                cross_term = cross_pos_emb
            
            # Token Sum
            out = emb + pos_emb + scale_emb + len_term + cross_term
            scale_tokens.append(out)

        # Concat all scales
        # 모든 스케일의 토큰을 sequence 차원(dim=2)으로 연결 -> Transformer가 한 번에 처리
        # z shape: [bs, nvars, sum(scale별 patch_num), d_model]
        z = torch.cat(scale_tokens, dim=2) 
        
        # Transpose for Transformer [bs * nvars, seq_len, d_model] logic in backbone
        z = z.permute(0,1,3,2) # [bs, nvars, d_model, total_tokens]
        
        # Model (Transformer Encoder)
        # TSTiEncoder는 입력을 [bs, nvars, d_model, patch_num] 형태로 받아서 처리함
        # 내부적으로 reshape해서 [bs*nvars, patch_num, d_model]로 만듦
        # 여기서는 patch_num 대신 total_tokens가 들어감
        z = self.backbone.encoder_forward_custom(z) # *아래 TSTiEncoder 수정 필요*
                                                                 
        # Head
        z = self.head(z) # z: [bs x nvars x target_window] 
        
        # denorm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0,2,1)
        return z
    
# Cell
class PatchTST_backbone(nn.Module):
    def __init__(self, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int, max_seq_len:Optional[int]=1024, 
                 n_layers:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
                 pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False,
                 verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len)/stride + 1)
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            patch_num += 1
        
        # Backbone 
        self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head: 
            self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout) # custom head passed as a partial func with all its kwargs
        elif head_type == 'flatten': 
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)
        
    
    def forward(self, z):                                                                   # z: [bs x nvars x seq_len]
        # norm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)
            
        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0,1,3,2)                                                              # z: [bs x nvars x patch_len x patch_num]
        
        # model
        z = self.backbone(z)                                                                # z: [bs x nvars x d_model x patch_num]
        z = self.head(z)                                                                    # z: [bs x nvars x target_window] 
        
        # denorm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0,2,1)
        return z
    
    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                    nn.Conv1d(head_nf, vars, 1)
                    )


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x
        
        
    
    
class TSTiEncoder(nn.Module):  #i means channel-independent
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):
        
        
        super().__init__()
        
        self.patch_num = patch_num
        self.patch_len = patch_len
        
        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)        # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)

       
    def forward(self, x) -> Tensor:                                              # x: [bs x nvars x patch_len x patch_num]
        
        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0,1,3,2)                                                   # x: [bs x nvars x patch_num x patch_len]
        x = self.W_P(x)                                                          # x: [bs x nvars x patch_num x d_model]

        u = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))      # u: [bs * nvars x patch_num x d_model]
        u = self.dropout(u + self.W_pos)                                         # u: [bs * nvars x patch_num x d_model]

        # Encoder
        z = self.encoder(u)                                                      # z: [bs * nvars x patch_num x d_model]
        z = torch.reshape(z, (-1,n_vars,z.shape[-2],z.shape[-1]))                # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0,1,3,2)                                                   # z: [bs x nvars x d_model x patch_num]
        
        return z    

    def encoder_forward_custom(self, x):
        # x: [bs, nvars, d_model, total_tokens]
        n_vars = x.shape[1]
        total_tokens = x.shape[3]
        
        x = x.permute(0,1,3,2) # [bs, nvars, total_tokens, d_model]
        u = torch.reshape(x, (x.shape[0]*x.shape[1], total_tokens, x.shape[3])) # [bs*nvars, total_tokens, d_model]
        
        # Dropout (Encoder 들어가기 전)
        u = self.dropout(u)
        
        # Encoder
        z = self.encoder(u) # [bs*nvars, total_tokens, d_model]
        
        z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1])) # [bs, nvars, total_tokens, d_model]
        z = z.permute(0,1,3,2) # [bs, nvars, d_model, total_tokens]
        return z
            
            
    
# Cell
class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output



class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src




class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights
