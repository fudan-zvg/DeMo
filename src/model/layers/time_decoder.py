import torch
import torch.nn as nn
from .transformer_blocks import Cross_Block, Block
import torch.nn.functional as F
from .mamba.vim_mamba import init_weights, create_block
from functools import partial
from timm.models.layers import DropPath, to_2tuple
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class GMMPredictor_dense(nn.Module):
    def __init__(self, future_len=60, dim=128):
        super(GMMPredictor_dense, self).__init__()
        self._future_len = future_len
        self.gaussian = nn.Sequential(
            nn.Linear(dim, 64), 
            nn.GELU(), 
            nn.Linear(64, 2)
        )
        self.score = nn.Sequential(
            nn.Linear(dim, 64), 
            nn.GELU(), 
            nn.Linear(64, 1),
        )
        self.scale = nn.Sequential(
            nn.Linear(dim, 64), 
            nn.GELU(), 
            nn.Linear(64, 2)
        )
    
    def forward(self, input):
        res = self.gaussian(input)
        scal = F.elu_(self.scale(input), alpha=1.0) + 1.0 + 0.0001
        input = input.max(dim=2)[0]  
        score = self.score(input).squeeze(-1)

        return res, score, scal


class GMMPredictor(nn.Module):
    def __init__(self, future_len=60, dim=128):
        super(GMMPredictor, self).__init__()
        self._future_len = future_len
        self.gaussian = nn.Sequential(
            nn.Linear(dim, 256), 
            nn.GELU(), 
            nn.Linear(256, self._future_len*2)
        )
        self.score = nn.Sequential(
            nn.Linear(dim, 64), 
            nn.GELU(), 
            nn.Linear(64, 1),
        )
        self.scale = nn.Sequential(
            nn.Linear(dim, 256), 
            nn.GELU(), 
            nn.Linear(256, self._future_len*2)
        )
    
    def forward(self, input):
        B, M, _ = input.shape
        res = self.gaussian(input).view(B, M, self._future_len, 2) 
        scal = F.elu_(self.scale(input), alpha=1.0) + 1.0 + 0.0001
        scal = scal.view(B, M, self._future_len, 2) 
        score = self.score(input).squeeze(-1)

        return res, score, scal
    

class TimeDecoder(nn.Module):
    def __init__(self, future_len=60, dim=128):
        super(TimeDecoder, self).__init__()

        ###### State Consistency Module ######
        # state cross attention
        self.cross_block_time = nn.ModuleList(
            Cross_Block()
            for i in range(2)
        )

        # state bidirectional mamba
        self.timequery_embed_mamba = nn.ModuleList(  
            [
                create_block(  
                    d_model=dim,
                    layer_idx=i,
                    drop_path=0.2,  
                    bimamba=True,  
                    rms_norm=True,  
                )
                for i in range(2)
            ]
        )  
        self.timequery_norm_f = RMSNorm(dim, eps=1e-5)
        self.timequery_drop_path = DropPath(0.2)

        # MLP for state query
        self.dense_predict = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, dim),
            nn.GELU(),
            nn.Linear(dim, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

        ###### Mode Localization Module ######
        # mode self attention
        self.self_block_mode = nn.ModuleList(
            Block()
            for i in range(3)
        )

        # mode cross attention
        self.cross_block_mode = nn.ModuleList(
            Cross_Block()
            for i in range(3)
        )

        # mode query initialization
        self.multi_modal_query_embedding = nn.Embedding(6, dim)
        self.register_buffer('modal', torch.arange(6).long())

        # MLP for mode query
        self.predictor = GMMPredictor(future_len)

        ###### Hybrid Coupling Module ######
        # hybrid self attention
        self.self_block_dense = nn.ModuleList(
            Block()
            for i in range(3)
        )

        # hybrid cross attention
        self.cross_block_dense = nn.ModuleList(
            Cross_Block()
            for i in range(3)
        )

        # mode self attention for hybrid spatiotemporal queries
        self.self_block_different_mode = nn.ModuleList(
            Block()
            for i in range(3)
        )

        # state bidirectional mamba for hybrid spatiotemporal queries 
        self.dense_embed_mamba = nn.ModuleList(  
            [
                create_block(  
                    d_model=dim,
                    layer_idx=i,
                    drop_path=0.2,  
                    bimamba=True,  
                    rms_norm=True,  
                )
                for i in range(2)
            ]
        )
        self.dense_norm_f = RMSNorm(dim, eps=1e-5)
        self.dense_drop_path = DropPath(0.2)

        # MLP for final output
        self.predictor_dense = GMMPredictor_dense(future_len)

    def forward(self, mode, encoding, mask=None):
        # Dynamic state consistency
        for blk in self.cross_block_time:
            mode = blk(mode, encoding, key_padding_mask=mask)
        
        residual = None
        for blk_mamba in self.timequery_embed_mamba:
            mode, residual = blk_mamba(mode, residual)
        fused_add_norm_fn = rms_norm_fn if isinstance(self.timequery_norm_f, RMSNorm) else layer_norm_fn
        mode = fused_add_norm_fn(
            self.timequery_drop_path(mode),
            self.timequery_norm_f.weight,
            self.timequery_norm_f.bias,
            eps=self.timequery_norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=True  
        )
        
        dense_pred = self.dense_predict(mode)

        mode_tmp = mode
        
        # Directional intention localization
        multi_modal_query = self.multi_modal_query_embedding(self.modal)
        mode_query = encoding[:, 0]
        mode = mode_query[:, None] + multi_modal_query

        for blk in self.cross_block_mode:
            mode = blk(mode, encoding, key_padding_mask=mask)
        for blk in self.self_block_mode:
            mode = blk(mode)

        y_hat, pi, scal = self.predictor(mode)

        # Hybrid query coupling
        mode_dense = mode[:, :, None] + mode_tmp[:, None, :]
        B, M, T, C = mode_dense.shape
        
        mode_dense = mode_dense.reshape(B, -1, C)
        for blk in self.cross_block_dense:
            mode_dense = blk(mode_dense, encoding, key_padding_mask=mask)
        for blk in self.self_block_dense:
            mode_dense = blk(mode_dense)
        mode_dense = mode_dense.reshape(B, M, T, C)
        
        mode_dense = mode_dense.transpose(1, 2).reshape(-1, M, C)
        for blk in self.self_block_different_mode:
            mode_dense = blk(mode_dense)
        mode_dense = mode_dense.reshape(B, -1, M, C).transpose(1, 2)

        mode_dense = mode_dense.reshape(-1, T, C)
        residual = None
        for blk_mamba in self.dense_embed_mamba:
            mode_dense, residual = blk_mamba(mode_dense, residual)
        fused_add_norm_fn = rms_norm_fn if isinstance(self.dense_norm_f, RMSNorm) else layer_norm_fn
        mode_dense = fused_add_norm_fn(
            self.dense_drop_path(mode_dense),
            self.dense_norm_f.weight,
            self.dense_norm_f.bias,
            eps=self.dense_norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=True  
        )
        mode_dense = mode_dense.reshape(B, M, T, C)

        y_hat_new, pi_new, scal_new = self.predictor_dense(mode_dense)

        return dense_pred, y_hat, pi, mode, y_hat_new, pi_new, mode_dense, scal, scal_new
