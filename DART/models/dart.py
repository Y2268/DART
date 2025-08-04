import os, math
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, List
import pytorch_lightning as pl
from main_DART import instantiate_from_config
from alignment_hq import alignmodel
from ..modules.vqvae.utils import get_roi_regions
import pyiqa
class MaskedEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, mask_token_id=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.mask_token_id = mask_token_id or vocab_size
        
        # 扩展词汇表包含掩码token
        extended_vocab_size = vocab_size + 1
        self.word_embeddings = nn.Embedding(extended_vocab_size, embedding_dim)
        
        # 使用截断正态分布初始化掩码token嵌入
        with torch.no_grad():
            self.word_embeddings.weight[self.mask_token_id].normal_(mean=0.0, std=0.02)
    
    def forward(self, input_ids):
        # 确保索引在有效范围内
        valid_ids = torch.clamp(input_ids, 0, self.mask_token_id)
        return self.word_embeddings(valid_ids)
    

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def cosine_schedule(t):
    """
    Cosine schedule for mask ratio:
    - t: time parameter between 0 and 1
    - Returns mask ratio that starts high and decreases following cosine curve
    """

    return math.cos(math.pi/2 * t)
     
    
class TransformerSALayer(nn.Module): 
    # 实现自注意力机制和前馈神经网络mlp的Transformer编码器层
    def __init__(self, embed_dim, nhead=8, dim_mlp=2048, dropout=0.0, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout) # 多头注意力机制
        # Implementation of Feedforward model - MLP
        self.linear1 = nn.Linear(embed_dim, dim_mlp)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_mlp, embed_dim)
        # LayerNormapplied to the sums of the self attention and the input embedding
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        #激活函数
        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        
        # self attention
        #先对输入进行LayerNorm处理
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos) #将位置编码加到输入query和key上
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, 
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2) 

        # ffn 
        # 再次进行LayerNorm处理，然后通过前馈神经网络
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2) #残差连接
        return tgt


class DARTModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 ckpt_path_HQ=None, # HQ checkpoint path
                 ckpt_path_LQ=None, # LQ checkpoint path
                 ignore_keys=[],
                 image_key="lq",
                 colorize_nlabels=None,
                 monitor=None,
                 special_params_lr_scale=1.0,
                 comp_params_lr_scale=1.0,
                 schedule_step=[80000, 200000],
                 mask_token_id=1024,  # 新增：掩码标记ID
                 timesteps=8,        # 新增：推理时的迭代步数
                 mask_scheduling_method='cosine',  # 掩码调度方法
                 text_folder="/home/yanhy27/data/FFHQ_text2",  # 添加文本目录参数
                 use_alignment=True,                           # 添加是否使用对齐的标志
                 is_training=False                      
                 ):
        super().__init__()

        # import pdb
        # pdb.set_trace()
        self.mask_token_id = mask_token_id
        self.timesteps = timesteps
        self.num_iter = 8 # 训练时使用的迭代步数，可以小于推理时的步数
        self.mask_scheduling_method = mask_scheduling_method
        self.vocab_size = 1024  # 词汇表大小，与codebook_size相同
        print("timesteps = ", self.num_iter)    
        # 掩码调度函数
        self.mask_schedule = lambda t, method='cosine': self._get_mask_ratio(t, method)

        
        # 新增：掩码调度函数
        self.mask_schedule = lambda t: cosine_schedule(t)
        self.image_key = image_key
        self.vqvae = instantiate_from_config(ddconfig)

        lossconfig['params']['distill_param'] = ddconfig['params']
        # get the weights from HQ and LQ checkpoints
        if (ckpt_path_HQ is not None) and (ckpt_path_LQ is not None):
            print('loading HQ and LQ checkpoints')
            self.init_from_ckpt_two(
                ckpt_path_HQ, ckpt_path_LQ, ignore_keys=ignore_keys)

        if ('comp_weight' in lossconfig['params'] and lossconfig['params']['comp_weight']) or ('comp_style_weight' in lossconfig['params'] and lossconfig['params']['comp_style_weight']):
            self.use_facial_disc = True
        else:
            self.use_facial_disc = False

        self.fix_decoder = ddconfig['params']['fix_decoder']

        self.disc_start = lossconfig['params']['disc_start']
        self.special_params_lr_scale = special_params_lr_scale
        self.comp_params_lr_scale = comp_params_lr_scale
        self.schedule_step = schedule_step


        # self.cross_attention = MultiHeadAttnBlock(in_channels=256,head_size=8)

        # codeformer code-----------------------------------
        dim_embd=512
        n_head=8
        n_layers=9 # transformer层数
        codebook_size=1024
        latent_size=256
        concat_size=512  # 连接后的序列长度

        connect_list=['32', '64', '128', '256']
        fix_modules=['quantize','generator']
        self.connect_list = connect_list
        self.n_layers = n_layers
        self.dim_embd = dim_embd
        self.dim_mlp = dim_embd*2
        self.feat_emb = nn.Linear(256, dim_embd)
        self.position_emb = nn.Parameter(torch.zeros(concat_size, dim_embd))

        # 使用专用掩码嵌入层替代普通嵌入层
        self.token_emb = MaskedEmbedding(
            vocab_size=self.vocab_size,
            embedding_dim=self.dim_embd,
            mask_token_id=self.mask_token_id
        )
        # transformer ft_layers由n_layers个TransformerSALayer组成
        self.ft_layers = nn.Sequential(*[TransformerSALayer(embed_dim=dim_embd, nhead=n_head, dim_mlp=self.dim_mlp, dropout=0.0) 
                                    for _ in range(self.n_layers)])

        # logits_predict head 用于预测索引的线性层
        self.idx_pred_layer = nn.Sequential(
            nn.LayerNorm(dim_embd),
            nn.Linear(dim_embd, codebook_size, bias=False))
        self.use_alignment = use_alignment
        self.text_folder = text_folder
        
        # 如果使用对齐，初始化对齐模型
        self.alignmodel = alignmodel(
                dim=256,
                text_folder=text_folder,
                is_training=is_training
            )

        self.musiq_model = pyiqa.create_metric('musiq')
        # self.show_model_summary()
    def show_model_summary(self, input_size=(1, 3, 512, 512)):
        """使用torchinfo显示详细的模型结构并计算总参数"""
        try:
            from torchinfo import summary
            
            print("\n" + "="*80)
            print("DART MODEL DETAILED SUMMARY")  
            print("="*80)
            
            # 显示完整模型结构
            model_summary = summary(self, input_size=input_size, verbose=2, 
                col_names=["input_size", "output_size", "num_params", "trainable"])
            
            print("\n" + "="*80)
            print("DETAILED COMPONENT ANALYSIS")
            print("="*80)
            
            # 单独显示VQ-VAE组件
            print("\n=== VQ-VAE Encoder ===")
            encoder_summary = summary(self.vqvae.encoder, input_size=input_size, verbose=1)
            
            print("\n=== VQ-VAE HQ Encoder ===") 
            hq_encoder_summary = summary(self.vqvae.HQ_encoder, input_size=input_size, verbose=1)
            
            print("\n=== VQ-VAE Decoder ===") 
            # 假设编码后的特征尺寸为 (1, 256, 16, 16)
            decoder_input_size = (1, 256, 16, 16)
            decoder_summary = summary(self.vqvae.decoder, input_size=decoder_input_size, verbose=1)
            
            print("\n=== Transformer Layers ===")
            # Transformer输入尺寸 (seq_len, batch_size, embed_dim)
            transformer_input_size = (512, 1, self.dim_embd)  # 256*2=512 (LQ+HQ tokens)
            transformer_summary = summary(self.ft_layers, input_size=transformer_input_size, verbose=1)
            
            # 手动计算各组件参数
            self._print_detailed_parameter_statistics()
            
        except ImportError:
            print("请安装torchinfo: pip install torchinfo")
            print("使用备用方法显示模型结构...")
            self._print_model_structure_fallback()

    def _print_detailed_parameter_statistics(self):
        """打印详细的参数统计信息"""
        print("\n" + "="*80)
        print("DETAILED PARAMETER STATISTICS")
        print("="*80)
        
        # 定义各个组件
        components = {
            "VQ-VAE LQ Encoder": self.vqvae.encoder,
            "VQ-VAE HQ Encoder": self.vqvae.HQ_encoder,
            "VQ-VAE Decoder": self.vqvae.decoder,
            "VQ-VAE LQ Quantizer": getattr(self.vqvae, 'quantize', None),
            "VQ-VAE HQ Quantizer": self.vqvae.HQ_quantize,
            "VQ-VAE Quant Conv": self.vqvae.quant_conv,
            "VQ-VAE HQ Quant Conv": getattr(self.vqvae, 'HQ_quant_conv', None),
            "VQ-VAE Post Quant Conv": self.vqvae.post_quant_conv,
            "Feature Embedding": self.feat_emb,
            "Token Embedding": self.token_emb,
            "Transformer Layers": self.ft_layers,
            "Index Prediction": self.idx_pred_layer,
        }
        
        # 如果使用对齐模型，添加到组件中
        if self.use_alignment and hasattr(self, 'alignmodel') and self.alignmodel is not None:
            components["Alignment Model"] = self.alignmodel
        
        total_params = 0
        total_trainable = 0
        
        print(f"{'Component':<25} {'Total Params':<15} {'Trainable':<15} {'Size (MB)':<12}")
        print("-" * 80)
        
        for name, component in components.items():
            if component is not None:
                # 计算参数数量
                component_params = sum(p.numel() for p in component.parameters())
                component_trainable = sum(p.numel() for p in component.parameters() if p.requires_grad)
                
                # 计算内存大小 (假设float32，每个参数4字节)
                size_mb = component_params * 4 / 1024 / 1024
                
                total_params += component_params
                total_trainable += component_trainable
                
                print(f"{name:<25} {component_params:<15,} {component_trainable:<15,} {size_mb:<12.2f}")
            else:
                print(f"{name:<25} {'N/A':<15} {'N/A':<15} {'N/A':<12}")
        
        # 添加位置嵌入参数
        if hasattr(self, 'position_emb'):
            pos_emb_params = self.position_emb.numel()
            pos_emb_size = pos_emb_params * 4 / 1024 / 1024
            total_params += pos_emb_params
            total_trainable += pos_emb_params
            print(f"{'Position Embedding':<25} {pos_emb_params:<15,} {pos_emb_params:<15,} {pos_emb_size:<12.2f}")
        
        print("-" * 80)
        print(f"{'TOTAL MODEL':<25} {total_params:<15,} {total_trainable:<15,} {total_params * 4 / 1024 / 1024:<12.2f}")
        
        # 额外统计信息
        print(f"\n{'Additional Statistics:':<30}")
        print("-" * 50)
        print(f"{'Non-trainable parameters:':<30} {total_params - total_trainable:,}")
        print(f"{'Trainable ratio:':<30} {total_trainable/total_params*100:.2f}%")
        print(f"{'Model size (float32):':<30} {total_params * 4 / 1024 / 1024:.2f} MB")
        print(f"{'Model size (float16):':<30} {total_params * 2 / 1024 / 1024:.2f} MB")
        
        # VQ-VAE特定统计
        print(f"\n{'VQ-VAE Specific Statistics:':<30}")
        print("-" * 50)
        if hasattr(self.vqvae, 'HQ_quantize') and self.vqvae.HQ_quantize is not None:
            hq_codebook_size = self.vqvae.HQ_quantize.embedding.weight.shape
            print(f"{'HQ Codebook shape:':<30} {hq_codebook_size}")
            print(f"{'HQ Codebook params:':<30} {self.vqvae.HQ_quantize.embedding.weight.numel():,}")
        
        # Transformer特定统计
        print(f"\n{'Transformer Specific Statistics:':<30}")
        print("-" * 50)
        print(f"{'Embedding dimension:':<30} {self.dim_embd}")
        print(f"{'Number of layers:':<30} {self.n_layers}")
        print(f"{'Vocab size:':<30} {self.vocab_size}")
        print(f"{'Position embedding shape:':<30} {self.position_emb.shape}")
        
        return {
            'total_params': total_params,
            'total_trainable': total_trainable,
            'model_size_mb': total_params * 4 / 1024 / 1024
        }
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def init_from_ckpt_two(self, path_HQ, path_LQ, ignore_keys=list()):
        """使用直接复制方式从HQ和LQ检查点加载权重到各个组件"""
        print('='*50)
        print('加载HQ和LQ检查点...')
        
        try:
            # 加载HQ检查点
            print(f"加载HQ检查点: {path_HQ}")
            sd_HQ = torch.load(path_HQ, map_location="cpu")
            if "state_dict" in sd_HQ:
                sd_HQ = sd_HQ["state_dict"]
            
            # 加载LQ检查点
            print(f"加载LQ检查点: {path_LQ}")
            sd_LQ = torch.load(path_LQ, map_location="cpu")
            if "state_dict" in sd_LQ:
                sd_LQ = sd_LQ["state_dict"]
            
            # 打印原始键数量
            print(f"HQ检查点键数量: {len(sd_HQ)}")
            print(f"LQ检查点键数量: {len(sd_LQ)}")
            
            # ----- 第1部分：定义组件映射 -----
            # 键格式: (检查点中的前缀, 模型中的组件, 检查点对象)
            component_mapping = [
                # HQ组件 - 从HQ检查点加载
                ('vqvae.quantize', self.vqvae.HQ_quantize, sd_HQ),
                ('vqvae.encoder', self.vqvae.HQ_encoder, sd_HQ),
                ('vqvae.quant_conv', self.vqvae.HQ_quant_conv, sd_HQ),
                ('vqvae.decoder', self.vqvae.decoder, sd_HQ),
                ('vqvae.post_quant_conv', self.vqvae.post_quant_conv, sd_HQ),
                
                # LQ组件 - 从LQ检查点加载
                ('vqvae.encoder', self.vqvae.encoder, sd_LQ),
                ('vqvae.quant_conv', self.vqvae.quant_conv, sd_LQ),
            ]
            
            # ----- 第2部分：加载各组件关键权重 -----
            # 直接加载embedding - 这是最关键的部分
            hq_embed_key = 'vqvae.quantize.embedding.weight'
            if hq_embed_key in sd_HQ:
                print(f"\n直接加载HQ codebook embedding权重:")
                print(f"  检查点中embedding: min={sd_HQ[hq_embed_key].min().item():.4f}, max={sd_HQ[hq_embed_key].max().item():.4f}")
                self.vqvae.HQ_quantize.embedding.weight.data.copy_(sd_HQ[hq_embed_key])
                print(f"  加载后embedding: min={self.vqvae.HQ_quantize.embedding.weight.min().item():.4f}, max={self.vqvae.HQ_quantize.embedding.weight.max().item():.4f}")
            else:
                print(f"警告: 未找到HQ embedding权重键 '{hq_embed_key}'")
            
            # ----- 第3部分：逐组件加载其余权重 -----
            loaded_params = 0
            
            for prefix, component, checkpoint in component_mapping:
                component_loaded = 0
                
                # 对于每个组件遍历其所有命名参数
                for name, param in component.named_parameters():
                    # 构建检查点中的对应键名
                    ckpt_key = f"{prefix}.{name}"
                    
                    # 检查键是否存在于检查点中
                    if ckpt_key in checkpoint:
                        # 直接复制权重
                        param.data.copy_(checkpoint[ckpt_key])
                        component_loaded += 1
                        loaded_params += 1
                
                # 打印每个组件加载的参数数量
                total_params = sum(1 for _ in component.parameters())
                print(f"组件 {prefix}: 加载参数 {component_loaded}/{total_params}")
            
            # ----- 第4部分: 验证权重加载 -----

            
            # ----- 第5部分: 打印最终HQ量化器统计信息 -----
            print('加载检查点完成')
            print('='*50)
        except Exception as e:
            print(f"加载检查点时出错: {e}")
            import traceback
            traceback.print_exc()
            raise

    # 3. 添加掩码调度函数
    def _get_mask_ratio(self, t, method='cosine'):
        """获取不同时间步的掩码比例"""
        if method == 'cosine':
            return math.cos(math.pi/2 * t)
        elif method == 'linear':
            return 1.0 - t
        else:
            return math.cos(math.pi/2 * t)  # 默认使用余弦调度  

        # ============ 掩码自回归部分 ============
    def masked_token_generation(self, lq_feat, gt_indices=None, is_training=True):
        """
        使用迭代式揭示实现渐进式生成HQ令牌
        
        Args:
            lq_feat: 低质量特征 [B, C, H, W]
            gt_indices: 目标HQ索引 (仅训练时需要) [B, seq_len]
            is_training: 是否处于训练模式
            
        Returns:
            dict: 包含生成结果和相关指标的字典
        """
        # 启用内存优化
        torch.cuda.empty_cache()
        
        # 获取设备和基本形状
        device = lq_feat.device
        batch_size = lq_feat.shape[0]
        seq_len = 256  # 16x16
        
        # 特征嵌入
        pos_emb = self.position_emb.unsqueeze(1).repeat(1, batch_size, 1) 
        feat_emb = self.feat_emb(lq_feat.flatten(2).permute(2, 0, 1))  # [seq_len, batch, embed_dim]
        
        # 记录各类统计信息的容器
        losses = []              # BCE损失
        batch_zero_ratio = 0.0   # 当前批次的零预测率
        # ===== 1. 初始化HQ令牌 =====
        target_indices = gt_indices
        if is_training and gt_indices is not None:
            # 修改掩码策略：使用更高的掩码率，更好地模拟推理过程
            r = torch.rand(1, device=device).item()
            mask_ratio = self._get_mask_ratio(r, self.mask_scheduling_method)
            
            # 精确选择掩码数量，而非随机概率
            num_masks = int(mask_ratio * seq_len)
            mask_indices = torch.randperm(seq_len, device=device)[:num_masks]
            mask = torch.zeros_like(target_indices, dtype=torch.bool, device=device)
            mask.scatter_(1, mask_indices.unsqueeze(0).expand(batch_size, -1), True)
            
            current_hq_tokens = target_indices.clone()
            current_hq_tokens[mask] = self.mask_token_id
            del mask
        else:
            # 推理模式保持不变
            mask_ratio = 1.0
            print(f"推理模式: 掩码比例 = {mask_ratio:.2f}")
            current_hq_tokens = torch.full((batch_size, seq_len), self.mask_token_id, device=device, dtype=torch.long)
        
            # ===== 2. 设置迭代参数 =====
        num_steps = self.num_iter if is_training else self.timesteps
            
        # ===== 3. 主要迭代循环 =====
        for step in range(num_steps):
            # A. 检查当前掩码状态
            current_mask = (current_hq_tokens == self.mask_token_id)
            masked_per_batch = current_mask.sum(dim=1)
            total_masked = masked_per_batch.sum().item()
        
            # 如果已经没有掩码，提前结束
            if total_masked == 0:
                break
            
            # B. Transformer前向传播和预测
            with torch.set_grad_enabled(is_training):
                try:
                    # 获取令牌嵌入
                    token_emb = self.token_emb(current_hq_tokens).permute(1, 0, 2)  # [B,seq_len,dim] -> [seq_len,B,dim]

                    # 将LQ特征作为前缀连接到HQ令牌
                    input_emb = torch.cat([feat_emb, token_emb], dim=0)  # [2*seq_len, batch, embed_dim]

                    # 创建注意力掩码 - 双向注意力但标记掩码位置
                    attn_mask = None  # 默认使用双向注意力
                    
                    # Transformer前向传播
                    query_emb = input_emb
                    for layer in self.ft_layers:
                        query_emb = layer(query_emb, 
                                        tgt_mask=attn_mask,
                                        query_pos=pos_emb)  # 直接使用完整的位置编码
                    
                    # 只取HQ令牌部分的输出进行预测
                    hq_output = query_emb[seq_len:]  # 只取后半部分 [seq_len, batch, embed_dim]
                    
                    # 输出logits
                    logits = self.idx_pred_layer(hq_output)  # [seq_len, B, vocab_size]
                    logits = logits.permute(1, 0, 2)  # [B, seq_len, vocab_size]
                    
                    # 计算概率和预测
                    probs = F.softmax(logits, dim=-1)
                    confidence, predictions = torch.max(probs, dim=-1)

                    # MaskGIT关键修改：将未掩码位置的置信度设为1.0
                    unmasked_positions = (current_hq_tokens != self.mask_token_id)
                    confidence = confidence.clone()
                    confidence[unmasked_positions] = 1.0
                    
                    # 零预测统计 - 只在掩码位置
                    masked_positions = (current_hq_tokens == self.mask_token_id)
                    if masked_positions.sum() > 0:
                        masked_predictions = predictions[masked_positions]
                        zero_count = (masked_predictions == 0).sum().item()
                        total_count = len(masked_predictions)
                        step_zero_ratio = zero_count / max(total_count, 1)
                        batch_zero_ratio = step_zero_ratio
                   
                    # ===== C. 计算每个批次项的揭示数量和进行揭示 =====
                    is_final_step = (step == num_steps - 1)
                    
                    if is_final_step:
                        # 最后一步: 打印统计信息并揭示所有掩码位置
                        print(f"\n最终预测统计 (步骤 {step+1}/{num_steps}):")
                        mask_pos = (current_hq_tokens[0] == self.mask_token_id)
                        if mask_pos.any():
                            mask_pos_probs = probs[0][mask_pos]
                            top_indices = torch.argmax(mask_pos_probs, dim=-1)
                            unique_indices, counts = torch.unique(top_indices, return_counts=True)
                            top_counts = {idx.item(): count.item() for idx, count in zip(unique_indices, counts)}
                            print(f"  被掩码位置的top1分布: {top_counts}")
                        
                        # 揭示所有掩码位置
                        new_tokens = current_hq_tokens.clone()
                        mask_positions = (current_hq_tokens == self.mask_token_id)
                        if mask_positions.any():
                            # 在掩码位置应用预测值
                            new_tokens[mask_positions] = predictions[mask_positions]
                        
                        current_hq_tokens = new_tokens
                        
                        # 在计算损失部分修改
                        if is_training and target_indices is not None:
                            # 修改：计算所有当前掩码位置的损失，而非新揭示的
                            mask_positions = (current_hq_tokens == self.mask_token_id)
                            
                            if mask_positions.sum() > 0:
                                step_loss = F.cross_entropy(
                                    logits.reshape(-1, logits.size(-1)),
                                    target_indices.view(-1),
                                    reduction='none',
                                    label_smoothing=0.1
                                ).view_as(target_indices)
                                
                                # 应用于所有当前掩码位置
                                masked_loss = step_loss * mask_positions.float()
                                avg_loss = masked_loss.sum() / (mask_positions.sum() + 1e-10)
                                losses.append(avg_loss)
                    else:
                        # 非最后一步: 按照掩码调度逐批次处理
                        # 计算此步应保留的token比例和对应的置信度阈值
                        progress_ratio = (step + 1) / num_steps
                        target_keep_ratio = 1.0 - self._get_mask_ratio(progress_ratio, self.mask_scheduling_method)

                        # 对所有位置应用Gumbel噪声
                        temperature = 2 * (1.0 - step/num_steps)**2  # 温度参数
                        u = torch.rand_like(confidence)
                        gumbel_noise = -torch.log(-torch.log(u + 1e-7) + 1e-7)
                        noisy_confidence = torch.log(confidence + 1e-9) + temperature * gumbel_noise

                        # 全局处理置信度
                        batch_tokens = []
                        for i in range(batch_size):
                            # 获取当前掩码状态
                            current_mask = (current_hq_tokens[i] == self.mask_token_id)
                            unmasked_count = (~current_mask).sum().item()
                            
                            # 计算根据进度应该揭示的总token数量
                            target_revealed = min(int(seq_len * target_keep_ratio), seq_len)
                            
                            # 计算还需要揭示多少token
                            tokens_to_reveal = max(0, target_revealed - unmasked_count)
                            
                            # 创建新token状态，首先复制所有已揭示的token
                            new_tokens = torch.full_like(current_hq_tokens[i], self.mask_token_id)
                            new_tokens[~current_mask] = current_hq_tokens[i][~current_mask]  # 保留所有已揭示位置
                            
                            # 如果需要揭示更多token且还有掩码位置
                            if tokens_to_reveal > 0 and current_mask.any():
                                # 只考虑掩码位置的置信度
                                mask_positions = torch.where(current_mask)[0]
                                masked_confidence = confidence[i][mask_positions]
                                
                                # 添加Gumbel噪声到掩码位置
                                u = torch.rand_like(masked_confidence)
                                gumbel_noise = -torch.log(-torch.log(u + 1e-7) + 1e-7)
                                noisy_masked_confidence = torch.log(masked_confidence + 1e-9) + temperature * gumbel_noise
                                
                                # 选择置信度最高的k个位置进行揭示
                                _, top_indices = torch.topk(noisy_masked_confidence, k=min(tokens_to_reveal, len(mask_positions)))
                                positions_to_reveal = mask_positions[top_indices]
                                
                                # 揭示选中的位置
                                new_tokens[positions_to_reveal] = predictions[i][positions_to_reveal]
                                
                            # 处理零预测问题
                            zero_mask = (new_tokens != self.mask_token_id) & (new_tokens == 0)
                            if zero_mask.any():
                                print(f"  警告: 批次项{i}有{zero_mask.sum().item()}个零预测")
                                # 随机替换零预测

                            batch_tokens.append(new_tokens)

                        # 更新当前tokens
                        current_hq_tokens = torch.stack(batch_tokens)
                        # 计算当前步骤的掩码损失
                        if is_training and target_indices is not None:
                            # 找出当前所有掩码位置
                            current_mask = (current_hq_tokens == self.mask_token_id)
                            
                            if current_mask.sum() > 0:
                                # 计算交叉熵损失
                                step_loss = F.cross_entropy(
                                    logits.reshape(-1, logits.size(-1)),
                                    target_indices.view(-1),
                                    reduction='none',
                                    label_smoothing=0.1
                                ).view_as(target_indices)
                                
                                # 只计算掩码位置的损失
                                masked_loss = step_loss * current_mask.float()
                                avg_loss = masked_loss.sum() / (current_mask.sum() + 1e-10)
                                
                                # 添加到损失列表
                                losses.append(avg_loss)
                        # 清理内存
                        torch.cuda.empty_cache()
                
                except RuntimeError as e:
                    print(f"警告: 运行时错误: {e}")
                    # 紧急恢复策略
                    new_tokens = current_hq_tokens.clone()
                    mask_positions = (current_hq_tokens == self.mask_token_id)
                    if mask_positions.any():
                        # 随机填充掩码位置
                        new_tokens[mask_positions] = torch.randint(
                            0, self.vocab_size-1, (mask_positions.sum(),), device=device
                        )
                    current_hq_tokens = new_tokens
                    # 尝试继续循环而不是崩溃
        
        # ===== 4. 最终安全检查 =====
        # 确保没有掩码标记
        mask_positions = (current_hq_tokens == self.mask_token_id)
        if mask_positions.any():
            print(f"警告: 仍有{mask_positions.sum().item()}个未揭示的掩码位置，进行随机填充")
            current_hq_tokens[mask_positions] = torch.randint(
                0, self.vocab_size-1, (mask_positions.sum().item(),), device=device
            )
        
        # 确保索引范围有效
        current_hq_tokens = torch.clamp(current_hq_tokens, 0, self.vocab_size-1)
        
        # ===== 5. 结果组装 =====
        result = {
            'mask_ratio': mask_ratio,
            'final_tokens': current_hq_tokens,
            'zero_pred_ratio': batch_zero_ratio,
        }
        
        if is_training and losses:
            result['bce_loss'] = torch.stack(losses).mean()
        else:
            result['bce_loss'] = torch.tensor(0.0, device=device)
        # print(f"当前掩码率: {mask_ratio:.4f}, 当前零预测率: {batch_zero_ratio:.4f}")
        
        # 最终内存清理
        torch.cuda.empty_cache()
        
        return result
        
    # 5. 修改forward方法集成新函数
    def forward(self, input, gt=None,filenames=None, save_features=False, features_dir="features"):

        if gt is not None:   #检查是否提供gt
            quant_gt, gt_indices, gt_info, gt_hs, gt_h, gt_dictionary = self.encode_to_gt(gt) #将gt编码
        # LQ feature from LQ encoder and quantizer  
        z_hs = self.vqvae.encoder(input)
        z_h = self.vqvae.quant_conv(z_hs['out'])
        print(f"z_h shape: {z_h.shape}")    
        
        # origin HQ codebook for index 用原始HQcodebook进行索引
        quant_z, emb_loss, z_info, z_dictionary = self.vqvae.HQ_quantize(z_h)
        indices = z_info[2].view(quant_z.shape[0], -1)
        z_indices = indices

        if gt is None:
            quant_gt = quant_z
            gt_indices = z_indices
            self.alignmodel.eval()
            self.alignmodel.set_eval_mode(True)

        print(f"量化图像值范围: [{quant_gt.min().item()}, {quant_gt.max().item()}]")
        # 添加特征对齐处理 =============================================
        # 使用特征对齐
        batch_size, c, h, w = z_h.shape
        z_seq = z_h.permute(0, 2, 3, 1).reshape(batch_size, h*w, c)
        if self.alignmodel is not None and filenames is not None:
            # 类型检查 - 避免处理Tensor类型的文件路径
            if isinstance(filenames, torch.Tensor):
                print("检测到Tensor类型的文件路径，跳过特征对齐")
                lq_feat = z_seq  # 直接使用z_seq，不需要重复计算
            else:
                # 正常处理字符串路径
                batch_with_gt_path = {"gt_path": filenames}
                text_features = self.alignmodel.get_text_features(z_h, batch=batch_with_gt_path)

                # 检测并匹配数据类型
                model_dtype = next(self.alignmodel.parameters()).dtype
                z_seq = z_seq.to(dtype=model_dtype)
                text_features = text_features.to(dtype=model_dtype)
                
                lq_feat = self.alignmodel(z_seq, text_features)

                # 转回原始数据类型以保持一致性
                lq_feat = lq_feat.to(dtype=z_h.dtype)
                print(f"lq_feat shape: {lq_feat.shape}")

        else:
            # 如果没有对齐模型，直接使用原始特征
            lq_feat = z_seq  # 直接使用z_seq，不需要重复计算
            print("未使用对齐模型，直接使用原始特征")

        # ============ 使用迭代式掩码自回归替代原方法 ============
        # 调用新实现的迭代式掩码生成函数
        generation_result = self.masked_token_generation(
            lq_feat=lq_feat, 
            gt_indices=gt_indices, 
            is_training=(gt is not None)  # 训练模式取决于是否提供了gt
        )
        
        # 获取最终的token和BCE损失
        final_tokens = generation_result['final_tokens']
        BCE_loss = generation_result['bce_loss']
        
        # 获取特征并解码
        quant_feat = self.vqvae.HQ_quantize.get_codebook_entry(
            final_tokens.reshape(-1), 
            shape=[z_h.shape[0], 16, 16, 256]
        )

        # 将quant_feat转换为与lq_feat相同的形状
        batch_size, h, w, c = z_h.shape[0], 16, 16, 256
        quant_feat_reshaped = quant_feat.permute(0, 2, 3, 1).reshape(batch_size, h*w, c)

        # 将quant_gt转换为与lq_feat相同的形状以计算L2损失
        batch_size_gt, c_gt, h_gt, w_gt = quant_gt.shape
        quant_gt_reshaped = quant_gt.permute(0, 2, 3, 1).reshape(batch_size_gt, h_gt*w_gt, c_gt)

        # 保留梯度 - 使用形状匹配的张量
        quant_feat_reshaped = z_seq + (quant_feat_reshaped - z_seq).detach()

        # 计算 L2 损失 - lq_feat 和 quant_gt_reshaped 现在有相同的形状
        L2_loss = F.mse_loss(z_seq, quant_gt_reshaped)
        # 将形状转回用于解码器
        quant_feat = quant_feat_reshaped.reshape(batch_size, h, w, c).permute(0, 3, 1, 2)
        dec = self.vqvae.decode(quant_feat)
        # if save_features:
        #     self.save_features(z_seq, quant_feat_reshaped, z_h_HQ, filenames, features_dir)    
        return dec, BCE_loss, L2_loss, z_info, z_hs, z_h, quant_gt, z_dictionary
    
    @torch.no_grad()
    def encode_to_gt(self, gt):
        quant_gt, _, info, hs, h, dictionary = self.vqvae.HQ_encode(gt)
        indices = info[2].view(quant_gt.shape[0], -1)
        return quant_gt, indices, info, hs, h, dictionary

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        self.alignmodel.train()
        self.alignmodel.set_eval_mode(False)
        if optimizer_idx == None:
            optimizer_idx = 0

        x = batch[self.image_key]
        gt = batch['gt']
        filenames = batch.get('gt_path', None)
        xrec, BCE_loss, L2_loss, info, hs,_,_,_ = self(x, gt, filenames)

        qloss = BCE_loss + 10*L2_loss

        if self.image_key != 'gt':
            x = batch['gt']

        if self.use_facial_disc:
            loc_left_eyes = batch['loc_left_eye']
            loc_right_eyes = batch['loc_right_eye']
            loc_mouths = batch['loc_mouth']
            face_ratio = xrec.shape[-1] / 512
            components = get_roi_regions(
                x, xrec, loc_left_eyes, loc_right_eyes, loc_mouths, face_ratio)
        else:
            components = None

        if optimizer_idx == 0:
            
            aeloss = BCE_loss + 10*L2_loss
            
            rec_loss = (torch.abs(gt.contiguous() - xrec.contiguous()))

            log_dict_ae = {
                   "train/BCE_loss": BCE_loss.detach().mean(),
                   "train/L2_loss": L2_loss.detach().mean(),
                   "train/Rec_loss": rec_loss.detach().mean()
                }
            
            bce_loss = log_dict_ae["train/BCE_loss"]
            self.log("BCE_loss", bce_loss, prog_bar=True,
                     logger=True, on_step=True, on_epoch=True)
            
            l2_loss = log_dict_ae["train/L2_loss"]
            self.log("L2_loss", l2_loss, prog_bar=True,
                     logger=True, on_step=True, on_epoch=True)
            
            Rec_loss = log_dict_ae["train/Rec_loss"]
            self.log("Rec_loss", Rec_loss, prog_bar=True,
                     logger=True, on_step=True, on_epoch=True)

            self.log_dict(log_dict_ae, prog_bar=False,
                          logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, components, optimizer_idx, self.global_step,
                                                last_layer=None, split="train")
            self.log("train/discloss", discloss, prog_bar=True,
                     logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False,
                          logger=True, on_step=True, on_epoch=True)
            return discloss

        if self.disc_start <= self.global_step:

            # left eye
            if optimizer_idx == 2:
                # discriminator
                disc_left_loss, log_dict_disc = self.loss(qloss, x, xrec, components, optimizer_idx, self.global_step,
                                                          last_layer=None, split="train")
                self.log("train/disc_left_loss", disc_left_loss,
                         prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_disc, prog_bar=False,
                              logger=True, on_step=True, on_epoch=True)
                return disc_left_loss

            # right eye
            if optimizer_idx == 3:
                # discriminator
                disc_right_loss, log_dict_disc = self.loss(qloss, x, xrec, components, optimizer_idx, self.global_step,
                                                           last_layer=None, split="train")
                self.log("train/disc_right_loss", disc_right_loss,
                         prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_disc, prog_bar=False,
                              logger=True, on_step=True, on_epoch=True)
                return disc_right_loss

            # mouth
            if optimizer_idx == 4:
                # discriminator
                disc_mouth_loss, log_dict_disc = self.loss(qloss, x, xrec, components, optimizer_idx, self.global_step,
                                                           last_layer=None, split="train")
                self.log("train/disc_mouth_loss", disc_mouth_loss,
                         prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_disc, prog_bar=False,
                              logger=True, on_step=True, on_epoch=True)
                return disc_mouth_loss

    def validation_step(self, batch, batch_idx):
        self.alignmodel.eval()
        self.alignmodel.set_eval_mode(True)
        x = batch[self.image_key]  # 获取低质量图像
        self.alignmodel.eval()
        # 检查是否有GT图像
        has_gt = 'gt' in batch
        
        if has_gt:
            gt = batch['gt']
            filenames = batch.get('gt_path', batch.get('filename', None))
            xrec, BCE_loss, L2_loss, info, hs, _, _, _ = self(x, gt, filenames)
            
            qloss = BCE_loss + L2_loss
            rec_loss = torch.abs(gt.contiguous() - xrec.contiguous()).mean()
        else:
            # 无GT模式 - 仅处理LQ图像
            filenames = batch.get('filename', None)
            # 将LQ图像同时作为输入和目标 (自重建模式)
            xrec, BCE_loss, L2_loss, info, hs, _, _, _ = self(x, x, filenames)
            
            rec_loss = torch.tensor(0.0, device=self.device)  # 无GT时不计算重建损失
    
        # 记录所有指标
        log_dict_ae = {
            "val/BCE_loss": BCE_loss.detach().mean(),
            "val/L2_loss": L2_loss.detach().mean(),
            "val/Rec_loss": rec_loss.detach().mean() if has_gt else rec_loss,
        }
        self.log_dict(log_dict_ae)

        return self.log_dict
    def configure_optimizers(self):
        lr = self.learning_rate

        normal_params = []
        special_params = []
        fixed_params = []
        fixed_parameter = 0
        test_count = 0
        # autoencoder part -------------------------------
        for name, param in self.vqvae.named_parameters():
            if not param.requires_grad:
                continue

            if 'HQ' in name:
                special_params.append(param)
                fixed_parameter = fixed_parameter + 1
                continue
            if 'decoder' in name or 'post_quant_conv' in name or 'quantize' in name:
                test_count = test_count + 1
                # continue
                special_params.append(param)
                # print(name)
            else:
                normal_params.append(param)

        # 添加alignmodel参数 - 这是新增的部分
        if self.alignmodel is not None:
            print("将alignmodel参数添加到优化器中...")
            for name, param in self.alignmodel.named_parameters():
                if not param.requires_grad:
                    continue
                else:
                    normal_params.append(param)
                    print(f"添加alignmodel参数: {name}")
                
        # transformer part--------------------------------
        
        normal_params.append(self.position_emb)   
        
        for name, param in self.feat_emb.named_parameters():
            if not param.requires_grad:
                continue
            else:
                normal_params.append(param) 

        for name, param in self.ft_layers.named_parameters():
            if not param.requires_grad:
                continue
            else:
                normal_params.append(param) 
        for name, param in self.token_emb.named_parameters():
            if not param.requires_grad:
                continue
            else:
                normal_params.append(param)

        for name, param in self.idx_pred_layer.named_parameters():
            if not param.requires_grad:
                continue
            else:
                normal_params.append(param)                 
        
        # print('special_params', special_params)
        opt_ae_params = [{'params': normal_params, 'lr': lr}]

        opt_ae = torch.optim.Adam(opt_ae_params, betas=(0.5, 0.9))

        optimizations = opt_ae

        if self.use_facial_disc:
            opt_l = torch.optim.Adam(self.loss.net_d_left_eye.parameters(),
                                     lr=lr*self.comp_params_lr_scale, betas=(0.9, 0.99))
            opt_r = torch.optim.Adam(self.loss.net_d_right_eye.parameters(),
                                     lr=lr*self.comp_params_lr_scale, betas=(0.9, 0.99))
            opt_m = torch.optim.Adam(self.loss.net_d_mouth.parameters(),
                                     lr=lr*self.comp_params_lr_scale, betas=(0.9, 0.99))
            optimizations += [opt_l, opt_r, opt_m]

            s2 = torch.optim.lr_scheduler.MultiStepLR(
                opt_l, milestones=self.schedule_step, gamma=0.1, verbose=True)
            s3 = torch.optim.lr_scheduler.MultiStepLR(
                opt_r, milestones=self.schedule_step, gamma=0.1, verbose=True)
            s4 = torch.optim.lr_scheduler.MultiStepLR(
                opt_m, milestones=self.schedule_step, gamma=0.1, verbose=True)
            schedules += [s2, s3, s4]

        # return optimizations, schedules
        return optimizations

    def get_last_layer(self):
        if self.fix_decoder:
            return self.vqvae.quant_conv.weight
        return self.vqvae.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = batch[self.image_key]
        x = x.to(self.device)
        # 检查是否有GT图像
        has_gt = 'gt' in batch
        
        if has_gt:
            gt = batch['gt'].to(self.device)
            filenames = batch.get('gt_path', batch.get('filename', None))
        else:
            # 无GT模式 - 使用LQ图像作为输入和目标
            gt = None  # 使用输入图像作为GT，而不是None
            filenames = batch.get('filename', None)
            print(f"日志记录: 检测到无GT批次，使用LQ图像作为GT")
        
        # 使用处理后的文件名调用forward方法
        xrec, _, _, _, _, _, _, _ = self(x, gt, filenames=filenames)
        
        log["inputs"] = x
        log["reconstructions"] = xrec
        
        if has_gt:
            log["gt"] = batch['gt'].to(self.device)
        else:
            log["gt"] = x  # 在无GT情况下也提供GT项，以保持一致性
        
        return log