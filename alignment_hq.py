import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import traceback
from pathlib import Path
import gc

class FeatureMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, output_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        output_dim = output_dim or input_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim*2),
            nn.GELU(),  # 使用GELU激活函数，也可以选择ReLU、LeakyReLU等
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*2, hidden_dim*2),  # 新增的隐藏层
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*2, output_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """特征预处理"""
        return self.mlp(x)
    
class SFTModule(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.scale_conv = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1)
        )
        
        self.shift_conv = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1)
        )
    
    def forward(self, features, conditions):
        """特征变换，基于条件进行缩放和平移"""
        # 将条件特征池化为单个向量
        pooled_conditions = conditions.mean(dim=1, keepdim=True)
        
        # 转换维度以适应卷积操作
        features_t = features.transpose(1, 2)
        conditions_t = pooled_conditions.transpose(1, 2)
        
        # 计算缩放和平移因子
        scale = self.scale_conv(conditions_t)
        shift = self.shift_conv(conditions_t)
        
        # 应用SFT: features * scale + shift
        transformed_features = features_t * (scale + 1.0) + shift
        
        # 转换回原始维度
        transformed_features = transformed_features.transpose(1, 2)
        
        return transformed_features

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 多头注意力投影
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 计算注意力
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # 应用注意力
        output = torch.matmul(attn_weights, v)
        output = output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        output = self.out_proj(output)
        
        return output
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, q, k, v):
        batch_size, q_len, _ = q.shape
        _, kv_len, _ = k.shape
        
        # 多头注意力投影
        q = self.q_proj(q).reshape(batch_size, q_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(k).reshape(batch_size, kv_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(v).reshape(batch_size, kv_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 计算注意力
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # 应用注意力
        output = torch.matmul(attn_weights, v)
        output = output.permute(0, 2, 1, 3).reshape(batch_size, q_len, -1)
        output = self.out_proj(output)
        
        return output

class FeatureProjector(nn.Module):
    """特征投影模块 - 可学习参数"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        # 特殊处理CLIP 512维降到256维的情况
        if input_dim == 512 and output_dim == 256:
            self.projection = nn.Sequential(
                nn.Linear(512, 256),
                nn.GELU(),
                nn.LayerNorm(256)
            )
            print(f"创建可学习的CLIP优化投影: 512 ->256 (带LayerNorm)")
            
        # 初始化
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """投影特征"""
        return self.projection(x)

class alignmodel(nn.Module):
    """文本图像特征对齐模型，使用跨注意力机制 - 支持可学习特征降维"""
    def __init__(self, dim=256, text_folder="./data/FFHQ_text", 
                 aligned_features_dir=None, 
                 text_features_dir=None,
                 is_training=True):
        super().__init__()
        self.dim = dim
        self.text_folder = text_folder
        self.aligned_features_dir = aligned_features_dir
        
        # 特征预处理MLP
        self.text_mlp = FeatureMLP(input_dim=dim)

        # 根据模式选择不同的特征目录
        if text_features_dir is not None:
            self.text_features_dir = text_features_dir
        elif is_training:
            self.text_features_dir = ""
        else:
            self.text_features_dir = "./models/DART/test_lfw"
        print(f"使用文本特征目录: {self.text_features_dir} ({'训练' if is_training else '验证/测试'}模式)")
        
        self.seq_len = 77  # CLIP的序列长度
        
        # 可学习特征投影层
        self.projectors = nn.ModuleDict()
    
        
        # 初始化常用的投影层
        self._create_projector(512, dim, prefix="text")  # 文本特征专用投影


        # 跨注意力和SFT模块
        self.cross_attention = CrossAttention(dim=dim)  
        self.sft_module = SFTModule(dim)
        
        # 特征缓存 - 只缓存原始特征
        self._feature_cache = {}

    def _create_projector(self, input_dim, output_dim, prefix=""):
        """创建特征投影器，可为不同特征类型创建专用投影器"""
        key = f"{prefix}_{input_dim}_{output_dim}" if prefix else f"{input_dim}_{output_dim}"
        if key not in self.projectors:
            self.projectors[key] = FeatureProjector(input_dim, output_dim)
            prefix_str = f"{prefix} " if prefix else ""
            print(f"创建{prefix_str}特征投影器: {input_dim} -> {output_dim}")
        return self.projectors[key]

    def forward(self, img_features, text_features):
        """模型前向传播 - 包含可学习的特征投影"""
        device = text_features.device
        # 1. 特征维度投影 - 在这里执行可学习的特征降维
        text_dim = text_features.shape[-1]
        if text_dim != self.dim:
            proj_key = f"text_{text_dim}_{self.dim}"
            if proj_key not in self.projectors:
                self._create_projector(text_dim, self.dim, prefix="text")
            
            # 确保投影层在正确的设备上

            projector = self.projectors[proj_key].to(device)
            text_features = projector(text_features)
        
        # 2. 使用MLP预处理特征
        text_features_processed = self.text_mlp(text_features)
        
        # 3. 使用sft模块处理图像特征
        sft_aligned_features = self.sft_module(img_features, text_features_processed)

        # 4. concat特征
        # concat_features = torch.cat([img_features, sft_aligned_features], dim=1)

        # 5. 自注意力对齐
        final_aligned_features = self.cross_attention(sft_aligned_features, img_features, img_features)

        return final_aligned_features

    def _load_feature(self, path):
        """仅加载特征，不进行任何处理"""
        # 检查缓存
        path_str = str(path)
        if path_str in self._feature_cache:
            return self._feature_cache[path_str]
        
        # 加载特征
        feature = torch.load(path, map_location="cpu")
        
        # 处理批次维度
        if len(feature.shape) == 2:
            feature = feature.unsqueeze(0)
        
        # 处理序列长度
        seq_len = feature.shape[1]
        if seq_len != self.seq_len:
            if seq_len > self.seq_len:
                # 截断
                feature = feature[:, :self.seq_len, :]
            else:
                # 填充
                padding = torch.zeros(
                    feature.shape[0], 
                    self.seq_len-seq_len, 
                    feature.shape[2],
                    device="cpu"
                )
                feature = torch.cat([feature, padding], dim=1)
        
        # 缓存并返回
        self._feature_cache[path_str] = feature
        return feature

    def get_text_features(self, images=None, filenames=None, batch=None):
        """加载文本特征 - 不进行降维，降维在forward中进行"""
        # 确定设备
        caller_device = images.device if images is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 优先使用batch中的gt_path
        if batch is not None and 'gt_path' in batch:
            filenames = batch['gt_path']
        
        # 如果没有文件名，返回随机特征
        if filenames is None:
            batch_size = images.shape[0] if images is not None else 1
            return self._get_random_features(caller_device, batch_size)
        
        # 标准化filenames格式
        if isinstance(filenames, str):
            filenames = [filenames]
            
        # 处理批次大小
        batch_size = len(filenames)
        if images is not None and images.shape[0] != batch_size:
            print(f"警告: 图像批次大小({images.shape[0]})与文件名数量({batch_size})不匹配")
            batch_size = images.shape[0]
            # 处理批次大小不匹配
            filenames = filenames[:batch_size] if len(filenames) > batch_size else filenames + [filenames[-1]] * (batch_size - len(filenames))
        
        # 批处理文本特征
        text_features_list = []
        found_count = 0
        
        for i in range(batch_size):
            # 内存管理
            if i > 0 and i % 4 == 0:
                torch.cuda.empty_cache()
                
            filename = filenames[i]
            img_id = self._extract_image_id(filename)
            
            # 构造可能的特征路径
            potential_paths = [
                Path(self.text_features_dir) / f"{img_id}_text.pt",
                Path(self.text_features_dir) / f"{img_id.zfill(5)}_text.pt",
                Path(self.text_features_dir) / f"{os.path.basename(filename)}_text.pt"
            ]
            
            feature_loaded = False
            for feature_path in potential_paths:
                if feature_path.exists():
                    try:
                        # 仅加载特征，不进行降维
                        text_feature = self._load_feature(feature_path)
                        # 转到目标设备
                        text_feature = text_feature.to(caller_device, dtype=torch.float16)
                        text_features_list.append(text_feature)
                        found_count += 1
                        feature_loaded = True
                        break
                    except Exception as e:
                        print(f"处理特征出错 {feature_path}: {str(e)}")
            
            if not feature_loaded:
                # 未找到特征，使用随机特征
                text_features_list.append(self._get_random_features(caller_device, 1))
        
        # 合并特征
        if found_count > 0:
            print(f"已找到 {found_count}/{batch_size} 个文本特征文件")
        
        try:
            return torch.cat(text_features_list, dim=0)
        except Exception as e:
            print(f"合并特征时出错: {e}")
            return self._get_random_features(caller_device, batch_size)

    def get_tag_features(self, filenames=None, batch=None):
        """加载标签特征 - 不进行降维，降维在forward中进行"""
        # 确定设备
        caller_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 优先使用batch中的gt_path
        if batch is not None and 'gt_path' in batch:
            filenames = batch['gt_path']
        
        # 如果没有文件名，返回随机特征
        if filenames is None:
            batch_size = len(batch['gt_path']) if batch is not None else 1
            return self._get_random_features(caller_device, batch_size)
        
        # 标准化filenames格式
        if isinstance(filenames, str):
            filenames = [filenames]
        
        # 处理批次大小
        batch_size = len(filenames)
        tag_features_list = []
        found_count = 0
        
        for i in range(batch_size):
            filename = filenames[i]
            tag_id = self._extract_image_id(filename)
            
            # 构造标签特征路径
            tag_feature_path = Path(self.tag_features_dir) / f"{tag_id}_tag.pt"
            
            if tag_feature_path.exists():
                try:
                    # 仅加载特征，不进行降维
                    tag_feature = self._load_feature(tag_feature_path)
                    # 转到目标设备
                    tag_feature = tag_feature.to(caller_device, dtype=torch.float16)
                    tag_features_list.append(tag_feature)
                    found_count += 1
                except Exception as e:
                    print(f"加载标签特征出错 {tag_feature_path}: {str(e)}")
                    tag_features_list.append(self._get_random_features(caller_device, 1))
            else:
                # 未找到特征，使用随机特征
                tag_features_list.append(self._get_random_features(caller_device, 1))
        
        # 合并特征
        if found_count > 0:
            print(f"已找到 {found_count}/{batch_size} 个标签特征文件")
        
        try:
            return torch.cat(tag_features_list, dim=0)
        except Exception as e:
            print(f"合并标签特征时出错: {e}")
            return self._get_random_features(caller_device, batch_size)

    def _extract_image_id(self, path, keep_zeros=True):
        """从路径提取图像ID"""
        basename = os.path.basename(path)
        basename = os.path.splitext(basename)[0]
        return basename

    def _get_random_features(self, device, batch_size=1):
        """生成随机特征"""
        if device.type == 'cuda' and torch.cuda.is_available():
            features = []
            chunk_size = min(batch_size, 4)
            
            for i in range(0, batch_size, chunk_size):
                end_idx = min(i + chunk_size, batch_size)
                chunk_size_actual = end_idx - i
                chunk = torch.randn(chunk_size_actual, self.seq_len, self.dim, 
                                  device=device, dtype=torch.float16)
                features.append(chunk)
                
                if i > 0 and i % 8 == 0:
                    torch.cuda.empty_cache()
            
            return torch.cat(features, dim=0)
        else:
            return torch.randn(batch_size, self.seq_len, self.dim, 
                              device=device, dtype=torch.float16)

    def set_eval_mode(self, eval_mode=True):
        """设置评估模式，切换文本特征目录"""
        if eval_mode:
            self.text_features_dir = "/home/yanhy27/models/DART/test_lfw"
            # print(f"切换到验证/测试特征目录: {self.text_features_dir}")
        else:
            self.text_features_dir = ""
        
        # 清除特征缓存
        self._feature_cache = {}
    
