import os
import glob
import torch
import random
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class CustomPairedDataset(Dataset):
    """支持从同一文件夹选择不同子集的数据集"""
    
    def __init__(self, lq_folder, hq_folder, image_size=512, split='all', train_ratio=0.8, seed=42):
        """
        参数:
            lq_folder: 低质量图像文件夹
            hq_folder: 高质量图像文件夹
            image_size: 图像大小
            split: 'train', 'val', 'all' 中的一个
            train_ratio: 训练集比例
            seed: 随机种子，确保相同的分割
        """
        super().__init__()
        self.lq_folder = lq_folder
        self.hq_folder = hq_folder
        self.image_size = image_size
        
        # 获取所有文件名
        self.lq_files = sorted(os.listdir(lq_folder))
        
        # 设置随机种子确保分割一致性
        random.seed(seed)
        
        # 打乱文件列表
        all_indices = list(range(len(self.lq_files)))
        random.shuffle(all_indices)
        
        # 根据分割类型选择子集
        if split == 'train':
            split_idx = int(len(all_indices) * train_ratio)
            self.indices = all_indices[:split_idx]
        elif split == 'val':
            split_idx = int(len(all_indices) * train_ratio)
            self.indices = all_indices[split_idx:]
        else:
            self.indices = all_indices
            # 添加图像变换
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        print(f"Dataset {split}: {len(self.indices)} images")


    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        # 使用映射索引
        file_idx = self.indices[idx]
        file_name = self.lq_files[file_idx]
        
        # 以下为原有的加载逻辑
        lq_path = os.path.join(self.lq_folder, file_name)
        hq_path = os.path.join(self.hq_folder, file_name)
        
        # 加载图像
        lq_img = Image.open(lq_path).convert('RGB')
        hq_img = Image.open(hq_path).convert('RGB')
        
        # 应用变换
        lq_tensor = self.transform(lq_img)
        hq_tensor = self.transform(hq_img)
        
        
        # 返回字典格式
        return {
            'lq': lq_tensor,
            'gt': hq_tensor,
            'filename': os.path.basename(lq_path)
        }

class CelebARandomPairedDataset(Dataset):
    """按照文件名顺序配对CelebA测试集和验证集图像"""
    
    def __init__(self, lq_folder, hq_folder, image_size=512, limit=None):
        """
        参数:
            lq_folder: 低质量图像文件夹路径
            hq_folder: 高质量图像文件夹路径
            image_size: 图像大小
            limit: 可选，限制图像数量
        """
        super().__init__()
        self.lq_folder = lq_folder
        self.hq_folder = hq_folder
        self.image_size = image_size
        
        # 获取排序后的文件列表
        self.lq_files = sorted(os.listdir(lq_folder))
        self.hq_files = sorted(os.listdir(hq_folder))
        
        # 确保文件数量相同或处理不匹配情况
        min_length = min(len(self.lq_files), len(self.hq_files))
        if min_length != len(self.lq_files) or min_length != len(self.hq_files):
            print(f"警告: LQ文件({len(self.lq_files)})和HQ文件({len(self.hq_files)})数量不匹配")
            self.lq_files = self.lq_files[:min_length]
            self.hq_files = self.hq_files[:min_length]
        
        # 如果需要限制数量
        if limit is not None and limit > 0 and limit < min_length:
            self.lq_files = self.lq_files[:limit]
            self.hq_files = self.hq_files[:limit]
        
        # 记录图像配对信息
        self.file_pairs = list(zip(self.lq_files, self.hq_files))
        print(f"CelebA有序配对数据集: 共{len(self.file_pairs)}对图像")
        print(f"前5对: {self.file_pairs[:5]}")
        
        # 图像变换
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.lq_files)
        
    def __getitem__(self, idx):
        # 获取对应的LQ和HQ文件名
        lq_file = self.lq_files[idx]
        hq_file = self.hq_files[idx]
        
        # 构建完整路径
        lq_path = os.path.join(self.lq_folder, lq_file)
        hq_path = os.path.join(self.hq_folder, hq_file)
        
        # 加载图像
        try:
            lq_img = Image.open(lq_path).convert('RGB')
            hq_img = Image.open(hq_path).convert('RGB')
        except Exception as e:
            print(f"加载图像出错: {e}, LQ:{lq_path}, HQ:{hq_path}")
            # 提供一个fallback选项
            return self.__getitem__((idx + 1) % len(self))
        
        # 应用变换
        lq_tensor = self.transform(lq_img)
        hq_tensor = self.transform(hq_img)
        
        # 返回字典格式，确保包含filename
        return {
            'lq': lq_tensor,
            'gt': hq_tensor,
            'filename': os.path.basename(lq_file)  # 使用LQ文件名作为标识符
        }
class RealImageDataset(Dataset):
    """通用真实数据集加载器 - 无GT模式，适用于任何真实世界图像集"""
    def __init__(self, lq_folder, image_size=512, recursive=True, limit=None, extensions=('.png', '.jpg', '.jpeg', '.webp')):
        """
        参数:
            image_folder: 图像文件夹路径
            image_size: 调整后的图像大小
            recursive: 是否递归搜索子文件夹
            limit: 可选，限制加载的图像数量
            extensions: 支持的图像文件扩展名
        """
        self.image_folder = lq_folder
        self.image_size = image_size
        
        # 获取所有图像文件
        self.image_paths = []
        
        if recursive:
            # 递归搜索所有子文件夹
            for root, _, files in os.walk(lq_folder):
                for file in files:
                    if file.lower().endswith(extensions):
                        self.image_paths.append(os.path.join(root, file))
        else:
            # 只搜索指定文件夹
            for ext in extensions:
                self.image_paths.extend(glob.glob(os.path.join(lq_folder, f"*{ext}")))
                self.image_paths.extend(glob.glob(os.path.join(lq_folder, f"*{ext.upper()}")))
        
        # 排序以确保一致的顺序
        self.image_paths = sorted(self.image_paths)
        
        # 限制图像数量
        if limit is not None and limit > 0 and limit < len(self.image_paths):
            self.image_paths = self.image_paths[:limit]
        
        # 默认变换
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        print(f"已加载 {len(self.image_paths)} 张真实图像，来自 {lq_folder}")
        print(f"前5张图像: {[os.path.basename(p) for p in self.image_paths[:5]]}")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        # 加载图像
        img_path = self.image_paths[index]
        
        try:
            img = Image.open(img_path).convert('RGB')
            
            # 应用变换
            img_tensor = self.transform(img)
            
            # 关键点：不提供gt键，这将触发无GT模式
            return {
                "lq": img_tensor,       # 只提供输入图像
                "filename": os.path.basename(img_path)
            }
        except Exception as e:
            print(f"加载图像 {img_path} 出错: {e}")
            # 发生错误时返回下一个图像
            return self.__getitem__((index + 1) % len(self))