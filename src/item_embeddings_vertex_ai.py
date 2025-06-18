from typing import Dict, List, Set
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch

class ItemEmbeddingGenerator:
    def __init__(self, 
                #model_name: str = "BAAI/bge-large-zh-v1.5",
                model_name: str = "BAAI/bge-small-zh-v1.5",
                include_fields: Set[str] = None,
                device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        使用 BGE 模型生成嵌入向量
        
        Args:
            model_name: BGE 模型名称（默认中文优化版本）
            include_fields: 包含的字段（title/description/category等）
            device: 运行设备（自动检测GPU）
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()  # 设置为评估模式
        self.device = device
        self.include_fields = include_fields or {'title', 'description', 'category'}
        
    def create_embedding_input(self, item_data: Dict) -> str:
        """生成 BGE 兼容的输入文本（简化版）"""
        prompt_parts = []
        
        if 'description' in self.include_fields:
            desc = str(item_data.get('description', '')).strip()
            if desc:
                prompt_parts.append(f"商品描述：{desc}")
        
        if 'title' in self.include_fields and (title := item_data.get('title')):
            prompt_parts.append(f"商品标题：{title}")
            
        if 'category' in self.include_fields and (cats := item_data.get('categories')):
            category_str = " > ".join(cats[0] if isinstance(cats[0], list) else cats)
            if category_str:
                prompt_parts.append(f"商品类别：{category_str}")
                
        # 其他字段处理（品牌/价格等）
        if 'brand' in self.include_fields and (brand := item_data.get('brand')):
            if not (brand.startswith('B0') and len(brand) >= 10):
                prompt_parts.append(f"品牌：{brand}")
        
        return "。".join(prompt_parts)  # 中文用句号分隔更自然

    def generate_item_embeddings(self, items: Dict) -> Dict[str, np.ndarray]:
        """生成嵌入向量（支持批量处理）"""
        embeddings = {}
        texts = [self.create_embedding_input(item_data) for _, item_data in items.items()]
        
        # 批量编码（建议 batch_size=32-128 根据显存调整）
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            with torch.no_grad():
                inputs = self.tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    return_tensors="pt",
                    max_length=512
                ).to(self.device)
                
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()  # 取[CLS]位置作为句向量
                
            # 归一化（BGE 建议使用）
            batch_embeddings = batch_embeddings / np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
            
            for idx, (item_id, _) in enumerate(list(items.items())[i:i + batch_size]):
                embeddings[item_id] = batch_embeddings[idx]
                
        return embeddings

    def debug_prompt(self, items: Dict, num_samples: int = 3):
        """调试输入文本"""
        print("\nBGE 输入文本示例：")
        print("=" * 80)
        for item_id in list(items.keys())[:num_samples]:
            print(f"\nItem ID: {item_id}")
            print("-" * 40)
            print(self.create_embedding_input(items[item_id]))
            print("=" * 80)

if __name__ == "__main__":
    # 测试数据
    items = {
        "item1": {"title": "智能手机", "description": "6.5英寸AMOLED屏幕", "categories": ["电子产品", "手机"]},
        "item2": {"title": "蓝牙耳机", "description": "主动降噪，30小时续航", "categories": ["数码配件", "音频设备"]},
    }
    
    # 初始化生成器（自动使用GPU）
    generator = ItemEmbeddingGenerator()
    
    # 调试输入
    generator.debug_prompt(items)
    
    # 生成嵌入向量
    embeddings = generator.generate_item_embeddings(items)
    print(f"\n生成的嵌入向量维度：{next(iter(embeddings.values())).shape}")
