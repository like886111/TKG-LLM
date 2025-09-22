import torch
import torch.nn as nn




class Prompt(nn.Module):
    """
    语义空间引导的提示学习模块
    
    该模块实现了基于语义空间的提示学习机制，通过维护一个提示池(prompt pool)，
    并根据输入序列的语义特征动态选择最相关的提示进行预测。
    
    主要功能:
    1. 维护一个可学习的提示池
    2. 根据输入序列的语义特征选择最相关的提示
    3. 将选定的提示与输入序列拼接，增强模型的预测能力
    """
    def __init__(self, length=2, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                 prompt_key=False, pool_size=30, top_k=4, batchwise_prompt=False, prompt_key_init='uniform',wte = None):
        """
        初始化提示学习模块
        
        参数:
            length (int): 每个提示的长度
            embed_dim (int): 嵌入维度，与模型隐藏层维度相同
            embedding_key (str): 计算输入序列嵌入的方法，可选'mean', 'max', 'mean_max', 'cls'
            prompt_init (str): 提示初始化方法，可选'zero', 'uniform'
            prompt_pool (bool): 是否使用提示池机制
            prompt_key (bool): 是否使用可学习的提示键
            pool_size (int): 提示池大小
            top_k (int): 为每个输入序列选择的提示数量
            batchwise_prompt (bool): 是否在批次级别选择提示
            prompt_key_init (str): 提示键初始化方法，可选'zero', 'uniform', 'gaussian', 'text_prototype'
            wte (torch.Tensor): 词嵌入矩阵，用于text_prototype初始化
        """
        super().__init__()

        self.length = length
        self.embed_dim = embed_dim
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.prompt_key_init = prompt_key_init
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.wte = wte

        # 初始化提示池
        if self.prompt_pool:
            prompt_pool_shape = (pool_size, length, embed_dim)
            if prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
            elif prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.uniform_(self.prompt, -1, 1)
        
        # 初始化提示键
        if prompt_key:
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(key_shape),requires_grad=False)
                print('zero initialized key')
                
            elif prompt_key_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(key_shape),requires_grad=False)
                nn.init.uniform_(self.prompt, -5, 5)
                print('uniform initialized key')
            
            elif prompt_key_init == 'gaussian':
                self.prompt = nn.Parameter(torch.randn(key_shape),requires_grad=False)
                nn.init.normal_(self.prompt, mean=0.0, std=5.0)
                print('gaussian initialized key')

            elif prompt_key_init == 'text_prototype':
                # 使用词嵌入矩阵初始化提示键
                self.text_prototype_linear = nn.Linear(50257, pool_size)
                
        else:
            # 如果不使用可学习的提示键，则使用提示的平均值作为键
            prompt_mean = torch.mean(self.prompt, dim=1)
            self.prompt_key = prompt_mean
    
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """
        L2归一化函数
        
        参数:
            x (torch.Tensor): 需要归一化的张量
            dim (int): 归一化的维度
            epsilon (float): 防止除零的小常数
            
        返回:
            torch.Tensor: 归一化后的张量
        """
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def forward(self, x_embed, prompt_mask=None, cls_features=None):
        """
        前向传播函数
        
        参数:
            x_embed (torch.Tensor): 输入序列的嵌入表示 [batch_size, seq_len, embed_dim]
            prompt_mask (torch.Tensor, optional): 预定义的提示掩码
            cls_features (torch.Tensor, optional): 分类特征，用于'cls'嵌入方法
            
        返回:
            dict: 包含以下键值对的字典:
                - 'prompt_idx': 选择的提示索引
                - 'prompt_norm': 归一化的提示键
                - 'x_embed_norm': 归一化的输入嵌入
                - 'similarity': 相似度矩阵
                - 'selected_key': 选择的提示键
                - 'reduce_sim': 减少的相似度
                - 'total_prompt_len': 提示总长度
                - 'prompted_embedding': 拼接提示后的嵌入
                - 'prompt_key': 提示键
        """
        out = dict()
        if self.prompt_key:   #if self.prompt_pool:
            # 根据指定的方法计算输入序列的嵌入表示
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0] # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")

            # 根据初始化方法获取提示键
            if self.prompt_key_init == 'text_prototype':
                prompt_key = self.text_prototype_linear(self.wte.transpose(0, 1)).transpose(0, 1)
            
            else:
                prompt_key = self.prompt
            
            # 对提示键和输入嵌入进行L2归一化
            prompt_norm = self.l2_normalize(prompt_key, dim=1) # Pool_size, C   self.prompt_key
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=1) # B, C

            # 计算相似度矩阵
            similarity = torch.matmul(x_embed_norm, prompt_norm.t()) # B, Pool_size
            
            # 选择最相似的top_k个提示
            if prompt_mask is None:
                _, idx = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k
                if self.batchwise_prompt:
                    # 在批次级别选择提示
                    prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                    # 在jnp.unique中，当指定'size'并且元素数量少于指定数量时，
                    # 剩余元素将用'fill_value'填充，默认为指定维度的最小值。
                    # 除非指定维度，否则如果不是1D，这将被展平。
                    if prompt_id.shape[0] < self.pool_size:
                        prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
                        id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                    _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
                    major_prompt_id = prompt_id[major_idx] # top_k
                    # 扩展到批次
                    idx = major_prompt_id.expand(x_embed.shape[0], -1) # B, top_k
            else:
                idx = prompt_mask # B, top_k

            # 获取选定的提示
            batched_prompt_raw = prompt_key[idx] # B, top_k, length, C
            batched_prompt_raw = batched_prompt_raw.unsqueeze(2) # B, top_k, 1, length, C

            # 重塑提示以便与输入序列拼接
            batch_size, top_k, length, c = batched_prompt_raw.shape
            batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c) # B, top_k * length, C

            # 保存中间结果用于调试和损失计算
            out['prompt_idx'] = idx
            out['prompt_norm'] = prompt_norm
            out['x_embed_norm'] = x_embed_norm
            out['similarity'] = similarity

            # 计算pull_constraint损失
            batched_key_norm = prompt_norm[idx] # B, top_k, C
            out['selected_key'] = batched_key_norm
            x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
            sim = batched_key_norm * x_embed_norm # B, top_k, C
            reduce_sim = torch.sum(sim) / x_embed.shape[0] # Scalar

            out['reduce_sim'] = reduce_sim
        else:
            # 如果不使用提示池，则使用单个提示
            if self.prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(self.length, self.embed_dim))
            elif self.prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(self.length, self.embed_dim))
                nn.init.uniform_(self.prompt)
            batched_prompt = self.prompt.unsqueeze(0).expand(x_embed.shape[0], -1, -1)
        
        # 将提示与输入序列拼接
        out['total_prompt_len'] = batched_prompt.shape[1]
        out['prompted_embedding'] = torch.cat([batched_prompt, x_embed], dim=1)
        out['prompt_key'] = prompt_key  # prompt_key

        return out