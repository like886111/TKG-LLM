#!pip install transformers

import math
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import pandas as pd
import pywt  # 添加PyWavelets库导入

# 导入GPT-2模型相关组件
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from einops import rearrange
from transformers import GPT2Tokenizer
from utils.tokenization import SerializerSettings, serialize_arr,serialize_arr 
from .prompt import Prompt 
from .knowledge_graph import KnowledgeGraphEnhancedPrompt  # 导入知识图谱增强的提示学习模块

# TKG-LLM模型类：结合语义空间表示和提示学习的时间序列预测模型
class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.is_ln = configs.ln
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.d_ff = 768
        # 计算patch数量
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1
        # 创建填充层，用于patch切分
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
       
        # 初始化GPT-2 tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('', trust_remote_code=True, local_files_only=True)
        # GPT2默认没有pad_token，设置为eos_token
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 初始化GPT-2模型
        if configs.pretrained == True:
            # 使用预训练的GPT-2模型
            self.gpt2 = GPT2Model.from_pretrained('', output_attentions=True, output_hidden_states=True)
            # 只使用指定数量的GPT-2层
            self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
        else:
            # 使用随机初始化的GPT-2模型
            print("------------------no pretrain------------------")
            self.gpt2 = GPT2Model(GPT2Config())
        
        # 冻结大部分GPT-2参数，只训练部分参数
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name:   # 只训练层归一化和位置编码
                param.requires_grad = True
            else:
                param.requires_grad = False  # 冻结其他参数

        # 针对长期预测任务的特定设置
        if self.task_name == 'long_term_forecast':
            # 输入层：将patch映射到d_model维度
            self.in_layer = nn.Linear(configs.patch_size*3, configs.d_model)
            # 输出层：将GPT-2输出映射到预测长度
            self.out_layer = nn.Linear(int(configs.d_model / 3 * (self.patch_num+configs.prompt_length)) , configs.pred_len)
            
            # 初始化知识图谱增强的提示学习模块
            self.use_kg_enhanced_prompt = getattr(configs, 'use_kg_enhanced_prompt', True)
            if self.use_kg_enhanced_prompt:
                self.kg_enhanced_prompt = KnowledgeGraphEnhancedPrompt(
                    seq_len=configs.seq_len,
                    num_features=configs.enc_in,  # 使用输入特征数量
                    embed_dim=configs.d_model,
                    hidden_dim=getattr(configs, 'kg_hidden_dim', 256),
                    fusion_type=getattr(configs, 'kg_fusion_type', 'concat')
                )
            
            # 初始化提示池：用于动态选择最相关的提示
            self.prompt_pool = Prompt(length=1, embed_dim=768, embedding_key='mean', prompt_init='uniform', 
                                    prompt_pool=False, prompt_key=True, pool_size=self.configs.pool_size, 
                                    top_k=self.configs.prompt_length, batchwise_prompt=False, 
                                    prompt_key_init=self.configs.prompt_init, wte=self.gpt2.wte.weight)
            
            # 将模型移动到GPU并设置为训练模式
            layers_to_cuda = [self.gpt2, self.in_layer, self.out_layer]
            if self.use_kg_enhanced_prompt:
                layers_to_cuda.append(self.kg_enhanced_prompt)
                
            for layer in layers_to_cuda:       
                layer.cuda()
                layer.train()

    # 前向传播函数
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # 根据任务类型调用不同的处理函数
        if self.task_name == 'long_term_forecast':
            dec_out, res = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :], res  # 返回预测结果和额外信息
        return None

    # 长期预测函数
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 获取输入张量的形状
        B, L, M = x_enc.shape
            
        # 数据标准化
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
 
        # 重新排列输入张量，准备进行分解
        x = rearrange(x_enc, 'b l m -> (b m) l') 

        # 时间序列分解函数：使用小波分解替代STL分解
        def decompose(x):
            # 确保数据长度是2的幂，如果不是则进行填充
            n = len(x)
            pad_length = 2**int(np.ceil(np.log2(n))) - n
            if pad_length > 0:
                x_padded = np.pad(x, (0, pad_length), mode='edge')
            else:
                x_padded = x
                
            # 选择小波基函数和分解级别
            wavelet = 'db4'  # 使用Daubechies 4小波
            level = min(3, pywt.dwt_max_level(len(x_padded), pywt.Wavelet(wavelet).dec_len))
            
            # 执行小波分解
            coeffs = pywt.wavedec(x_padded, wavelet, level=level)
            
            # 重构各个分量
            # 近似分量（低频部分，对应趋势）
            trend_coeffs = [coeffs[0]] + [None] * level
            trend = pywt.waverec(trend_coeffs, wavelet)
            trend = trend[:n]
            
            # 第一层细节（对应季节性）
            seasonal_coeffs = [None] + [coeffs[1]] + [None] * (level-1)
            seasonal = pywt.waverec(seasonal_coeffs, wavelet)
            seasonal = seasonal[:n]
            
            # 计算残差：原始信号减去趋势和季节性
            residuals = x - trend - seasonal
            
            # 将三个分量堆叠在一起
            combined = np.stack([trend, seasonal, residuals], axis=1)
            return combined
                
        # 对所有时间序列应用分解
        # 确保在GPU上进行计算
        device = next(self.parameters()).device
        x_cpu = x.cpu().numpy()
        decomp_results = np.apply_along_axis(decompose, 1, x_cpu)
        x = torch.tensor(decomp_results).to(device)
        
        # 修复：调整张量维度处理
        # 原始形状: [B*M, L, 3]
        # 需要调整为: [B*M, 3, L]
        x = rearrange(x, 'b l c -> b c l')
        
        # 填充数据以便更好地进行patch切分
        x = self.padding_patch_layer(x)
        # 使用滑动窗口创建patch
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        # 将趋势、季节性和残差的patch拼接在一起
        x = rearrange(x, 'b c n p -> b n (c p)', c = 3)  
        # 通过MLP处理拼接后的patch获取embedding
        pre_prompted_embedding = self.in_layer(x.float())

        # 使用知识图谱增强的提示学习模块
        if self.use_kg_enhanced_prompt:
            # 确保x_enc在GPU上
            x_enc = x_enc.to(device)
            enhanced_embedding = self.kg_enhanced_prompt(pre_prompted_embedding, x_enc)
        else:
            enhanced_embedding = pre_prompted_embedding

        # 从提示池中选择最相关的提示并应用
        outs = self.prompt_pool(enhanced_embedding)
        prompted_embedding = outs['prompted_embedding']
        sim = outs['similarity']
        prompt_key = outs['prompt_key']
        simlarity_loss = outs['reduce_sim']

        # 使用GPT-2处理带有提示的嵌入
        last_embedding = self.gpt2(inputs_embeds=prompted_embedding).last_hidden_state
        # 通过输出层生成预测结果
        outputs = self.out_layer(last_embedding.reshape(B*M*3, -1))
            
        # 重新排列输出张量
        outputs = rearrange(outputs, '(b m c) h -> b m c h', b=B, m=M, c=3)
        # 合并三个分量的预测结果
        outputs = outputs.sum(dim=2)
        # 重新排列为最终输出格式
        outputs = rearrange(outputs, 'b m l -> b l m')

        # 收集额外信息
        res = dict()
        res['simlarity_loss'] = simlarity_loss
            
        # 反标准化预测结果
        outputs = outputs * stdev[:,:,:M]
        outputs = outputs + means[:,:,:M]

        return outputs, res





    










