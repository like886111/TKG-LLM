import numpy as np
import torch
import torch.nn as nn
from torch import optim
import pandas as pd
import pywt  # 添加PyWavelets库导入

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from einops import rearrange
from transformers import GPT2Tokenizer
from utils.tokenization import SerializerSettings, serialize_arr,serialize_arr 
from .prompt import Prompt
from .knowledge_graph import KnowledgeGraphEnhancedPrompt  # 导入知识图谱增强的提示学习模块


class TKGLLM(nn.Module):
    
    def __init__(self, configs, device):
        super(TKGLLM, self).__init__()
        self.configs = configs
        # self.is_gpt = configs.is_gpt
        self.patch_size = configs.patch_size
        # self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        
        self.gpt2 = GPT2Model.from_pretrained('', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
            # else:
            #     print("------------------no pretrain------------------")
            #     self.gpt2 = GPT2Model(GPT2Config())
        self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
        print("gpt2 = {}".format(self.gpt2))



        self.in_layer = nn.Linear(configs.patch_size*3, configs.d_model)  
        
        # 初始化知识图谱增强的提示学习模块
        self.use_kg_enhanced_prompt = getattr(configs, 'use_kg_enhanced_prompt', True)
        if self.use_kg_enhanced_prompt:
            self.kg_enhanced_prompt = KnowledgeGraphEnhancedPrompt(
                seq_len=configs.seq_len,
                num_features=configs.enc_in,
                embed_dim=configs.d_model,
                hidden_dim=getattr(configs, 'kg_hidden_dim', 256),
                fusion_type=getattr(configs, 'kg_fusion_type', 'concat')
            )
        
        self.prompt_pool = Prompt(length=1, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                 prompt_key=True, pool_size=self.configs.pool_size, top_k=self.configs.prompt_length, batchwise_prompt=False, prompt_key_init=self.configs.prompt_init,wte = self.gpt2.wte.weight)

        
       
        
        self.out_layer = nn.Linear(int(configs.d_model / 3 * (self.patch_num+configs.prompt_length)), configs.pred_len)
        
       
        
        # if configs.freeze and configs.pretrain:
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

     
        
        self.cnt = 0


    def forward(self, x, itr):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
        x /= stdev

        x = rearrange(x, 'b l m -> (b m) l')

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

        decomp_results = np.apply_along_axis(decompose, 1, x.cpu().numpy())
        x = torch.tensor(decomp_results).to(self.gpt2.device)
        x = rearrange(x, 'b l c -> b c l')
            
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
            
        x = rearrange(x, 'b c n p -> b n (c p)', c = 3)  
        pre_prompted_embedding = self.in_layer(x.float())

        # 使用知识图谱增强的提示学习模块
        if self.use_kg_enhanced_prompt:
            enhanced_embedding = self.kg_enhanced_prompt(pre_prompted_embedding, x)
        else:
            enhanced_embedding = pre_prompted_embedding
       
        outs = self.prompt_pool(enhanced_embedding)
        prompted_embedding = outs['prompted_embedding']
        sim = outs['similarity']
        prompt_key = outs['prompt_key']
        simlarity_loss = outs['reduce_sim']

        last_embedding = self.gpt2(inputs_embeds=prompted_embedding).last_hidden_state
        outputs = self.out_layer(last_embedding.reshape(B*M*3, -1))
            
        outputs = rearrange(outputs, '(b m c) h -> b m c h', b=B,m=M,c=3)
        outputs = outputs.sum(dim=2)
        outputs = rearrange(outputs, 'b m l -> b l m')

        res = dict()
        res['simlarity_loss'] = simlarity_loss

        outputs = outputs * stdev
        outputs = outputs + means

        return outputs,res
        




        