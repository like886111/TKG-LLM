import torch
import numpy as np

# ========== 1. 与训练时完全一致的超参 ==========
ckp_path = './checkpoints/long_term_forecast_test_TKGLLM_ETTh1_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc1_eb_timeF_dtTrue_visual_0/checkpoint.pth'
seq_len, pred_len, enc_in, d_model, patch_size, stride = 96, 96, 7, 512, 1, 1
gpu = 0
# ===============================================

device = torch.device(f'cuda:{gpu}')
model = Model(argparse.Namespace(
        task_name='long_term_forecast', seq_len=seq_len, pred_len=pred_len,
        enc_in=enc_in, d_model=d_model, patch_size=patch_size, stride=stride,
        gpt_layers=6, pretrained=True, use_kg_enhanced_prompt=True,
        kg_hidden_dim=256, kg_fusion_type='concat', pool_size=1000,
        prompt_length=1, prompt_init='text_prototype')).to(device)
model.load_state_dict(torch.load(ckp_path, map_location=device))
model.eval()

# ========== 2. 取一条样本（验证集第 0 条） ==========
data_set = Dataset_ETT_hour(root_path='./data/raw_data/ETTh1/',
                            data_path='ETTh1.csv',
                            flag='val',
                            size=[seq_len, 0, pred_len],
                            features='M')
x, y, _, _ = data_set[0]          # x:[seq_len, dim], y:[pred_len, dim]
x_in = x.unsqueeze(0).to(device)  # [1, seq_len, dim]
y_true = y.numpy()

# ========== 3. 推理 ==========
with torch.no_grad():
    y_pred, _ = model(x_in, None, None, None)   # [1, pred_len, dim]
y_pred = y_pred.cpu().numpy()[0]

# ========== 4. 画图（第一维） ==========
plt.figure(figsize=(10,3))
plt.plot(np.arange(seq_len), x[:,0], label='History')
plt.plot(np.arange(seq_len, seq_len+pred_len), y_true[:,0], label='GT')
plt.plot(np.arange(seq_len, seq_len+pred_len), y_pred[:,0], label='Pred')
plt.axvline(seq_len, color='gray', ls='--')
plt.legend(); plt.tight_layout()
plt.savefig('quick_pred.png', dpi=150)
plt.show()