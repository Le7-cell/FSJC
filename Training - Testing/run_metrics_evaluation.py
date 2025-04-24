# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 16:52:06 2021

@author: Admin
"""
import torch
from metric_evaluation import plot_confusion_matrix, iterate_data, spectrum_score_norm, spectrum_score
import glob
import os

# 设置路径
data_dir = os.path.join('D:', 'BS', 'FSJC', 'Dateset-1')
model_dir = os.path.join('D:', 'BS', 'FSJC', 'DeepV3')

# 删除旧的模型加载代码
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载所有模型并评估
pt_files = glob.glob(os.path.join(model_dir, '*.pt'))
for model_path in pt_files:
    print(f"\n正在评估模型: {os.path.basename(model_path)}")
    model = torch.load(model_path, map_location=torch.device('cuda'))
    model.to(device)
    model.eval()
    
    iOU, f1, confm_sum, y_pred = iterate_data(model, data_dir)
    
    print('iOU: ' + str(iOU))
    print('f1 score: ' + str(f1))
    
    plot_confusion_matrix(confm_sum, target_names=['Background', 'Fair', 'Poor', 'Severe'], 
                         normalize=True, title=f'Confusion Matrix - {os.path.basename(model_path)}')
