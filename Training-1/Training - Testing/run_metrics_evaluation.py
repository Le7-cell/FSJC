# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 16:52:06 2021

@author: Admin
"""
import torch
from metric_evaluation import plot_confusion_matrix, iterate_data, spectrum_score_norm, spectrum_score
# 确保只使用torch.hub的导入方式
from torch.hub import load_state_dict_from_url


# 修改前: data_dir = './D:\BS\FSJC\Dateset-1/'
data_dir = r'D:\BS\FSJC\Dateset-1'  # 使用原始字符串(raw string)表示路径
batchsize = 1

# 修改前: model = torch.load(f'D:\BS\FSJC\DeepV3\Corrosion Condition State Classification - Trained Model\var_original_wbatch_2_plus\var_original_wbatch_2_plus_weights_40.pt', map_location=torch.device('cuda'))
model = torch.load(r'D:\BS\FSJC\DeepV3\Corrosion Condition State Classification - Trained Model\var_aug_batch_2_resnet50\var_aug_batch_2_resnet50_weights_18.pt', map_location=torch.device('cuda'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()   # Set model to evaluate mode
##############################################################################

iOU, f1, confm_sum, y_pred = iterate_data(model, data_dir)

print('iOU: ' + str(iOU))
print('f1 score: ' + str(f1))

plot_confusion_matrix(confm_sum, target_names=['Background', 'Fair', 'Poor', 'Severe'], normalize=True, 
                      title='Confusion Matrix')