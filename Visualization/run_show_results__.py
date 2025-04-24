# -*- coding: utf-8 -*-

"""

Created on Fri Dec 11 13:48:39 2020


@author: Eric Bianchi

"""

import os 
from show_results__ import*
from tqdm import tqdm   
import torch
from PIL import Image
import glob
import numpy as np

# 初始化模型列表
models = []

# 设置路径
source_image_dir = os.path.join('D:', 'BS', 'FSJC', 'Dateset-1')
destination_mask = os.path.join('D:', 'BS', 'FSJC', 'results')  # 修改为正确的路径格式
os.makedirs(destination_mask, exist_ok=True)

# 加载指定模型
model_path = r'D:\BS\FSJC\DeepV3\Corrosion Condition State Classification - Trained Model\l1_loss\weights_27.pt'
print(f"加载模型: {os.path.basename(model_path)}")
model = torch.load(model_path, map_location=torch.device('cuda'))
model.eval()
models.append(model)

# 获取所有图片文件
image_files = []
search_path = r'D:\BS\FSJC\Dateset-1'  # 使用原始字符串处理路径
found_files = glob.glob(os.path.join(search_path, '*.jpg'))
print(f"在 {search_path} 中搜索")
print(f"找到 JPG 文件: {len(found_files)} 个")
print("找到的文件:")
for f in found_files:
    print(f"  - {os.path.basename(f)}")
image_files.extend(found_files)

print(f"\n总共找到 {len(image_files)} 个图片文件")
if len(image_files) == 0:
    print("请检查文件路径是否正确")
    print(f"当前搜索路径: {os.path.abspath(search_path)}")
    print(f"在目录 {source_image_dir} 中没有找到图片文件")
    print("请确认目录路径是否正确，以及图片格式是否支持")


# 处理所有图像，叠加多个模型的结果
for image_path in tqdm(image_files):
    image_name = os.path.basename(image_path)
    print(f"\n处理图片: {image_name}")
    
    # 创建合并结果的目录
    combined_result_dir = os.path.join('D:', 'BS', 'FSJC', 'results', 'combined_results')
    os.makedirs(combined_result_dir, exist_ok=True)
    overlay_dir = os.path.join(combined_result_dir, 'overlays')
    os.makedirs(overlay_dir, exist_ok=True)
    
    print(f"保存结果到: {combined_result_dir}")
    print(f"保存叠加结果到: {overlay_dir}")
    
    # 对每张图像应用所有模型并叠加结果
    for current_model in models:
        # 所有模型的结果都保存在同一个目录下
        result = generate_images(current_model, image_path, image_name, combined_result_dir, overlay_dir)
        print(f"处理完成: {image_name}")

# 处理完成后自动打开叠加结果文件夹
import subprocess
subprocess.Popen(f'explorer "{overlay_dir}"')
print("\n处理完成，已打开叠加结果文件夹")

# 删除以下重复的代码块
# model_result_dir = os.path.join('D:', 'BS', 'FSJC', 'results', f'model_{i}_results')
# os.makedirs(model_result_dir, exist_ok=True)
# overlay_dir = os.path.join(model_result_dir, 'overlays')
# os.makedirs(overlay_dir, exist_ok=True)
# generate_images(model, image_path, image_name, model_result_dir, overlay_dir)

    # 删除重复的处理代码
    # image_path = source_image_dir + image_name
    # generate_images(model, image_path, image_name, destination_mask, './combined_overlays_l1_loss_corrosion_progression/')
    
    

'''

def crop(im, height, width):

    # im = Image.open(infile)

    imgwidth, imgheight = im.size

    rows = np.int(imgheight/height)

    cols = np.int(imgwidth/width)

    im_list = []

    for i in range(rows):

        for j in range(cols):

            # print (i,j)

            box = (j*width, i*height, (j+1)*width, (i+1)*height)

            im_list.append(im.crop(box))
    return im_list



for image_name in tqdm(os.listdir(source_image_dir)):
    print(image_name)

    image_path = source_image_dir + image_name

    image_name, ext =  image_name.split('.')  

    im = Image.open(image_path)

    imgwidth, imgheight = im.size


    height = np.int(imgheight/3)

    width = np.int(imgwidth/3)

    start_num = 0

    imList = crop(im, height, width)

    i = 0 

    for image in imList:

        image.save(source_image_dir + image_name + '_mini_' +str(i)+'.'+ext)

        i = i + 1
'''