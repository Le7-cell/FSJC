import torch.optim as optim
from sklearn.metrics import roc_auc_score, f1_score, jaccard_score
from metrics.loss import get_spectrum_score
from model_plus import createDeepLabv3Plus
from metrics.loss import spectrum_loss

import sys
print(sys.version, sys.platform, sys.executable)
from trainer_plus import train_model
import datahandler_plus
import argparse
import os
import torch
import numpy
torch.cuda.empty_cache()

def main():
    """
        Version requirements:
            PyTorch Version:  1.2.0
            Torchvision Version:  0.4.0a0+6b959ee
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-data_directory", help='Specify the dataset directory path')
    parser.add_argument(
        "-exp_directory", help='Specify the experiment directory where metrics and model weights shall be stored.')
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batchsize", default=2, type=int)
    parser.add_argument("--output_stride", default=8, type=int)
    parser.add_argument("--channels", default=1, type=int)  # 确保通道数为1
    parser.add_argument("--class_weights", nargs='+', default=None)
    parser.add_argument("--loss", default='l1')  # 使用L1损失函数
    parser.add_argument("--pretrained", default='')
    parser.add_argument("--epoch_number", default=0, type=int)
    
    args = parser.parse_args()
    
    bpath = args.exp_directory
    print('Export Directory: ' + bpath)
    data_dir = args.data_directory
    print('Data Directory: ' + data_dir)
    epochs = args.epochs
    print('Epochs: ' + str(epochs))
    batchsize = args.batchsize
    print('Batch size: ' + str(batchsize))
    output_stride = args.output_stride
    channels = args.channels
    print('Number of classes: ' + str(channels))
    class_weights = args.class_weights
    print('Class weights: ' + str(class_weights))
    loss_function = args.loss
    print('Loss function: ' + str(loss_function))
    model_path = args.pretrained
    print('loading pre-trained model from saved state: ' + model_path)
    epoch_number = args.epoch_number
    print('starting from epoch number: ' + epoch_number)
    
    if not os.path.exists(bpath): # if it doesn't exist already
        os.makedirs(bpath)  
    
    # Create the deeplabv3 resnet101 model which is pretrained on a subset of COCO train2017, 
    # on the 20 categories that are present in the Pascal VOC dataset.
    if model_path != '':
        try:
            model = torch.load(model_path)
            print('LOADED MODEL')
            model.train()
        except:
            print('model path did not load')
            model = createDeepLabv3Plus(outputchannels=channels, output_stride=output_stride)
    else:
        model = createDeepLabv3Plus(outputchannels=channels, output_stride=output_stride)
    
    
    model.train()
    # Create the experiment directory if not present
    if not os.path.isdir(bpath):
        os.mkdir(bpath)
    
    # Specify the loss function
    if loss_function == 'cross_entropy':
        # 交叉熵损失函数适用于多类别分类，不适用于单通道输出
        if class_weights == None:
            print('class not weighted')
            criterion = torch.nn.CrossEntropyLoss()
        elif class_weights != None and len(class_weights) == channels:
            print('class weighted')
            class_weights =  numpy.array(class_weights).astype(float)
            torch_class_weights = torch.FloatTensor(class_weights).cuda()
            criterion = torch.nn.CrossEntropyLoss(weight=torch_class_weights)
        else:
            print('channels did not allign with class weights or Loss function not found - default applied')
            print('class not weighted')
            criterion = torch.nn.CrossEntropyLoss()
    elif loss_function == 'l1':
        print('spectrum loss, l1')
        #criterion = spectrum_loss()
        #criterion.set_diag_matrix(channels)
        criterion = torch.nn.L1Loss()
    elif loss_function == 'l2':
        criterion = torch.nn.MSELoss()
    elif loss_function == 'bce':  # 添加二元交叉熵损失函数，适用于单通道输出
        print('Binary Cross Entropy')
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        # 如果输出通道为1，默认使用二元交叉熵
        if channels == 1:
            print('Binary Cross Entropy')
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            print('Cross Entropy')
            criterion = torch.nn.CrossEntropyLoss()
    
        
    # Specify the optimizer with a lower learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Specify the evalutation metrics
    metrics = {'f1_score': f1_score, 'jaccard_score': jaccard_score, 'spectrum_score': get_spectrum_score}
    
    # Create the dataloader
    dataloaders = datahandler_plus.get_dataloader_sep_folder(data_dir, batch_size=batchsize)
    trained_model = train_model(model, criterion, dataloaders,
                                optimizer, bpath=bpath, metrics=metrics, l=loss_function, num_epochs=epochs, epoch_number=epoch_number)
    
    # Save the trained model
    # torch.save({'model_state_dict':trained_model.state_dict()},os.path.join(bpath,'weights'))
    torch.save(model, os.path.join(bpath, 'weights.pt'))

if __name__ == '__main__':
    main()