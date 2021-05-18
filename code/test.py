# coding=utf-8
import argparse
import os
import urllib.request
import numpy as np
import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.autograd import Variable
from torchvision.utils import make_grid
import torch.nn.functional as F
from utils import *
import os
import torchvision.transforms as transforms
import cv2
import time
from PIL import Image
from net import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    start_time = time.time()
    psnrs = []
    ssims = []
    file_names = []
    cuda = 1
    cudnn.benchmark = True
    print("==========> Setting GPU")

    print("==========> Building model")
    
    model = final_Net()
    checkpoint = torch.load('./best_psnr.pth')
    model.load_state_dict(checkpoint["state_dict"])
    model = nn.DataParallel(model, device_ids=[i for i in range(1)]).cuda()
    
    #===== Load input image =====
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
        ]
    )
    transform_gt = transforms.Compose([transforms.ToTensor()])
    model.eval()
    file_dir = "./images"
    for root, dirs, files in os.walk(file_dir):  
        for i in range(len(files)):
            #print(files) #
            imagePath = file_dir+'/'+files[i]
            #print(imagePath)
            frame = Image.open(imagePath)#2
            imgIn = transform(frame).unsqueeze_(0)
            #===== Test procedures =====
            varIn = Variable(imgIn) 
            with torch.no_grad():
                output1,output = model(varIn)
            output = torch.clamp(output, 0., 1.)
            computer_psnr = 0
            if computer_psnr:
                label_imagePath = './clear'+'/'+files[i].split('_')[0] + '.png'
                #print(imagePath)
                gt_img = Image.open(label_imagePath)
                label = transform_gt(gt_img).unsqueeze_(0)
                label = label.cuda()
                psnrs.extend(to_psnr(output, label))
            prediction = output.data.cpu().numpy().squeeze().transpose((1,2,0))
            prediction = (prediction*255.0).astype("uint8")
            im = Image.fromarray(prediction)
            save_path = "./results"
            if  not os.path.exists(save_path):
                os.makedirs(save_path)
            im.save(save_path+"/"+files[i])
            file_names.append(files[i])
    end_time = time.time() - start_time
    print(end_time)
    if computer_psnr:
        psnr_mean = sum(psnrs) / len(psnrs)
        print(psnr_mean)
        import pandas as pd 
        data = {"files":file_names,"psnr":psnrs}
        test=pd.DataFrame(data,columns = ['files','psnr'])
        test.to_csv('results'+'.csv',index=False)

if __name__ == "__main__":
    os.system('clear')
    main()
