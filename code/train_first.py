# coding=utf-8
import argparse
import os
import urllib.request
import numpy as np
import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, CenterCrop, RandomCrop
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
import torch.nn.functional as F
from net import *
from utils import *
import loss
from train_data import TrainData
from val_data import ValData
from torchvision.models import vgg16
from perceptual import LossNetwork


# Training settings
parser = argparse.ArgumentParser(description="PyTorch DeepDehazing")
parser.add_argument("--tag", type=str, help="tag for this training")
parser.add_argument("--train", default="../data/train/", type=str,
                    help="path to load train datasets(default: none)")
parser.add_argument("--test", default="../data/test/", type=str,
                    help="path to load test datasets(default: none)")
parser.add_argument("--batchSize", type=int, default=4, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=100, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=1200, help="step to test the model performance. Default=2000")
parser.add_argument("--cuda", action="store_true",default=1,help="Use cuda?")
parser.add_argument("--gpus", type=int, default=1, help="nums of gpu to use")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=8, help="Number of threads for data loader to use, Default: 4")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")

def adjust_learning_rate_second(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (0.5 ** (epoch //5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        print('Learning rate sets to {}.'.format(param_group['lr']))

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def main():
    global opt, name, logger, model, criterion_L1,criterion_mse,model_second,best_psnr,loss_network
    global edge_loss
    opt = parser.parse_args()
    print(opt)
    import random
    
    opt.best_psnr = 0
    # Tag_ResidualBlocks_BatchSize
    name = "%s_%d" % (opt.tag, opt.batchSize)
    logger = SummaryWriter("runs/" + name)
    
    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    opt.seed_python = random.randint(1, 10000)
    random.seed(opt.seed_python)
    print("Random Seed_python: ", opt.seed_python)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True

    print("==========> Loading datasets")    
    train_data_dir = opt.train
    val_data_dir = opt.test
    # --- Load training data and validation/test data --- #
    training_data_loader = DataLoader(TrainData([240, 240], train_data_dir), batch_size=opt.batchSize, shuffle=True, num_workers=12)
    indoor_test_loader = DataLoader(ValData(val_data_dir), batch_size=1, shuffle=False, num_workers=12)

    print("==========> Building model")
    model = first_Net()
    criterion_mse = nn.MSELoss(size_average=True)
    criterion_L1 = nn.L1Loss(size_average=True)

    
    print(model)
    if opt.resume:
	    if os.path.isfile(opt.resume):
	        print("=> loading checkpoint '{}'".format(opt.resume))
	        checkpoint = torch.load(opt.resume)
	        opt.start_epoch = checkpoint["epoch"]+1 
	        model.load_state_dict(checkpoint["state_dict"])
	    else:
	        print("=> no checkpoint found at '{}'".format(opt.resume))
    
    
    # --- Set the GPU --- #
    print("==========> Setting GPU")
    if cuda:
        model = nn.DataParallel(model, device_ids=[i for i in range(opt.gpus)]).cuda()
        criterion_L1 = criterion_L1.cuda()
        criterion_mse = criterion_mse.cuda()

        
# --- Calculate all trainable parameters in network --- #
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))
    print("==========> Setting Optimizer")
    # --- Build optimizer --- #
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    print("==========> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        adjust_learning_rate_second(optimizer, epoch-1)
        train(training_data_loader, indoor_test_loader,optimizer, epoch)
        test(indoor_test_loader, epoch)

def train(training_data_loader, indoor_test_loader,optimizer, epoch):
    loss_mse = []
    device_ids = [Id for Id in range(torch.cuda.device_count())]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vgg_model = vgg16(pretrained=True).features[:16]
    vgg_model = vgg_model.to(device)
    for param in vgg_model.parameters():
        param.requires_grad = False
    loss_network = LossNetwork(vgg_model)
    loss_network.eval()
    
    print("epoch =", epoch, "lr =", optimizer.param_groups[0]["lr"])
    for iteration, batch in enumerate(training_data_loader, 1):
        model.train()
        model.zero_grad()
        optimizer.zero_grad()

        steps = len(training_data_loader) * (epoch-1) + iteration

        data, label = \
            Variable(batch[0]), \
            Variable(batch[1], requires_grad=False)
        if opt.cuda:
            data = data.cuda()
            label = label.cuda()
        else:
            data = data.cpu()
            label = label.cpu()
        output = model(data)
        mse_loss = criterion_mse(output, label)
        loss = criterion_mse(output, label)
      #loss = criterion_mse(output, label) + 0.04*loss_network(output, label)
        
        
        loss_final = loss
        loss_mse.append(mse_loss.data)
        loss_final.backward()
        optimizer.step()
        if iteration % 200 == 0:
            loss_mse_mean = sum(loss_mse) / len(loss_mse)
            print("===> Epoch[{}]({}/{}): Loss: {:.6f}".format(epoch, iteration, len(training_data_loader),
                                                             loss_mse_mean))
            logger.add_scalar('mse_loss', loss_mse_mean, steps)
            loss_mse = []
        if iteration % opt.step == 0:
            data_temp = make_grid(data.data)
            label_temp = make_grid(label.data)
            output_temp = make_grid(output.data)

            logger.add_image('data_temp', data_temp, steps)
            logger.add_image('label_temp', label_temp, steps)
            logger.add_image('output_temp', output_temp, steps)


def test(test_data_loader, epoch):
    psnrs = []
    ssims = []
    for iteration, batch in enumerate(test_data_loader, 1):
        model.eval()
        with torch.no_grad():
            data, label = \
            Variable(batch[0]), \
            Variable(batch[1])

        if opt.cuda:
            data = data.cuda()
            label = label.cuda()
        else:
            data = data.cpu()
            label = label.cpu()

        with torch.no_grad():
            output = model(data)
        
        output = torch.clamp(output, 0., 1.)
        # --- Calculate the average PSNR --- #
        psnrs.extend(to_psnr(output, label))
        # --- Calculate the average SSIM --- #
        ssims.extend(to_ssim_skimage(output, label))

    psnr_mean = sum(psnrs) / len(psnrs)
    ssim_mean = sum(ssims) / len(ssims)
    if opt.best_psnr < psnr_mean:
        opt.best_psnr = psnr_mean
        save_checkpoint(model, epoch, name)
    print("test  epoch %d psnr: %f ssim: %f" % (epoch, psnr_mean,ssim_mean))
    print("pyotrch_seed %d python_seed %d best_psnr %f" % (opt.seed, opt.seed_python,opt.best_psnr))
    logger.add_scalar('psnr', psnr_mean, epoch)
    logger.add_scalar('ssim', ssim_mean, epoch)

if __name__ == "__main__":
    os.system('clear')
    main()