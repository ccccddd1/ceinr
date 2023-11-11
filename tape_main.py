import math

import torch
import torch.nn as nn
import sys
sys.path.append('core')
import scipy.io as sio
import numpy as np
import os
import random
from argparse import ArgumentParser
from PositionEncoding import  positionencoding2D,positionencoding3D
from sklearn.cluster import KMeans
from utils import getresponse,shift_back_3d
from Pyranid_net import ConvFcn


class ModelParams(nn.Module):
    def fill_args(self, **kwargs):
        for key, item in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, item)

    def __init__(self,**kwargs):
        self.L=32
        self.out_dim=28
        self.size=256
        self.fill_args(**kwargs)


class ProgressiveEncoderModel(nn.Module):

    def __init__(self,params:ModelParams,args):
        super(ProgressiveEncoderModel, self).__init__()
        self.res=params.size
        self.L=params.L
        self.output_channels=params.out_dim
        self.mlp = ConvFcn(params.L, self.output_channels, args.activation)


    def forward(self,coords,attachedvec):
        base_coding=coords.clone()
        out = self.mlp(base_coding, attachedvec)
        return out


def _as_floats(im1, im2):
    float_type = np.result_type(im1.dtype, im2.dtype, np.float32)
    im1 = np.asarray(im1, dtype=float_type)
    im2 = np.asarray(im2, dtype=float_type)
    return im1, im2


def compare_mse(im1, im2):
    im1, im2 = _as_floats(im1, im2)
    return np.mean(np.square(im1 - im2), dtype=np.float64)


def compare_psnr(im_true, im_test, data_range=1.0):
    im_true, im_test = _as_floats(im_true, im_test)
    err = compare_mse(im_true, im_test)
    return 10 * np.log10((data_range ** 2) / err)


def predictwholeimage(model, coords,attachedvec):

    model.eval()

    pred_specimg = model(coords, attachedvec)
    return pred_specimg


def main(model_params,batch_response,batch_mask,args):
    start_epoch = args.start_epoch
    end_epoch=args.end_epoch
    learning_rate = args.learning_rate
    m_psnr_cnn = 0
    avg_psnr_cnn=0
    j = 0
    num_cluster = 0
    avg_loss = 0
    loss_p=[]

    batch_response=batch_response.to(device)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    model =ProgressiveEncoderModel(model_params,args).to(device)
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # pos = np.transpose(positionencoding3D(args.size, args.size, 28, args.L, args.L), (3, 2, 0, 1))
    pos = np.transpose(positionencoding2D(args.size, args.size, args.L, 'sin_cos',0), (2, 0, 1))
    pos = torch.from_numpy(pos).cuda().unsqueeze(0)
    # p=pos.view(1,-1, 28, 256*256)
    attachedvec=shift_back_3d(batch_response.repeat(28,1,1).unsqueeze(0))
    # attachedvec = torch.randn(1,28,256,256).cuda()

    for epoch_i in range(start_epoch+1, end_epoch):
        if epoch_i>5000 and epoch_i %2000 == 1999:
            pred_specimg= predictwholeimage(model, pos,attachedvec)
            pred_specimg=pred_specimg.detach().squeeze().cpu().double().numpy()
            # num_cluster=20
            num_cluster = np.where(random.random()<0.9, math.floor((j)/3000) + 2, 1)

            kmeans = KMeans(n_clusters=num_cluster, random_state=random.randint(0, 100)).fit(np.transpose(pred_specimg.reshape(28,256*256),(1,0)))
            segments_quick = np.reshape(kmeans.labels_, (args.size, args.size))
            segments_quick = np.repeat(segments_quick.reshape((1, args.size, args.size)), 28, axis=0)

            attachedvec1 = np.zeros((28, args.size, args.size))
            for k in range(num_cluster):
                selected = (segments_quick == k)
                spec_cluster = kmeans.cluster_centers_[k, :]
                attachedvec1 = np.where(selected, np.repeat(np.repeat(spec_cluster.reshape(28, 1, 1), 256, 1), 256, 2),
                                       attachedvec1)
            attachedvec = torch.from_numpy(attachedvec1).unsqueeze(0).float().cuda()

        model.train()
        optimizer.zero_grad()
        spec_pred = model(pos,attachedvec)

        pred_response = getresponse(spec_pred, batch_mask,2)
        loss_mlp = nn.SmoothL1Loss(reduction='none')(pred_response, batch_response)
        loss_all = loss_mlp.mean()
        loss_all.backward(retain_graph=True)
        optimizer.step()

        if args.experiment=='simulation':
            m_psnr_cnn = compare_psnr(data.squeeze(0).permute(1,2,0).cpu().detach().numpy(), spec_pred.squeeze(0).permute(1,2,0).cpu().detach().numpy())

        avg_loss += loss_all.detach().item()

        output_data = "[%02d/%02d/%02d] AVG Loss: %.7f, PSNR: %.7f, loss_mlp: %.7f,num_cluster: %d, l_rate: %.8f" \
                      % (j, epoch_i, end_epoch, avg_loss/(j+1),  m_psnr_cnn, loss_all.item(), num_cluster,optimizer.state_dict()['param_groups'][0]['lr'])
        if j % 10 == 0:
            print(output_data)
        if epoch_i >=20000:
        #     if loss_all.item() < min_loss:
        #         min_loss = loss_all.item()
            if avg_psnr_cnn<m_psnr_cnn:
                avg_psnr_cnn=m_psnr_cnn
                sio.savemat('result/scene{}_{}_{}_rec_fix20.mat'.format(args.img_index, epoch_i, j),
                                {'pred_specimg': spec_pred.detach().cpu().numpy()})
        j = j + 1


if __name__ == '__main__':
    parser = ArgumentParser(description='PINR-Net')

    parser.add_argument('--path', help="simulation for evaluation")

    parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
    parser.add_argument('--end_epoch', type=int, default=10000000, help='epoch number of end training')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')

    parser.add_argument('--experiment', type=str, default='simulation', help='real or simulation')
    parser.add_argument('--img_index',type=str,default='06',help='training img index')
    parser.add_argument('--data_dir', type=str, default='kaist', help='training data directory')
    parser.add_argument('--L', type=int, default=32, help='position encoding')
    parser.add_argument('--eta', type=float, default=1.0, help='weight')
    parser.add_argument('--batchsize', type=int, default=1, help='batchsize')
    parser.add_argument('--size',type=int,default=660,help='size of input img')
    parser.add_argument('--shift', type=int, default=2, help='shift step')
    parser.add_argument('--seed', type=int, default=3, help='random seed')
    parser.add_argument('--activation', type=str, default='GELU', help='activation function')
    parser.add_argument('--cluster', type=str,default='kmeans',help='cluster algorithm or no cluster')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.experiment == 'real':
        datapath = os.path.join('real_dataset', 'scene{}.mat'.format(args.img_index))
        mask = sio.loadmat('mask.mat')['mask']
        mask_3d = np.tile(mask[:, :, np.newaxis], (1, 1, 28))
        mask_3d_shift = np.zeros((args.size, args.size + (28 - 1) * args.shift, 28))
        mask_3d_shift[:, 0:args.size, :] = mask_3d
        for t in range(28):
            mask_3d_shift[:, :, t] = np.roll(mask_3d_shift[:, :, t], args.shift * t, axis=1)
        mask_3d_shift = torch.from_numpy(np.transpose(mask_3d_shift, (2, 0, 1))).unsqueeze(0)
        batch_mask = mask_3d_shift.float().to(device)
        b_response = sio.loadmat(datapath)['meas_real']
        b_response[b_response < 0] = 0.0
        b_response[b_response > 1] = 1.0
        b_response = b_response / b_response.max() * 0.8
        b_response = torch.tensor(b_response).unsqueeze(0)
    elif args.experiment == 'simulation':
        respond_fun = getresponse
        datapath = os.path.join('simulation', args.data_dir, 'scene{}.mat'.format(args.img_index))
        mask = sio.loadmat('mask_3d_shift.mat')['mask_3d_shift']
        mask = np.where(mask > 0.5, np.ones_like(mask), np.zeros_like(mask))
        X_ori = sio.loadmat(datapath)['img']
        data = np.transpose(X_ori, (2, 0, 1))
        data = torch.from_numpy(data).cuda().unsqueeze(0)
        mask = np.transpose(mask, (2, 0, 1))
        batch_mask = torch.from_numpy(mask).unsqueeze(0).float().to(device)  # [1,28,256,310]
        b_response = respond_fun(data, batch_mask,args.shift)  # [1,256,310]
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    args.size=batch_mask.shape[2]
    Phi=batch_mask.squeeze(0).permute(1,2,0).cpu().numpy()
    meas_shif=np.multiply(np.repeat(b_response.permute(1,2,0).cpu().numpy(), Phi.shape[2], axis=2), Phi)
    # sio.savemat('meas_shift.mat', {'pred_specimg': meas_shif})
    model_params=ModelParams(L=args.L,out_dim=28,size=args.size)
    main(model_params,b_response,batch_mask,args)