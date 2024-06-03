import os

# set the device for training
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import torch
import torch.nn.functional as F
import sys
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from Code.lib.model import Net
from Code.utils.data import get_loader, test_dataset
from Code.utils.utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from Code.utils.options import opt
import torch.nn as nn
from smooth_loss import get_saliency_smoothness

cudnn.benchmark = True

# build the model
model = Net(32, 50)
if (opt.load is not None):
    model.load_state_dict(torch.load(opt.load))
    print('load model from ', opt.load)

model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

# set the path
train_image_root = opt.rgb_label_root
train_gt_root = opt.gt_label_root
train_depth_root = opt.depth_label_root

val_image_root = opt.val_rgb_root
val_gt_root = opt.val_gt_root
val_depth_root = opt.val_depth_root
save_path = opt.save_path

if not os.path.exists(save_path):
    os.makedirs(save_path)

# load data
print('load data...')
train_loader = get_loader(train_image_root, train_gt_root, train_depth_root, batchsize=opt.batchsize,
                          trainsize=opt.trainsize)
test_loader = test_dataset(val_image_root, val_gt_root, val_depth_root, opt.trainsize)
total_step = len(train_loader)

logging.basicConfig(filename=save_path + 'log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("BBSNet_unif-Train")
logging.info("Config")
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load, save_path,
        opt.decay_epoch))

# set loss function
CE = torch.nn.BCEWithLogitsLoss()

step = 0
writer = SummaryWriter(save_path + 'summary')
best_mae = 1
best_epoch = 0

print(len(train_loader))


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1

        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score


dice = SoftDiceLoss()


def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_ids = [0, 1]
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model, device_ids)
    if torch.cuda.is_available():
        model = model.to(device)

    try:
        for i, (images, gts, depths) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.to(device)
            gts = gts.to(device)
            depths = depths.to(device)

            ##
            out1_fusion, out2_fusion, out3_fusion, out4_fusion = model(images, depths)
            sml = get_saliency_smoothness(torch.sigmoid(out1_fusion), gts)
            loss1_fusion = F.binary_cross_entropy_with_logits(out1_fusion, gts)
            loss2_fusion = F.binary_cross_entropy_with_logits(out2_fusion, gts)
            loss3_fusion = F.binary_cross_entropy_with_logits(out3_fusion, gts)
            loss4_fusion = F.binary_cross_entropy_with_logits(out4_fusion, gts)
            dice_loss = dice(out1_fusion, gts)

            loss_seg = loss1_fusion + loss2_fusion * 0.8 + loss3_fusion * 0.6 + loss4_fusion * 0.5 + sml + dice_loss

            loss = loss_seg
            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.data
            if i % 50 == 0 or i == total_step or i == 1:
                print(
                    '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss: {:.4f} loss2_fusion: {:.4f} loss3_fusion: {:.4f} loss4_fusion: {:.4f} sml: {:.4f} loss1_fusion: {:0.4f} dice: {:0.4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data, loss2_fusion.data,
                           loss3_fusion.data, loss4_fusion.data, sml.data, loss1_fusion.data, dice_loss.data))
                logging.info(
                    '#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss: {:.4f} loss2_fusion: {:.4f} loss3_fusion: {:.4f} loss4_fusion: {:.4f} sml: {:.4f} loss1_fusion: {:0.4f} dice: {:0.4f}'.
                    format(epoch, opt.epoch, i, total_step, loss.data, loss2_fusion.data, loss3_fusion.data,
                           loss4_fusion.data, sml.data, loss1_fusion.data, dice_loss.data))

        loss_all /= epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)

        if (epoch) % 5 == 0 and epoch >= 20:
            torch.save(model.module.state_dict(), save_path + 'HyperNet_epoch_{}.pth'.format(epoch))

    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.module.state_dict(), save_path + 'HyperNet_epoch_{}.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise


# test function
def val(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, depth, name, img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            depth = depth.cuda()
            out1_fusion, _, _, _ = model(image, depth)
            res = out1_fusion
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'SPNet_epoch_best.pth')
                print('best epoch:{}'.format(epoch))

        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    print("Start train...")

    for epoch in range(1, opt.epoch):

        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        # train
        train(train_loader, model, optimizer, epoch, save_path)

        # eval
        if epoch >= 20:
            val(test_loader, model, epoch, save_path)
