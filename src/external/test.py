# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import time
from cp_dataset import CPDataset, CPDataLoader
from networks import GMM, UnetGenerator, load_checkpoint

from tensorboardX import SummaryWriter
from visualization import board_add_image, board_add_images, save_images


def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", default="GMM")
    # parser.add_argument("--name", default="TOM")

    parser.add_argument("--gpu_ids", default="")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)

    parser.add_argument("--dataroot", default="data")

    # parser.add_argument("--datamode", default="train")
    parser.add_argument("--datamode", default="test")

    parser.add_argument("--stage", default="GMM")
    # parser.add_argument("--stage", default="TOM")

    # parser.add_argument("--data_list", default="train_pairs.txt")
    parser.add_argument("--data_list", default="test_pairs.txt")
    # parser.add_argument("--data_list", default="test_pairs_same.txt")

    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=5)

    parser.add_argument('--tensorboard_dir', type=str,
                        default='tensorboard', help='save tensorboard infos')

    parser.add_argument('--result_dir', type=str,
                        default='result', help='save result infos')

    parser.add_argument('--checkpoint', type=str, default='checkpoints/GMM/gmm_final.pth', help='model checkpoint for test')
    # parser.add_argument('--checkpoint', type=str, default='checkpoints/TOM/tom_final.pth', help='model checkpoint for test')

    parser.add_argument("--display_count", type=int, default=1)
    parser.add_argument("--shuffle", action='store_true',
                        help='shuffle input data')

    opt = parser.parse_args()
    return opt


def test_gmm(opt, test_loader, model, board):
    device = torch.device('cuda' if opt.gpu_ids != "" and torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    base_name = os.path.basename(opt.checkpoint)
    name = opt.name
    save_dir = os.path.join(opt.result_dir, name, opt.datamode)
    os.makedirs(save_dir, exist_ok=True)
    warp_cloth_dir = os.path.join(save_dir, 'warp-cloth')
    os.makedirs(warp_cloth_dir, exist_ok=True)
    warp_mask_dir = os.path.join(save_dir, 'warp-mask')
    os.makedirs(warp_mask_dir, exist_ok=True)
    result_dir1 = os.path.join(save_dir, 'result_dir')
    os.makedirs(result_dir1, exist_ok=True)
    overlayed_TPS_dir = os.path.join(save_dir, 'overlayed_TPS')
    os.makedirs(overlayed_TPS_dir, exist_ok=True)
    warped_grid_dir = os.path.join(save_dir, 'warped_grid')
    os.makedirs(warped_grid_dir, exist_ok=True)

    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()

        c_names = inputs['c_name']
        im_names = inputs['im_name']
        im = inputs['image'].to(device)
        im_pose = inputs['pose_image'].to(device)
        im_h = inputs['head'].to(device)
        shape = inputs['shape'].to(device)
        agnostic = inputs['agnostic'].to(device)
        c = inputs['cloth'].to(device)
        cm = inputs['cloth_mask'].to(device)
        im_c = inputs['parse_cloth'].to(device)
        im_g = inputs['grid_image'].to(device)
        shape_ori = inputs['shape_ori']  # original body shape without blurring

        grid, theta = model(agnostic, cm)
        warped_cloth = F.grid_sample(c, grid, padding_mode='border')
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
        warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')
        overlay = 0.7 * warped_cloth + 0.3 * im

        visuals = [[im_h, shape, im_pose],
                   [c, warped_cloth, im_c],
                   [warped_grid, (warped_cloth + im) * 0.5, im]]

        # save_images(warped_cloth, c_names, warp_cloth_dir)
        # save_images(warped_mask*2-1, c_names, warp_mask_dir)
        save_images(warped_cloth, im_names, warp_cloth_dir)
        save_images(warped_mask * 2 - 1, im_names, warp_mask_dir)
        save_images(shape_ori.to(device) * 0.2 + warped_cloth * 0.8, im_names, result_dir1)
        save_images(warped_grid, im_names, warped_grid_dir)
        save_images(overlay, im_names, overlayed_TPS_dir)

        if (step + 1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step + 1)
            t = time.time() - iter_start_time
            print(f'step: {step + 1:8d}, time: {t:.3f}', flush=True)


def test_tom(opt, test_loader, model, board):
    device = torch.device('cuda' if opt.gpu_ids != "" and torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    base_name = os.path.basename(opt.checkpoint)
    save_dir = os.path.join(opt.result_dir, opt.name, opt.datamode)
    os.makedirs(save_dir, exist_ok=True)
    try_on_dir = os.path.join(save_dir, 'try-on')
    os.makedirs(try_on_dir, exist_ok=True)
    p_rendered_dir = os.path.join(save_dir, 'p_rendered')
    os.makedirs(p_rendered_dir, exist_ok=True)
    m_composite_dir = os.path.join(save_dir, 'm_composite')
    os.makedirs(m_composite_dir, exist_ok=True)
    im_pose_dir = os.path.join(save_dir, 'im_pose')
    os.makedirs(im_pose_dir, exist_ok=True)
    shape_dir = os.path.join(save_dir, 'shape')
    os.makedirs(shape_dir, exist_ok=True)
    im_h_dir = os.path.join(save_dir, 'im_h')
    os.makedirs(im_h_dir, exist_ok=True)  # for test data

    print('Dataset size: %05d!' % (len(test_loader.dataset)), flush=True)
    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()

        im_names = inputs['im_name']
        device_inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

        im = device_inputs['image']
        im_pose = device_inputs['pose_image']
        im_h = device_inputs['head']
        shape = device_inputs['shape']

        agnostic = device_inputs['agnostic']
        c = device_inputs['cloth']
        cm = device_inputs['cloth_mask']

        # outputs = model(torch.cat([agnostic, c], 1))  # CP-VTON
        outputs = model(torch.cat([agnostic, c, cm], 1))  # CP-VTON+
        p_rendered, m_composite = torch.split(outputs, 3, 1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        p_tryon = c * m_composite + p_rendered * (1 - m_composite)

        visuals = [[im_h, shape, im_pose],
                   [c, 2 * cm - 1, m_composite],
                   [p_rendered, p_tryon, im]]

        save_images(p_tryon, im_names, try_on_dir)
        save_images(im_h, im_names, im_h_dir)
        save_images(shape, im_names, shape_dir)
        save_images(im_pose, im_names, im_pose_dir)
        save_images(m_composite, im_names, m_composite_dir)
        save_images(p_rendered, im_names, p_rendered_dir)  # For test data

        if (step + 1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step + 1)
            t = time.time() - iter_start_time
            print(f'step: {step + 1:8d}, time: {t:.3f}', flush=True)


def main():
    opt = get_opt()
    print(opt)
    print(f"Start to test stage: {opt.stage}, named: {opt.name}!")

    # create dataset
    test_dataset = CPDataset(opt)

    # create dataloader
    test_loader = CPDataLoader(opt, test_dataset)

    # visualization
    os.makedirs(opt.tensorboard_dir, exist_ok=True)
    board = SummaryWriter(logdir=os.path.join(opt.tensorboard_dir, opt.name))

    # create model & test
    if opt.stage == 'GMM':
        model = GMM(opt)
        load_checkpoint(model, opt.checkpoint)
        with torch.no_grad():
            test_gmm(opt, test_loader, model, board)
    elif opt.stage == 'TOM':
        # model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)  # CP-VTON
        model = UnetGenerator(26, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)  # CP-VTON+
        load_checkpoint(model, opt.checkpoint)
        with torch.no_grad():
            test_tom(opt, test_loader, model, board)
    else:
        raise NotImplementedError(f'Model [{opt.stage}] is not implemented')

    print(f'Finished test {opt.stage}, named: {opt.name}!')


if __name__ == "__main__":
    main()
