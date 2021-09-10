'''
visualization function for identity net only
Author: Yichen Li
'''
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from pylab import rcParams


rcParams['figure.figsize'] = 8,8
rcParams['figure.dpi'] = 150

def get_pts_color(points):
    num_pts = points.shape[1]

    colors = points[:,2:]
    colors = colors.tolist()




    return colors


def get_uv_color(points, eps=0.001):
    num_pts = points.shape[1]

    x_min, y_min = points.min(0)
    x_max, y_max = points.max(0)

    x_min, x_max, y_min, y_max = x_min-eps, x_max+eps, y_min-eps, y_max+eps

    colors = []
    xs, ys = points[:,0], points[:,1]
    xs -= x_min
    xs /= (x_max-x_min)
    xs = np.clip(xs, a_min=0.0, a_max=1.0)
    ys -= y_min
    ys /= (y_max-y_min)
    ys = np.clip(ys, a_min=0.0, a_max=1.0)

    colors = np.array(list(zip(list(xs), list(ys), list(np.zeros(xs.shape)))))

    return colors

def visualize(input_batch, input_patch, pred1, pred2, batch_idx=0, save_dir='./'):
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        fig = plt.figure(figsize=(16,8))

        img1 = input_batch['img1'][0].detach().cpu().numpy()#.squeeze(0)
        img2 = input_batch['img2'][0].detach().cpu().numpy()#.squeeze(0)

        cropped_pts1 = input_batch['pts1'][0].detach().cpu().numpy()#.squeeze(0)
        cropped_pts2 = input_batch['pts2'][0].detach().cpu().numpy()#.squeeze(0)

        input_color1 = get_pts_color(cropped_pts1)
        input_color2 = get_pts_color(cropped_pts2)

        pred1 = pred1[0].clone().detach().cpu().numpy()#.squeeze(0)
        pred2 = pred2[0].clone().detach().cpu().numpy()#.squeeze(0)

        pred_color1 = get_pts_color(pred1)
        pred_color2 = get_pts_color(pred2)

        patch_1 = input_patch[0].clone().detach().cpu().numpy().transpose()
        patch_2 = input_patch[0].clone().detach().cpu().numpy().transpose()


        # plot imgs first
        fig.add_subplot(2, 3, 1)
        plt.imshow(img1)
        plt.title('Input Image')
        fig.add_subplot(2, 3, 4)
        plt.imshow(img2)

        # plot cropped points
        fig.add_subplot(2, 3, 2)
        plt.scatter(cropped_pts1[:,0], cropped_pts1[:,1], c=input_color1)
        plt.title('Colored Cropped Points')
        fig.add_subplot(2, 3, 5)
        plt.scatter(cropped_pts2[:,0], cropped_pts2[:,1], c=input_color2)


        # plot predicted points
        fig.add_subplot(2, 3, 3)
        plt.scatter(cropped_pts1[:,0], cropped_pts1[:,1], c=pred_color1)
        plt.title('Colored Predicted Points')
        fig.add_subplot(2, 3, 6)
        plt.scatter(cropped_pts1[:,0], cropped_pts1[:,1], c=pred_color2)


        plt.suptitle(f'iteration {batch_idx}')

        plt.savefig(f'{save_dir}/pred_batch{batch_idx}.png')
        plt.close()



        





    









