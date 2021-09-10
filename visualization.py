'''
General purpose visualization function
Author: Yichen Li
'''
import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

from pylab import rcParams


rcParams['figure.figsize'] = 8,8
rcParams['figure.dpi'] = 150


def pick_scatter_plot():
    x, y, c, s = rand(4, 100)

    def onpick3(event):
        ind = event.ind
        print('onpick3 scatter:', ind, x[ind], y[ind])

    fig, ax = plt.subplots()
    ax.scatter(x, y, 100*s, c, picker=True)
    fig.canvas.mpl_connect('pick_event', onpick3)

def get_pts_color(points):
    num_pts = points.shape[1]

    colors = points[:,2:]
    colors = colors.tolist()

    return colors

def get_displace_color(colors, eps=0.00001):
    color_max = colors.max()
    color_min = colors.min()

    colors /= (color_max-color_min)
    colors = (1.0 - np.clip(colors, a_min=0.0, a_max=1.0))
    return colors


def get_uv_color(points, eps=0.00001):
    num_pts = points.shape[1]

    x_min, y_min = points.min(0)
    x_max, y_max = points.max(0)

    x_min, x_max, y_min, y_max = x_min-eps, x_max+eps, y_min-eps, y_max+eps
    colors = []
    xs_orig, ys_orig = points[:,0], points[:,1]
    xs = xs_orig - x_min
    xs = xs / (x_max-x_min)
    xs = np.clip(xs, a_min=0.0, a_max=1.0)
    ys = ys_orig - y_min
    ys = ys / (y_max-y_min)
    ys = np.clip(ys, a_min=0.0, a_max=1.0)

    colors = np.array(list(zip(list(xs), list(ys), list(np.zeros(xs.shape)))))

    return colors


def get_matching_color(points, idx1, idx2, eps=0.00001):
    colors1 = get_uv_color(points)

    colors2 = colors1[idx2]

    return (colors1, colors2)

def visualize(input_batch, input_patch, pred1, pred2, pairing= None, displace=None, batch_idx=0, all_frame=False, save_dir='./'):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir+'/save_picker/', exist_ok=True)


    with torch.no_grad():
        all_img1 = input_batch['img1'].detach().cpu().numpy()#.squeeze(0)
        all_img2 = input_batch['img2'].detach().cpu().numpy()#.squeeze(0)

        all_cropped_pts1 = input_batch['pts1'].detach().cpu().numpy()#.squeeze(0)
        all_cropped_pts2 = input_batch['pts2'].detach().cpu().numpy()#.squeeze(0)
        
        all_pred1 = pred1.clone().detach().cpu().numpy()#.squeeze(0)
        all_pred2 = pred2.clone().detach().cpu().numpy()#.squeeze(0)

        all_patch_1 = input_patch.clone().detach().cpu().numpy()
        all_patch_2 = input_patch.clone().detach().cpu().numpy()

        pairing1_1, pairing1_2 =  pairing['pairing1']
        pairing2_1, pairing2_2 =  pairing['pairing2']
        all_pairing1_1 = pairing1_1.clone().detach().cpu().numpy()
        all_pairing1_2 = pairing1_2.clone().detach().cpu().numpy()
        all_pairing2_1 = pairing2_1.clone().detach().cpu().numpy()
        all_pairing2_2 = pairing2_2.clone().detach().cpu().numpy()
        
        if all_frame: selected_inds = np.arange(pred1.shape[0]-1).tolist()
        elif batch_idx == 0: selected_inds = [0]
        else: selected_inds = [random.randint(0, pred1.shape[0]-1)]
        for selected_batch_ind in selected_inds:
            fig = plt.figure(figsize=(40,8))

            img1 = all_img1[selected_batch_ind]
            img2 = all_img2[selected_batch_ind]

            cropped_pts1 = all_cropped_pts1[selected_batch_ind]
            cropped_pts2 = all_cropped_pts2[selected_batch_ind]

            pred1 = all_pred1[selected_batch_ind]
            pred2 = all_pred2[selected_batch_ind]

            patch_1 = all_patch_1[selected_batch_ind].transpose()
            patch_2 = all_patch_2[selected_batch_ind].transpose()


            input_color1 = get_pts_color(cropped_pts1)
            input_color2 = get_pts_color(cropped_pts2)

            pred_color1 = get_pts_color(pred1)
            pred_color2 = get_pts_color(pred2)

            uv_color1 = get_uv_color(patch_1)
            uv_color2 = get_uv_color(patch_2)


            if pairing is not None:
                pairing1_1 = all_pairing1_1[selected_batch_ind]
                pairing1_2 = all_pairing1_2[selected_batch_ind]
                pairing2_1 = all_pairing2_1[selected_batch_ind]
                pairing2_2 = all_pairing2_2[selected_batch_ind]

                uv_color1_1, pairing_color1_1 = get_matching_color(pred1[:,:2], pairing1_2, pairing1_1)
                uv_color1_2, pairing_color1_2 = get_matching_color(cropped_pts1[:,:2], pairing1_1, pairing1_2)
                uv_color2_1, pairing_color2_1 = get_matching_color(pred2[:,:2], pairing2_2, pairing2_1)
                uv_color2_2, pairing_color2_2 = get_matching_color(cropped_pts2[:,:2], pairing2_1, pairing2_2)

            row_imgs = 9
            row_img_idx = 1
            # plot imgs first
            fig.add_subplot(2, row_imgs, row_img_idx)
            plt.imshow(img1)
            plt.title('Input Image')
            fig.add_subplot(2, row_imgs, row_img_idx+row_imgs)
            plt.imshow(img2)
            
            # plot cropped points
            row_img_idx = 2
            fig.add_subplot(2, row_imgs, 2)
            fig.add_subplot(2, row_imgs, row_img_idx)
            plt.scatter(cropped_pts1[:,0], cropped_pts1[:,1], c=input_color1)
            plt.title('Colored Cropped Points')
            fig.add_subplot(2, row_imgs, row_img_idx+row_imgs)
            plt.scatter(cropped_pts2[:,0], cropped_pts2[:,1], c=input_color2)


            # plot predicted points
            row_img_idx = 3
            fig.add_subplot(2, row_imgs, row_img_idx)
            plt.scatter(pred1[:,0], pred1[:,1], c=pred_color1)
            plt.title('Colored Predicted Points')
            fig.add_subplot(2, row_imgs, row_img_idx+row_imgs)
            plt.scatter(pred2[:,0], pred2[:,1], c=pred_color2)


            # plot colored patch points
            row_img_idx = 4
            fig.add_subplot(2, row_imgs, row_img_idx)
            plt.scatter(patch_1[:,0], patch_1[:,1], c=pred_color1)
            plt.title('Colored Patch Points')
            fig.add_subplot(2, row_imgs, row_img_idx+row_imgs)
            plt.scatter(patch_2[:,0], patch_2[:,1], c=pred_color2)

        # plot predicted points
            row_img_idx = 5
            fig.add_subplot(2, row_imgs, row_img_idx)
            plt.scatter(pred1[:,0], pred1[:,1], c=uv_color1)
            plt.title('Predicted UV colored Patch')
            fig.add_subplot(2, row_imgs, row_img_idx+row_imgs)
            plt.scatter(pred2[:,0], pred2[:,1], c=uv_color2)


            pairing1_2 = pairing1_2[:,np.newaxis]
            pairing2_2 = pairing2_2[:,np.newaxis]
            pairing1_1 = pairing1_1[:,np.newaxis]
            pairing2_1 = pairing2_1[:,np.newaxis]

            pred_save_color1 = np.concatenate([pred1, pairing1_2, np.array(uv_color1)], axis=1)
            pred_save_color2 = np.concatenate([pred2, pairing2_2, np.array(uv_color2)], axis=1)
            np.save(f'{save_dir}/save_picker/pred_{batch_idx}_gt2pred_frame0.npy', pred_save_color1)
            np.save(f'{save_dir}/save_picker/pred_{batch_idx}_gt2pred_frame1.npy', pred_save_color2)


            # plot uv coloed patch 
            row_img_idx = 6
            fig.add_subplot(2, row_imgs, row_img_idx)
            plt.scatter(patch_1[:,0], patch_1[:,1], c=uv_color1)
            plt.title('Input UV colorted Patch')
            fig.add_subplot(2, row_imgs, row_img_idx+row_imgs)
            plt.scatter(patch_2[:,0], patch_2[:,1], c=uv_color2)


            # plot pairing patch 
            if pairing is not None:
                row_img_idx = 7
                fig.add_subplot(2, row_imgs, row_img_idx)
                plt.scatter(cropped_pts1[:,0], cropped_pts1[:,1], c=pairing_color1_1)
                plt.title('gt -> pred / gt')
                fig.add_subplot(2, row_imgs, row_img_idx+row_imgs)
                plt.scatter(cropped_pts2[:,0], cropped_pts2[:,1], c=pairing_color2_1)
            
                target_save_color1 = np.concatenate([cropped_pts1, pairing1_1, np.array(pairing_color1_1)], axis=-1)
                target_save_color2 = np.concatenate([cropped_pts2, pairing2_1, np.array(pairing_color2_1)], axis=-1)
                np.save(f'{save_dir}/save_picker/target_{batch_idx}_gt2pred_frame0.npy', target_save_color1)
                np.save(f'{save_dir}/save_picker/target_{batch_idx}_gt2pred_frame1.npy', target_save_color2)
                

                row_img_idx = 8
                fig.add_subplot(2, row_imgs, row_img_idx)
                plt.scatter(cropped_pts1[:,0], cropped_pts1[:,1],  c=uv_color1_2)
                plt.title('pred -> gt / gt')
                fig.add_subplot(2, row_imgs, row_img_idx+row_imgs)
                plt.scatter(cropped_pts2[:,0], cropped_pts2[:,1], c=uv_color2_2)
                
                target_save_color1 = np.concatenate([cropped_pts1, pairing1_1, np.array(uv_color1_2)], axis=-1)
                target_save_color2 = np.concatenate([cropped_pts2, pairing2_1, np.array(uv_color2_2)], axis=-1)
                np.save(f'{save_dir}/save_picker/target_{batch_idx}_pred2gt_frame0.npy', target_save_color1)
                np.save(f'{save_dir}/save_picker/target_{batch_idx}_pred2gt_frame1.npy', target_save_color2)

                row_img_idx = 9
                fig.add_subplot(2, row_imgs, row_img_idx)
                plt.scatter(pred1[:,0], pred1[:,1], c=pairing_color1_2)
                plt.title('pred -> gt / pred')
                fig.add_subplot(2, row_imgs, row_img_idx+row_imgs)
                plt.scatter(pred2[:,0], pred2[:,1], c=pairing_color2_2)

                pred_save_color1 = np.concatenate([pred1, pairing1_2, np.array(pairing_color1_2)], axis=-1)
                pred_save_color2 = np.concatenate([pred2, pairing2_2, np.array(pairing_color2_2)], axis=-1)
                np.save(f'{save_dir}/save_picker/pred_{batch_idx}_pred2gt_frame0.npy', pred_save_color1)
                np.save(f'{save_dir}/save_picker/pred_{batch_idx}_pred2gt_frame1.npy', pred_save_color2)

            # plot pairing patch 
            if displace is not None:
                row_img_idx = 8
                fig.add_subplot(2, row_imgs, row_img_idx)
                plt.scatter(pred1[:,0], pred1[:,1], c=displace_color1)
                plt.title('Colored Points Displacement Magnitude')
                fig.add_subplot(2, row_imgs, row_img_idx+row_imgs)
                plt.scatter(pred2[:,0], pred2[:,1], c=displace_color2)


            plt.suptitle(f'iteration {batch_idx}')

            plt.savefig(f'{save_dir}/pred_batch{batch_idx}_{selected_batch_ind}.png')
            plt.close()



        





    










            # if displace is not None:
            #     displace1 =  displace['displace1'][selected_batch_ind].clone().detach().cpu().numpy()
            #     displace2 =  displace['displace2'][selected_batch_ind].clone().detach().cpu().numpy()
            #     displace_color1 = get_displace_color(displace1)
            #     displace_color2 = get_displace_color(displace2)  