""" Network for making affine transformations

Author: Yichen Li
"""
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from torchvision.datasets import MNIST
import pytorch_lightning as pl

from models.modules import apply_transform, resnet18, resnet50, resnet18_resetlast, PointGenCon, PointGenConAffine, PointGenConColor, distChamfer, FCBlock
from visualization import visualize
import visualization_identitynet

import ChamferDistancePytorch.chamfer2D.dist_chamfer_2D as dist_chamfer_2D
import ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as dist_chamfer_3D
import ChamferDistancePytorch.chamfer5D.dist_chamfer_5D as dist_chamfer_5D



# network that directly regress r, s, t parameters
class AffineNet(pl.LightningModule):

    def __init__(self, args, learning_rate=1e-3, batch_size=32, num_points = 2048, bottleneck_size=64, nb_primitives = 1, output_dim=2):
        super().__init__()
        super(AffineNet, self).__init__()
        self.args = args
        self.learning_rate = learning_rate
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        self.rand_grid = Variable(torch.cuda.FloatTensor(1, 2 ,args.num_points//args.nb_primitives))
        self.rand_grid.data.uniform_(0,1)
        self.rand_grid = self.rand_grid.repeat(args.batch_size, 1, 1)
        # scale = points1[0,:,:2].max(0)[0] - points1[0,:,:2].min(0)[0]
        # remove scaling 
        self.rand_grid = self.rand_grid * torch.tensor([0.15, 0.2]).to(self.rand_grid.device).unsqueeze(0).unsqueeze(-1)

        self.encoder = resnet18_resetlast(pretrained=True, num_classes=bottleneck_size)
        self.t1  = nn.Parameter(torch.zeros(self.args.batch_size, 2, 1, requires_grad=True).float())
        self.t2  = nn.Parameter(torch.zeros(self.args.batch_size, 2, 1, requires_grad=True).float())
        self.s1  = nn.Parameter(torch.ones(self.args.batch_size, 1, 1, requires_grad=True).float())
        self.s2  = nn.Parameter(torch.ones(self.args.batch_size, 1, 1, requires_grad=True).float())
        self.r1  = nn.Parameter(torch.zeros(self.args.batch_size, 1, 1, requires_grad=True).float())
        self.r2  = nn.Parameter(torch.zeros(self.args.batch_size, 1, 1, requires_grad=True).float())
        # self.color_decoder = nn.ModuleList([PointGenConColor(bottleneck_size = 2 + self.bottleneck_size, output_dim=3, activation=self.args.color_activation) for i in range(0,self.nb_primitives)])
        self.color_decoder = nn.ModuleList([FCBlock(in_features=2, out_features=3, num_hidden_layers=3, hidden_features=bottleneck_size,  \
                                    outermost_linear=False, nonlinearity=self.args.color_activation)])
        self.chamfer2D = dist_chamfer_2D.chamfer_2DDist()
        self.chamfer3D = dist_chamfer_3D.chamfer_3DDist()
        self.chamfer5D = dist_chamfer_5D.chamfer_5DDist()
        
        self.vis_counter = 0

    def from_pretrained(self, weight_path):
        return self.load_from_checkpoint(weight_path, strict=False)

    def loss(self, points1, output1, points2, output2, s=1.0, delta=1.0):
        # implement the loss that 
        # new_tensor = F.pad(gt_color1,(0,0,0,output_color_1.shape[1]-gt_color1.shape[1],0,0), 'constant', 0)
        # first = new_tensor.gather(1, idx1_2.unsqueeze(-1).repeat(1,1,3).long())

        gt_shape1 = points1[:,:,:2]
        gt_shape2 = points2[:,:,:2]
        output_shape_1 = output1[:,:,:2]
        output_shape_2 = output2[:,:,:2]
        gt_color1 = points1[:,:,2:]
        gt_color2 = points2[:,:,2:]
        output_color_1 = output1[:,:,2:]
        output_color_2 = output2[:,:,2:]

        batch_loss = {
            'chamfer5d_loss': 0.0, 
            'shape_loss': 0.0,
            'color_loss': 0.0,
            'temporal_loss': 0.0,
            'pairing': None,
            'displace': None
        }

        if self.args.chamfer5d_loss:
            # change the relative scaling of [x,y] and [color] 
            relative_scale = torch.tensor([1.0, 1.0, s, s, s]).view(1, 1, 5).to(points1.device)
            scaled_points1 =  points1 * relative_scale
            scaled_output1 = output1 * relative_scale
            scaled_points2 =  points2 * relative_scale
            scaled_output2 = output2 * relative_scale
            
            dist1_1, dist1_2, idx1_1, idx1_2 = self.chamfer5D(scaled_points1, scaled_output1)
            loss1 = (torch.mean(dist1_1)) + (torch.mean(dist1_2))

            dist2_1, dist2_2, idx2_1, idx2_2 = self.chamfer5D(scaled_points2, scaled_output2)
            loss2 = (torch.mean(dist2_1)) + (torch.mean(dist2_2))

            loss = ( loss1 + loss2 ) * self.args.shape_loss_weight
            batch_loss['chamfer5d_loss'] = loss1.item() + loss2.item() 
            batch_loss['pairing'] = { 'pairing1': (idx1_1, idx1_2), 'pairing2': (idx2_1, idx2_2)}

        else:
            
            dist1_1, dist1_2, idx1_1, idx1_2 = self.chamfer2D(gt_shape1, output_shape_1)
            shape_loss1 = (torch.mean(dist1_1)) + (torch.mean(dist1_2))

            dist2_1, dist2_2, idx2_1, idx2_2 = self.chamfer2D(gt_shape2, output_shape_2)
            shape_loss2 = (torch.mean(dist2_1)) + (torch.mean(dist2_2))

            loss = ( shape_loss1 + shape_loss2 ) * self.args.shape_loss_weight
            batch_loss['shape_loss'] = shape_loss1.item() + shape_loss2.item() 
            batch_loss['pairing'] = { 'pairing1': (idx1_1, idx1_2), 'pairing2': (idx2_1, idx2_2)}


            if not self.args.no_color_loss:
                idx1_2 = torch.clamp(idx1_2, min=0, max=gt_color2.shape[1])
                idx2_2 = torch.clamp(idx2_2, min=0, max=gt_color2.shape[1])

                color_loss1 = torch.pow(gt_color1.gather(1, idx1_2.unsqueeze(-1).repeat(1,1,3).long()) - output_color_1, 2).mean()
                color_loss2 = torch.pow(gt_color2.gather(1, idx2_2.unsqueeze(-1).repeat(1,1,3).long()) - output_color_2, 2).mean()

                loss += ( color_loss1 + color_loss2 ) * self.args.color_loss_weight

                batch_loss['color_loss'] = color_loss1.item() + color_loss2.item()

        if not self.args.no_temporal_loss:
            
            if self.args.chamfer5d_loss:        
                dist_time_1, dist_time_2, idx_time_1, idx_time_2 = self.chamfer5D(output1, output2)
                time_loss = torch.mean(dist_time_1) + torch.mean(dist_time_2)

                loss += time_loss * self.args.temporal_loss_weight
                batch_loss['temporal_loss'] = time_loss.item()

            else:
                dist_time_1, dist_time_2, idx_time_1, idx_time_2 = self.chamfer2D(output_shape_1, output_shape_2)
                shape_time_loss = dist_time_1.mean() + dist_time_2.mean()

                color_time_loss = torch.pow(output_color_2 - output_color_1, 2).mean()
                loss += ( color_time_loss + shape_time_loss )  * self.args.temporal_loss_weight
                batch_loss['temporal_loss'] = color_time_loss.item() + shape_time_loss.item()

        batch_loss['loss'] = loss
        
        return batch_loss
        
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x1, x2, points1, points2 = batch['img1'].cuda(), batch['img2'].cuda(), batch['pts1'].cuda(), batch['pts2'].cuda()
        x1 = x1.permute(0,3,1,2)[:,:3,:,:].contiguous()
        x2 = x2.permute(0,3,1,2)[:,:3,:,:].contiguous()
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)

        outs1 = []
        outs2 = []
        
        for i in range(0,self.nb_primitives):
            # context1 = x1.unsqueeze(2).expand(x1.size(0),x1.size(1), self.rand_grid.size(2)).contiguous()
            
            # deformed_y1 = apply_transform(self.rand_grid, self.t1, torch.relu(self.s1), torch.relu(self.r1))
            # no scaling
            deformed_y1 = apply_transform(self.rand_grid, self.t1, torch.ones_like(self.s1).to(self.s1.device), torch.relu(self.r1))
            # shape_y1 = torch.cat( (self.rand_grid, context1), 1).contiguous()
            # shape_y1 = self.rand_grid
            c1 = self.color_decoder[i](self.rand_grid.transpose(2,1)).transpose(2, 1)
            if self.args.color_activation == "siren": c1 = (c1 + 1.0)/2.0
            outs1.append(torch.cat(( deformed_y1, c1), 1))

            # context2 = x2.unsqueeze(2).expand(x2.size(0),x2.size(1), self.rand_grid.size(2)).contiguous()
            # deformed_y2 = apply_transform(self.rand_grid, self.t2, torch.relu(self.s2), torch.relu(self.r2))
            # no scaling
            deformed_y2 = apply_transform(self.rand_grid, self.t2, torch.ones_like(self.s1).to(self.s1.device), torch.relu(self.r2))
            # shape_y2 = torch.cat( (self.rand_grid, context2), 1).contiguous()
            # shape_y2 = self.rand_grid
            c2 = self.color_decoder[i](self.rand_grid.transpose(2,1)).transpose(2, 1)
            if self.args.color_activation == "siren": c2 = (c2 + 1.0)/2.0
            outs2.append(torch.cat(( deformed_y2, c2), 1))

        outs1 = torch.cat(outs1,2).contiguous().transpose(2,1).contiguous()
        outs2 = torch.cat(outs2,2).contiguous().transpose(2,1).contiguous()


        all_loss = self.loss(points1, outs1, points2, outs2, s=self.args.shape_color_scale)
        loss = all_loss['loss']
        self.log('loss', {'loss': loss.item(), 'shape_loss': all_loss['shape_loss'], 'color_loss': all_loss['color_loss'], 'temporal_loss': all_loss['temporal_loss']})

        if self.vis_counter % self.args.vis_freq == 0:
            visualize(input_batch=batch, input_patch=self.rand_grid, pred1=outs1, pred2=outs2, pairing=all_loss['pairing'], displace=None, batch_idx=self.vis_counter, save_dir=self.args.output_dir)
        
        if self.vis_counter % self.args.save_freq == 0:
            os.makedirs(self.args.output_dir+f'/ckpts/', exist_ok=True)
            torch.save(self.state_dict, self.args.output_dir+f'/ckpts/ckpts{self.vis_counter}.pt')
        self.vis_counter += 1

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def forward_inference(self, x, grid):
        x = self.encoder(x)
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(grid[i]))
            rand_grid = rand_grid.transpose(0,1).contiguous().unsqueeze(0)
            rand_grid = rand_grid.expand(x.size(0),rand_grid.size(1), rand_grid.size(2)).contiguous()
            # print(rand_grid.sizerand_grid())
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def forward_inference_from_latent_space(self, x, grid):
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(grid[i]))
            rand_grid = rand_grid.transpose(0,1).contiguous().unsqueeze(0)
            rand_grid = rand_grid.expand(x.size(0),rand_grid.size(1), rand_grid.size(2)).contiguous()
            # print(rand_grid.sizerand_grid())
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()  

    def forward(self, batch, batch_idx):
        
        return self.training_step(batch, batch_idx)
        
        
# network that uses mlp to predict r, s, t parameters
class AffineMLPNet(pl.LightningModule):

    def __init__(self, args, learning_rate=1e-3, batch_size=32, num_points = 2048, bottleneck_size=64, nb_primitives = 1, output_dim=2):
        super().__init__()
        super(AffineMLPNet, self).__init__()
        self.args = args
        self.learning_rate = learning_rate
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        self.rand_grid = Variable(torch.cuda.FloatTensor(1, 2 ,args.num_points//args.nb_primitives))
        self.rand_grid.data.uniform_(0,1)
        self.rand_grid = self.rand_grid.repeat(args.batch_size, 1, 1)
        self.rand_grid = self.rand_grid * torch.tensor([0.15, 0.2]).to(self.rand_grid.device).unsqueeze(0).unsqueeze(-1)
        self.encoder = resnet18_resetlast(pretrained=True, num_classes=bottleneck_size)
        self.affine_decoder = nn.ModuleList([PointGenConAffine(bottleneck_size = 2 + self.bottleneck_size) for i in range(0,self.nb_primitives)])
        self.color_decoder = nn.ModuleList([FCBlock(in_features=2, out_features=3, num_hidden_layers=3, hidden_features=bottleneck_size,  \
                            outermost_linear=False, nonlinearity=self.args.color_activation)])
        self.chamfer2D = dist_chamfer_2D.chamfer_2DDist()
        self.chamfer3D = dist_chamfer_3D.chamfer_3DDist()
        self.chamfer5D = dist_chamfer_5D.chamfer_5DDist()
        
        self.vis_counter = 0

    def from_pretrained(self, weight_path):
        return self.load_from_checkpoint(weight_path, strict=False)

    def loss(self, points1, output1, points2, output2, s=1.0, delta=1.0):
        gt_shape1 = points1[:,:,:2]
        gt_shape2 = points2[:,:,:2]
        output_shape_1 = output1[:,:,:2]
        output_shape_2 = output2[:,:,:2]
        gt_color1 = points1[:,:,2:]
        gt_color2 = points2[:,:,2:]
        output_color_1 = output1[:,:,2:]
        output_color_2 = output2[:,:,2:]

        batch_loss = {
            'chamfer5d_loss': 0.0, 
            'shape_loss': 0.0,
            'color_loss': 0.0,
            'temporal_loss': 0.0,
            'pairing': None,
            'displace': None
        }

        if self.args.chamfer5d_loss:
            # change the relative scaling of [x,y] and [color] 
            relative_scale = torch.tensor([1.0, 1.0, s, s, s]).view(1, 1, 5).to(points1.device)
            scaled_points1 =  points1 * relative_scale
            scaled_output1 = output1 * relative_scale
            scaled_points2 =  points2 * relative_scale
            scaled_output2 = output2 * relative_scale
            
            dist1_1, dist1_2, idx1_1, idx1_2 = self.chamfer5D(scaled_points1, scaled_output1)
            loss1 = (torch.mean(dist1_1)) + (torch.mean(dist1_2))

            dist2_1, dist2_2, idx2_1, idx2_2 = self.chamfer5D(scaled_points2, scaled_output2)
            loss2 = (torch.mean(dist2_1)) + (torch.mean(dist2_2))

            loss = ( loss1 + loss2 ) * self.args.shape_loss_weight
            batch_loss['chamfer5d_loss'] = loss1.item() + loss2.item() 
            batch_loss['pairing'] = { 'pairing1': (idx1_1, idx1_2), 'pairing2': (idx2_1, idx2_2)}

        else:
            
            dist1_1, dist1_2, idx1_1, idx1_2 = self.chamfer2D(gt_shape1, output_shape_1)
            shape_loss1 = (torch.mean(dist1_1)) + (torch.mean(dist1_2))

            dist2_1, dist2_2, idx2_1, idx2_2 = self.chamfer2D(gt_shape2, output_shape_2)
            shape_loss2 = (torch.mean(dist2_1)) + (torch.mean(dist2_2))

            loss = ( shape_loss1 + shape_loss2 ) * self.args.shape_loss_weight
            batch_loss['shape_loss'] = shape_loss1.item() + shape_loss2.item() 

            if not self.args.no_color_loss:
                idx1_2 = torch.clamp(idx1_2, min=0, max=gt_color2.shape[1])
                idx2_2 = torch.clamp(idx2_2, min=0, max=gt_color2.shape[1])

                color_loss1 = torch.pow(gt_color1.gather(1, idx1_2.unsqueeze(-1).repeat(1,1,3).long()) - output_color_1, 2).mean()
                color_loss2 = torch.pow(gt_color2.gather(1, idx2_2.unsqueeze(-1).repeat(1,1,3).long()) - output_color_2, 2).mean()

                loss += ( color_loss1 + color_loss2 ) * self.args.color_loss_weight

                batch_loss['color_loss'] = color_loss1.item() + color_loss2.item()

        if not self.args.no_temporal_loss:
            
            if self.args.chamfer5d_loss:        
                dist_time_1, dist_time_2, idx_time_1, idx_time_2 = self.chamfer5D(output1, output2)
                time_loss = torch.mean(dist_time_1) + torch.mean(dist_time_2)

                loss += time_loss * self.args.temporal_loss_weight
                batch_loss['temporal_loss'] = time_loss.item()

            else:
                dist_time_1, dist_time_2, idx_time_1, idx_time_2 = self.chamfer2D(output_shape_1, output_shape_2)
                shape_time_loss = dist_time_1.mean() + dist_time_2.mean()

                color_time_loss = torch.pow(output_color_2 - output_color_1, 2).mean()
                loss += ( color_time_loss + shape_time_loss )  * self.args.temporal_loss_weight
                batch_loss['temporal_loss'] = color_time_loss.item() + shape_time_loss.item()

        batch_loss['loss'] = loss
        
        return batch_loss
        
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x1, x2, points1, points2 = batch['img1'].cuda(), batch['img2'].cuda(), batch['pts1'].cuda(), batch['pts2'].cuda()
        x1 = x1.permute(0,3,1,2)[:,:3,:,:].contiguous()
        x2 = x2.permute(0,3,1,2)[:,:3,:,:].contiguous()
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)

        outs1 = []
        outs2 = []
        for i in range(0,self.nb_primitives):
            context1 = x1.unsqueeze(2).expand(x1.size(0),x1.size(1), self.rand_grid.size(2)).contiguous()
            shape_y1 = torch.cat( (self.rand_grid, context1), 1).contiguous()
            t1, s1, r1 = self.affine_decoder[i](shape_y1)
            deformed_y1 = apply_transform(self.rand_grid, t1.unsqueeze(-1), torch.ones_like(s1.unsqueeze(-1)).to(s1.device), r1.unsqueeze(-1))
            c1 = self.color_decoder[i](self.rand_grid.transpose(2,1)).transpose(2, 1)
            if self.args.color_activation == "siren": c1 = (c1 + 1.0)/2.0
            outs1.append(torch.cat(( deformed_y1, c1), 1))

            context2 = x2.unsqueeze(2).expand(x2.size(0),x2.size(1), self.rand_grid.size(2)).contiguous()
            shape_y2 = torch.cat( (self.rand_grid, context2), 1).contiguous()
            t2, s2, r2 = self.affine_decoder[i](shape_y2)
            deformed_y2 = apply_transform(self.rand_grid, t2.unsqueeze(-1), torch.ones_like(s2.unsqueeze(-1)).to(s2.device), r2.unsqueeze(-1))

            c2 = self.color_decoder[i](self.rand_grid.transpose(2,1)).transpose(2, 1)
            if self.args.color_activation == "siren": c2 = (c2 + 1.0)/2.0
            outs2.append(torch.cat(( deformed_y2, c2), 1))

        outs1 = torch.cat(outs1,2).contiguous().transpose(2,1).contiguous()
        outs2 = torch.cat(outs2,2).contiguous().transpose(2,1).contiguous()

        all_loss = self.loss(points1, outs1, points2, outs2, s=self.args.shape_color_scale)
        loss = all_loss['loss']
        self.log('loss', {'loss': loss.item(), 'shape_loss': all_loss['shape_loss'], 'color_loss': all_loss['color_loss'], 'temporal_loss': all_loss['temporal_loss']})

        if self.vis_counter % self.args.vis_freq == 0:
            visualize(input_batch=batch, input_patch=self.rand_grid, pred1=outs1, pred2=outs2, pairing=all_loss['pairing'], displace=None, batch_idx=self.vis_counter, save_dir=self.args.output_dir)
        
        if self.vis_counter % self.args.save_freq == 0:
            os.makedirs(self.args.output_dir+f'/ckpts/', exist_ok=True)
            torch.save(self.state_dict, self.args.output_dir+f'/ckpts/ckpts{self.vis_counter}.pt')
        self.vis_counter += 1

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def forward_inference(self, x, grid):
        x = self.encoder(x)
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(grid[i]))
            rand_grid = rand_grid.transpose(0,1).contiguous().unsqueeze(0)
            rand_grid = rand_grid.expand(x.size(0),rand_grid.size(1), rand_grid.size(2)).contiguous()
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def forward_inference_from_latent_space(self, x, grid):
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(grid[i]))
            rand_grid = rand_grid.transpose(0,1).contiguous().unsqueeze(0)
            rand_grid = rand_grid.expand(x.size(0),rand_grid.size(1), rand_grid.size(2)).contiguous()
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()  

    def forward(self, batch, batch_idx):
        # in lightning, forward defines the prediction/inference actions
        grid = None
        if x.shape[-1] == self.bottleneck_size:
            return self.forward_inference_from_latent_space(x, grid)
        else:
            return self.forward_inference(x, grid)


# identity matrix network for predicting color only
class IdentityNet(pl.LightningModule):

    def __init__(self, args, learning_rate=1e-3, batch_size=32, num_points = 2048, bottleneck_size = 1000, nb_primitives = 1, output_dim=3):
        super().__init__()
        super(IdentityNet, self).__init__()

        self.args = args
        self.learning_rate = learning_rate
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        self.encoder = resnet18(pretrained=True, num_classes=bottleneck_size)
        self.color_decoder = nn.ModuleList([FCBlock(in_features=2, out_features=3, num_hidden_layers=3, hidden_features=256,  \
                                    outermost_linear=False, nonlinearity=self.args.color_activation)])
        self.chamfer2D = dist_chamfer_2D.chamfer_2DDist()
        self.chamfer3D = dist_chamfer_3D.chamfer_3DDist()
        self.chamfer5D = dist_chamfer_5D.chamfer_5DDist()
        
        self.vis_counter = 0

    def from_pretrained(self, weight_path):
        return self.load_from_checkpoint(weight_path, strict=False)

    def loss(self, points1, output1, points2, output2):
        # implement the loss that 

        gt_shape1 = points1[:,:,:2]
        gt_shape2 = points2[:,:,:2]
        output_shape_1 = output1[:,:,:2]
        output_shape_2 = output2[:,:,:2]
        gt_color1 = points1[:,:,2:]
        gt_color2 = points2[:,:,2:]
        output_color_1 = output1[:,:,2:]
        output_color_2 = output2[:,:,2:]

        batch_loss = {
            'chamfer5d_loss': 0.0, 
            'shape_loss': 0.0,
            'color_loss': 0.0,
            'temporal_loss': 0.0,
            'loss': 0.0
        }

        # color only loss test siren
        color_loss1 = torch.pow(gt_color1 - output_color_1, 2).mean()
        color_loss2 = torch.pow(gt_color2 - output_color_2, 2).mean()
        loss = (color_loss1 + color_loss2) 
        batch_loss['color_loss'] = color_loss1 + color_loss2 

        batch_loss['loss'] = loss
        
        return batch_loss
            
        
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x1, x2, points1, points2 = batch['img1'].cuda(), batch['img2'].cuda(), batch['pts1'].cuda(), batch['pts2'].cuda()
        x1 = x1.permute(0,3,1,2)[:,:3,:,:].contiguous()
        x2 = x2.permute(0,3,1,2)[:,:3,:,:].contiguous()
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)


        outs1 = []
        outs2 = []
        for i in range(0,self.nb_primitives):
            rand_grid = points1[:,:,:2].clone().detach()
            c1 = self.color_decoder[i](rand_grid)
            if self.args.color_activation == "siren": c1 = (c1 + 1.0)/2.0
            outs1.append(torch.cat((rand_grid, c1), -1))

            rand_grid2 = points2[:,:,:2].clone().detach()
            c2 = self.color_decoder[i](rand_grid2)
            if self.args.color_activation == "siren": c2 = (c2 + 1.0)/2.0
            outs2.append(torch.cat((rand_grid2, c2), -1))

        outs1 = torch.cat(outs1,2).contiguous().contiguous()
        outs2 = torch.cat(outs2,2).contiguous().contiguous()

        all_loss = self.loss(points1, outs1, points2, outs2)
        loss = all_loss['loss']
        self.log('loss', {'loss': loss.item(), 'color_loss': all_loss['color_loss'].item()})


        if batch_idx % self.args.vis_freq == 0:
            visualization_identitynet.visualize(input_batch=batch, input_patch=rand_grid, pred1=outs1, pred2=outs2, batch_idx=self.vis_counter, save_dir=self.args.output_dir)
            self.vis_counter += self.args.vis_freq

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def forward_inference(self, x, grid):
        x = self.encoder(x)
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(grid[i]))
            rand_grid = rand_grid.transpose(0,1).contiguous().unsqueeze(0)
            rand_grid = rand_grid.expand(x.size(0),rand_grid.size(1), rand_grid.size(2)).contiguous()
            # print(rand_grid.sizerand_grid())
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def forward_inference_from_latent_space(self, x, grid):
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(grid[i]))
            rand_grid = rand_grid.transpose(0,1).contiguous().unsqueeze(0)
            rand_grid = rand_grid.expand(x.size(0),rand_grid.size(1), rand_grid.size(2)).contiguous()
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()  

    def forward(self, batch, batch_idx):
        # in lightning, forward defines the prediction/inference actions
        grid = None
        if x.shape[-1] == self.bottleneck_size:
            return self.forward_inference_from_latent_space(x, grid)
        else:
            return self.forward_inference(x, grid)
    
