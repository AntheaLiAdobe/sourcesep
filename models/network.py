import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.autograd as ag
from torch.autograd import Variable
from torchvision import transforms
from torchvision.datasets import MNIST
import pytorch_lightning as pl

from models.modules import df, get_jacobian, resnet18, resnet18_resetlast, resnet50, PointGenCon, PointGenCon5D, PointGenConColor, PointGenConScale, distChamfer
from visualization import visualize
import ChamferDistancePytorch.chamfer2D.dist_chamfer_2D as dist_chamfer_2D
import ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as dist_chamfer_3D
import ChamferDistancePytorch.chamfer5D.dist_chamfer_5D as dist_chamfer_5D


# atlasnet decoder that predicts shape and color separately
class AE_AtlasNet(pl.LightningModule):

    def __init__(self, args, learning_rate=1e-3, batch_size=32, num_points = 2048, bottleneck_size=64, nb_primitives = 1, output_dim=2):
        super().__init__()
        super(AE_AtlasNet, self).__init__()
        self.args = args
        self.learning_rate = learning_rate
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        self.encoder = resnet18_resetlast(pretrained=True, num_classes=bottleneck_size)
        self.shape_decoder = nn.ModuleList([PointGenCon(bottleneck_size = 2 + self.bottleneck_size, output_dim=2, activation=self.args.shape_activation) for i in range(0,self.nb_primitives)])
        self.color_decoder = nn.ModuleList([PointGenConColor(bottleneck_size = 2 + self.bottleneck_size, output_dim=3, activation=self.args.color_activation) for i in range(0,self.nb_primitives)])
        self.chamfer2D = dist_chamfer_2D.chamfer_2DDist()
        self.chamfer3D = dist_chamfer_3D.chamfer_3DDist()
        self.chamfer5D = dist_chamfer_5D.chamfer_5DDist()
        self.rand_grid = Variable(torch.cuda.FloatTensor(self.args.batch_size,2,self.num_points//self.nb_primitives))
        self.rand_grid.data.uniform_(0,1)
        self.rand_grid.requires_grad = True

        self.vis_counter = 0

    def from_pretrained(self, weight_path):
        return self.load_from_checkpoint(weight_path, strict=False)

    def loss(self, points1, output1, points2, output2, jacobian=None, s=1.0, delta=1.0):
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
            'pairing': None,
            'displace': None
        }

        if self.args.chamfer5d_loss:
            # change the relative scaling of [x,y] and [color] 
            relative_scale = torch.tensor([1.0, 1.0, s, s, s]).view(1, 1, 5).to(points1.device)
            points1 =  points1 * relative_scale
            output1 = output1 * relative_scale
            points2 =  points2 * relative_scale
            output2 = output2 * relative_scale
            
            dist1_1, dist1_2, idx1_1, idx1_2 = self.chamfer5D(points1, output1)
            loss1 = (torch.mean(dist1_1)) + (torch.mean(dist1_2))

            dist2_1, dist2_2, idx2_1, idx2_2 = self.chamfer5D(points2, output2)
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
                time_loss = torch.pow(output1 - output2, 2).mean()

                loss += time_loss * self.args.temporal_loss_weight
                batch_loss['temporal_loss'] = time_loss.item()

            else:
                shape_time_loss = torch.pow(output_shape_1 - output_shape_2, 2).mean()

                color_time_loss = torch.pow(output_color_2 - output_color_1, 2).mean()
                loss += ( color_time_loss * self.args.color_temporal_loss_weight +  shape_time_loss * self.args.shape_temporal_loss_weight) * self.args.temporal_loss_weight
                batch_loss['temporal_loss'] = color_time_loss.item() + shape_time_loss.item()
        if self.args.jacobian and jacobian is not None:
            j1, j2 = jacobian
            loss += torch.pow(torch.abs(j1) - j1, 2).mean() + torch.pow(torch.abs(j2) - j2, 2).mean()
            batch_loss['jacobian'] = torch.pow(torch.abs(j1) - j1, 2).mean().item() + torch.pow(torch.abs(j2) - j2, 2).mean().item()


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

            context1 = x1.unsqueeze(2).expand(x1.size(0),x1.size(1), self.rand_grid.size(2))
            deformed_y1 = self.shape_decoder[i](self.rand_grid, context1)
            j1 = get_jacobian(deformed_y1, self.rand_grid, transpose=True)
            shape_y1 = torch.cat( (deformed_y1, context1), 1)
            c1 = self.color_decoder[i](shape_y1)
            if self.args.color_activation == "siren": c1 = (c1 + 1.0)/2.0
            outs1.append(torch.cat(( deformed_y1, c1), 1))

            context2 = x2.unsqueeze(2).expand(x2.size(0),x2.size(1), self.rand_grid.size(2))
            deformed_y2 = self.shape_decoder[i](self.rand_grid, context2)
            j2 = get_jacobian(deformed_y2, self.rand_grid, transpose=True)
            shape_y2 = torch.cat( (deformed_y2, context2), 1)
            c2 = self.color_decoder[i](shape_y2)
            if self.args.color_activation == "siren": c2 = (c2 + 1.0)/2.0
            outs2.append(torch.cat(( deformed_y2, c2), 1))

        outs1 = torch.cat(outs1,2).contiguous().transpose(2,1).contiguous()
        outs2 = torch.cat(outs2,2).contiguous().transpose(2,1).contiguous()
        jacobians = (j1, j2) if self.args.jacobian else None

        all_loss = self.loss(points1, outs1, points2, outs2, jacobian=jacobians, s=self.args.shape_color_scale)
        loss = all_loss['loss']
        self.log('loss', {'loss': loss.item(), 'shape_loss': all_loss['shape_loss'], 'color_loss': all_loss['color_loss'], 'temporal_loss': all_loss['temporal_loss']})

        if self.vis_counter % self.args.vis_freq == 0:
            vis_batch = True if (self.vis_counter+1) % (self.args.vis_freq*100) == 0 else False
            visualize(input_batch=batch, input_patch=self.rand_grid, pred1=outs1, pred2=outs2, pairing=all_loss['pairing'], displace=None, batch_idx=self.vis_counter, all_frame=vis_batch, save_dir=self.args.output_dir)
        
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
        

# atlasnet decoder that predicts shape and color together
class AE_AtlasNet5D(pl.LightningModule):

    def __init__(self, args, learning_rate=1e-3, batch_size=32, num_points = 2048, bottleneck_size=64, nb_primitives = 1, output_dim=2):
        super().__init__()
        super(AE_AtlasNet5D, self).__init__()
        self.args = args
        self.learning_rate = learning_rate
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        self.encoder = resnet18_resetlast(pretrained=True, num_classes=bottleneck_size)
        self.decoder = nn.ModuleList([PointGenCon5D(bottleneck_size = 2 + self.bottleneck_size, output_dim=5, activation=self.args.color_activation) for i in range(0,self.nb_primitives)])
        self.chamfer2D = dist_chamfer_2D.chamfer_2DDist()
        self.chamfer3D = dist_chamfer_3D.chamfer_3DDist()
        self.chamfer5D = dist_chamfer_5D.chamfer_5DDist()
        
        self.rand_grid = Variable(torch.cuda.FloatTensor(x1.size(0),2,self.num_points//self.nb_primitives))
        self.rand_grid.data.uniform_(0,1)
        self.vis_counter = 0

    def from_pretrained(self, weight_path):
        return self.load_from_checkpoint(weight_path, strict=False)

    def loss(self, points1, output1, points2, output2, jacobian=None, s=1.0, delta=1.0):
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
            'pairing': None,
            'displace': None
        }

        if self.args.chamfer5d_loss:
            # change the relative scaling of [x,y] and [color] 
            relative_scale = torch.tensor([1.0, 1.0, s, s, s]).view(1, 1, 5).to(points1.device)
            points1 =  points1 * relative_scale
            output1 = output1 * relative_scale
            points2 =  points2 * relative_scale
            output2 = output2 * relative_scale
            
            dist1_1, dist1_2, idx1_1, idx1_2 = self.chamfer5D(points1, output1)
            loss1 = (torch.mean(dist1_1)) + (torch.mean(dist1_2))

            dist2_1, dist2_2, idx2_1, idx2_2 = self.chamfer5D(points2, output2)
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
                time_loss = torch.pow(output1 - output2, 2).mean()

                loss += time_loss * self.args.temporal_loss_weight
                batch_loss['temporal_loss'] = time_loss.item()

            else:
                shape_time_loss = torch.pow(output_shape_1 - output_shape_2, 2).mean()

                color_time_loss = torch.pow(output_color_2 - output_color_1, 2).mean()
                loss += ( color_time_loss * self.args.color_temporal_loss_weight +  shape_time_loss * self.args.shape_temporal_loss_weight) * self.args.temporal_loss_weight
                batch_loss['temporal_loss'] = color_time_loss.item() + shape_time_loss.item()
        
        if self.args.jacobian and jacobian is not None:
            j1, j2 = jacobian
            loss += torch.pow(torch.abs(j1) - j1, 2).mean() + torch.pow(torch.abs(j2) - j2, 2).mean()
            batch_loss['jacobian'] = torch.pow(torch.abs(j1) - j1, 2).mean().item() + torch.pow(torch.abs(j2) - j2, 2).mean().item()

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

            context1 = x1.unsqueeze(2).expand(x1.size(0),x1.size(1), self.rand_grid.size(2))
            out_y1 = self.decoder[i](self.rand_grid, context1)
            j1 = get_jacobian(out_y1[:,:,:2], self.rand_grid, transpose=True)
            c1 = out_y1[:,:,2:]
            if self.args.color_activation == "siren": c1 = (c1 + 1.0)/2.0
            outs1.append(torch.cat((out_y1[:,:,:2], c1), 1))

            context2 = x2.unsqueeze(2).expand(x2.size(0),x2.size(1), self.rand_grid.size(2))
            out_y2 = self.decoder[i](self.rand_grid, context2)
            j2 = get_jacobian(out_y2[:,:,:2], self.rand_grid, transpose=True)
            c2 = out_y2[:,:,2:]
            if self.args.color_activation == "siren": c2 = (c2 + 1.0)/2.0
            outs2.append(torch.cat((out_y2[:,:,:2], c2), 1))

        outs1 = torch.cat(outs1,2).contiguous().transpose(2,1).contiguous()
        outs2 = torch.cat(outs2,2).contiguous().transpose(2,1).contiguous()

        jacobians = (j1, j2) if self.args.jacobian else None
        
        all_loss = self.loss(points1, outs1, points2, outs2, jacobian=jacobians, s=self.args.shape_color_scale)
        loss = all_loss['loss']
        self.log('loss', {'loss': loss.item(), 'shape_loss': all_loss['shape_loss'], 'color_loss': all_loss['color_loss'], 'temporal_loss': all_loss['temporal_loss']})

        if self.vis_counter % self.args.vis_freq == 0:
            vis_batch = True if (self.vis_counter+1) % (self.args.vis_freq*100) == 0 else False
            visualize(input_batch=batch, input_patch=self.rand_grid, pred1=outs1, pred2=outs2, pairing=all_loss['pairing'], displace=None, batch_idx=self.vis_counter, all_frame=vis_batch, save_dir=self.args.output_dir)
        
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
        

# network that predict shape first and use the predicted shape to predict color.
class AE_AtlasNetInPlace(pl.LightningModule):

    def __init__(self, args, learning_rate=1e-3, batch_size=32, num_points = 2048, bottleneck_size=64, nb_primitives = 1, output_dim=2):
        super().__init__()
        super(AE_AtlasNetInPlace, self).__init__()
        self.args = args
        self.learning_rate = learning_rate
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        self.encoder = resnet18_resetlast(pretrained=True, num_classes=bottleneck_size)
        self.shape_decoder = nn.ModuleList([PointGenCon(bottleneck_size = 2 + self.bottleneck_size, output_dim=2, activation=self.args.shape_activation) for i in range(0,self.nb_primitives)])
        self.color_decoder = nn.ModuleList([PointGenConColor(bottleneck_size = 2 + self.bottleneck_size, output_dim=3, activation=self.args.color_activation) for i in range(0,self.nb_primitives)])
        self.s1  = nn.Parameter(torch.ones(1, 1, 1, requires_grad=True).float())
        self.s2  = nn.Parameter(torch.ones(1, 1, 1, requires_grad=True).float())
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
            'loss': 0.0,
        }

        if self.args.chamfer5d_loss:
            dist1_1, dist1_2, idx1_1, idx1_2 = self.chamfer5D(points1, output1)
            loss1 = (torch.mean(dist1_1)) + (torch.mean(dist1_2))

            dist2_1, dist2_2, idx2_1, idx2_2 = self.chamfer5D(points2, output2)
            loss2 = (torch.mean(dist2_1)) + (torch.mean(dist2_2))

            loss = ( loss1 + loss2 ) * self.args.shape_loss_weight
            batch_loss['chamfer5d_loss'] = loss1 + loss2 

        else:
            
            dist1_1, dist1_2, idx1_1, idx1_2 = self.chamfer2D(gt_shape1, output_shape_1)
            shape_loss1 = (torch.mean(dist1_1)) + (torch.mean(dist1_2))

            dist2_1, dist2_2, idx2_1, idx2_2 = self.chamfer2D(gt_shape2, output_shape_2)
            shape_loss2 = (torch.mean(dist2_1)) + (torch.mean(dist2_2))

            loss = ( shape_loss1 + shape_loss2 ) * self.args.shape_loss_weight
            batch_loss['shape_loss'] = shape_loss1 + shape_loss2 

            if not self.args.no_color_loss:
                idx1_2 = torch.clamp(idx1_2, min=0, max=gt_color2.shape[1])
                idx2_2 = torch.clamp(idx2_2, min=0, max=gt_color2.shape[1])

                color_loss1 = torch.pow(gt_color1.gather(1, idx1_2.unsqueeze(-1).repeat(1,1,3).long()) - output_color_1, 2).mean()
                color_loss2 = torch.pow(gt_color2.gather(1, idx2_2.unsqueeze(-1).repeat(1,1,3).long()) - output_color_2, 2).mean()

                loss += ( color_loss1 + color_loss2 ) * self.args.color_loss_weight

                batch_loss['color_loss'] = color_loss1 + color_loss2 

        if not self.args.no_temporal_loss:
            
            if self.args.chamfer5d_loss:        
                dist_time_1, dist_time_2, idx_time_1, idx_time_2 = self.chamfer5D(output1, output2)
                time_loss = torch.mean(dist_time_1) + torch.mean(dist_time_2)

                loss += time_loss * self.args.temporal_loss_weight
                batch_loss['temporal_loss'] = time_loss

            else:
                dist_time_1, dist_time_2, idx_time_1, idx_time_2 = self.chamfer2D(output_shape_1, output_shape_2)
                shape_time_loss = dist_time_1.mean() + dist_time_2.mean()

                color_time_loss = torch.pow(output_color_2 - output_color_1, 2).mean()
                loss += ( color_time_loss + shape_time_loss )  * self.args.temporal_loss_weight
                batch_loss['temporal_loss'] = color_time_loss + shape_time_loss
        
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
            rand_grid = Variable(torch.cuda.FloatTensor(x1.size(0),2,self.num_points//self.nb_primitives))
            rand_grid.data.uniform_(0,1)
            self.rand_grid = rand_grid

            shape_y1 = rand_grid * torch.relu(self.s1)
            context1 = x1.unsqueeze(2).expand(x1.size(0),x1.size(1), rand_grid.size(2)).contiguous()
            shape_y1 = torch.cat( (shape_y1, context1), 1).contiguous()
            deformed_grid1 = self.shape_decoder[i](shape_y1)
            y1 = torch.cat( (deformed_grid1, context1), 1).contiguous()
            outs1.append(torch.cat(( deformed_grid1, self.color_decoder[i](y1)), 1))


            shape_y2 = rand_grid * torch.relu(self.s2)
            context2 = x2.unsqueeze(2).expand(x2.size(0),x2.size(1), shape_y2.size(2)).contiguous()
            shape_y2 = torch.cat( (rand_grid, context2), 1).contiguous()
            deformed_grid2, s2 = self.shape_decoder[i](shape_y2)
            y2 = torch.cat( (deformed_grid2, context2), 1).contiguous()
            outs2.append(torch.cat(( deformed_grid2, self.color_decoder[i](y2)), 1))

        outs1 = torch.cat(outs1,2).contiguous().transpose(2,1).contiguous()
        outs2 = torch.cat(outs2,2).contiguous().transpose(2,1).contiguous()

        all_loss = self.loss(points1, outs1, points2, outs2)
        loss = all_loss['loss']        
        self.log('loss', {'loss': loss.item(), 'shape_loss': all_loss['shape_loss'].item(), 'color_loss': all_loss['color_loss'].item(), 'chamfer5d_loss': all_loss['chamfer5d_loss'].item(), 'temporal_loss': all_loss['temporal_loss'].item()})



        if batch_idx % self.args.vis_freq == 0:
            visualize(input_batch=batch, input_patch=rand_grid, pred1=outs1, pred2=outs2, batch_idx=self.vis_counter, save_dir=self.args.output_dir)
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
        


