import platform
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch3d.ops
import pytorch3d.transforms
import cv2 as cv

import config
from network.styleunet.dual_styleunet import DualStyleUNet
from gaussians.gaussian_model import GaussianModel
from gaussians.gaussian_renderer import render3
from utils.image_util import interpolate_image_masked


class AvatarNet(nn.Module):
    def __init__(self, opt):
        super(AvatarNet, self).__init__()
        self.opt = opt
        self.random_style = opt.get('random_style', False)
        self.with_viewdirs = opt.get('with_viewdirs', True)
        self.face_U = opt.get('face_U', 118)
        self.face_V1 = 510
        self.face_V2 = 510 + 1024
        self.face_d = 64
        U, V1, V2, d = self.face_U, self.face_V1, self.face_V2, self.face_d
        # init canonical Gaussian model (canonical = T-pose; pos map from gen_pos_maps.py)
        self.max_sh_degree = 0
        self.cano_gaussian_model = GaussianModel(sh_degree = self.max_sh_degree)
        cano_smpl_map = cv.imread(config.opt['train']['data']['data_dir'] + '/smpl_pos_map/cano_smpl_pos_map.exr', cv.IMREAD_UNCHANGED)
        self.cano_smpl_map = torch.from_numpy(cano_smpl_map).to(torch.float32).to(config.device)
        self.cano_smpl_mask = torch.linalg.norm(self.cano_smpl_map, dim = -1) > 0.
        # save the cano_smpl_mask as png file (OpenCV expects uint8 0-255, not bool)
        cv.imwrite('cano_smpl_mask.png', (self.cano_smpl_mask.cpu().numpy().astype(np.uint8) * 255))
        self.init_points = self.cano_smpl_map[self.cano_smpl_mask]
        self.lbs = torch.from_numpy(np.load(config.opt['train']['data']['data_dir'] + '/smpl_pos_map/init_pts_lbs.npy')).to(torch.float32).to(config.device)
        self.cano_gaussian_model.create_from_pcd(self.init_points, torch.rand_like(self.init_points), spatial_lr_scale = 2.5)
        self.face_gaussian_model = GaussianModel(sh_degree = self.max_sh_degree)
        face_cano_raw = torch.concat([self.cano_smpl_map[U-d:U+d, V1-d:V1+d], self.cano_smpl_map[U-d:U+d, V2-d:V2+d]], 1)
        self.face_cano_smpl_map = interpolate_image_masked(
            face_cano_raw,
            target_h=2*face_cano_raw.shape[0],
            target_w=2*face_cano_raw.shape[1],
            eps=1e-6,
            threshold=0.5,
            fill=0.0,
        )
        self.face_cano_smpl_mask = torch.linalg.norm(self.face_cano_smpl_map, dim = -1) > 0.
        # save the face_cano_smpl_mask as png file
        # save the face_cano_smpl_map as png file
        self.face_init_points = self.face_cano_smpl_map[self.face_cano_smpl_mask]
        # save the face_cano_smpl_map as png file
        self.face_gaussian_model.create_from_pcd(self.face_init_points, torch.rand_like(self.face_init_points), spatial_lr_scale = 2.5)
        # Interpolate LBS from body points to face points (KNN) so transform_cano2live works for face
        # initialize styleunet networks for the body canonical gaussian model
        self.color_net = DualStyleUNet(inp_size = 512, inp_ch = 3, out_ch = 3, out_size = 1024, style_dim = 512, n_mlp = 2)
        self.position_net = DualStyleUNet(inp_size = 512, inp_ch = 3, out_ch = 3, out_size = 1024, style_dim = 512, n_mlp = 2)
        self.other_net = DualStyleUNet(inp_size = 512, inp_ch = 3, out_ch = 8, out_size = 1024, style_dim = 512, n_mlp = 2)
        # Face nets: n_mlp and channel_multiplier configurable via model.face_n_mlp / model.face_channel_multiplier
        face_n_mlp = opt.get('face_n_mlp', 2)
        face_channel_multiplier = opt.get('face_channel_multiplier', 1)
        self.color_net_face = DualStyleUNet(inp_size=256, inp_ch=3, out_ch=3, out_size=256, style_dim=256, n_mlp=face_n_mlp, channel_multiplier=face_channel_multiplier)
        self.position_net_face = DualStyleUNet(inp_size=256, inp_ch=3, out_ch=3, out_size=256, style_dim=256, n_mlp=face_n_mlp, channel_multiplier=face_channel_multiplier)
        self.other_net_face = DualStyleUNet(inp_size=256, inp_ch=3, out_ch=8, out_size=256, style_dim=256, n_mlp=face_n_mlp, channel_multiplier=face_channel_multiplier)
        # initalize styles for the body canonical gaussian model
        self.color_style = torch.ones([1, self.color_net.style_dim], dtype=torch.float32, device=config.device) / np.sqrt(self.color_net.style_dim)
        self.position_style = torch.ones([1, self.position_net.style_dim], dtype=torch.float32, device=config.device) / np.sqrt(self.position_net.style_dim)
        self.other_style = torch.ones([1, self.other_net.style_dim], dtype=torch.float32, device=config.device) / np.sqrt(self.other_net.style_dim)
        # initalize styles for the face canonical gaussian model
        self.color_style_face = torch.ones([1, self.color_net_face.style_dim], dtype=torch.float32, device=config.device) / np.sqrt(self.color_net_face.style_dim)
        self.position_style_face = torch.ones([1, self.position_net_face.style_dim], dtype=torch.float32, device=config.device) / np.sqrt(self.position_net_face.style_dim)
        self.other_style_face = torch.ones([1, self.other_net_face.style_dim], dtype=torch.float32, device=config.device) / np.sqrt(self.other_net_face.style_dim)
        self._register_face_lbs(k=8)

        if self.with_viewdirs:
            cano_nml_map = cv.imread(config.opt['train']['data']['data_dir'] + '/smpl_pos_map/cano_smpl_nml_map.exr', cv.IMREAD_UNCHANGED)
            self.cano_nml_map = torch.from_numpy(cano_nml_map).to(torch.float32).to(config.device)
            self.cano_nmls = self.cano_nml_map[self.cano_smpl_mask]
            self.viewdir_net = nn.Sequential(
                nn.Conv2d(1, 64, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace = True),
                nn.Conv2d(64, 128, 4, 2, 1)
            )

    def _register_face_lbs(self, k=8, eps=1e-6, chunk_size=2048):
        """Compute face LBS by KNN interpolation from body init_points and lbs; register as buffer.
        Distance matrix and topk are computed on CPU to avoid GPU OOM."""
        device = self.init_points.device
        face_cpu = self.face_init_points.cpu()
        body_cpu = self.init_points.cpu()
        n_face = face_cpu.size(0)
        n_body = body_cpu.size(0)
        k_actual = min(k, n_body)
        knn_dists_list = []
        knn_idx_list = []
        for start in range(0, n_face, chunk_size):
            end = min(start + chunk_size, n_face)
            chunk = face_cpu[start:end]
            dists_chunk = torch.cdist(chunk, body_cpu, p=2)
            d, idx = dists_chunk.topk(k_actual, dim=1, largest=False)
            knn_dists_list.append(d)
            knn_idx_list.append(idx)
        knn_dists = torch.cat(knn_dists_list, dim=0)
        knn_idx = torch.cat(knn_idx_list, dim=0)
        w = 1.0 / (knn_dists + eps)
        w = w / w.sum(dim=1, keepdim=True)
        # Move only small (N_face, K) tensors to GPU; LBS blend on device
        knn_idx = knn_idx.to(device)
        w = w.to(device)
        neighbor_lbs = self.lbs[knn_idx]   # (N_face, K, J)
        face_lbs = (w.unsqueeze(-1) * neighbor_lbs).sum(dim=1)
        face_lbs = face_lbs.detach()  # Ensure face_lbs has no grad
        # print some rows from face_lbs
        self.register_buffer('face_lbs', face_lbs)

    def generate_mean_hands(self):
        # print('# Generating mean hands ...')
        import glob
        # get hand mask
        lbs_argmax = self.lbs.argmax(1)
        self.hand_mask = lbs_argmax == 20
        self.hand_mask = torch.logical_or(self.hand_mask, lbs_argmax == 21)
        self.hand_mask = torch.logical_or(self.hand_mask, lbs_argmax >= 25)

        pose_map_paths = sorted(glob.glob(config.opt['train']['data']['data_dir'] + '/smpl_pos_map/%08d.exr' % config.opt['test']['fix_hand_id']))
        smpl_pos_map = cv.imread(pose_map_paths[0], cv.IMREAD_UNCHANGED)
        pos_map_size = smpl_pos_map.shape[1] // 2
        smpl_pos_map = np.concatenate([smpl_pos_map[:, :pos_map_size], smpl_pos_map[:, pos_map_size:]], 2)
        smpl_pos_map = smpl_pos_map.transpose((2, 0, 1))
        pose_map = torch.from_numpy(smpl_pos_map).to(torch.float32).to(config.device)
        pose_map = pose_map[:3]

        cano_pts = self.get_positions(pose_map)
        opacity, scales, rotations = self.get_others(pose_map)
        colors, color_map = self.get_colors(pose_map)

        self.hand_positions = cano_pts#[self.hand_mask]
        self.hand_opacity = opacity#[self.hand_mask]
        self.hand_scales = scales#[self.hand_mask]
        self.hand_rotations = rotations#[self.hand_mask]
        self.hand_colors = colors#[self.hand_mask]

    def update_face_gaussian_model(self):
        return self.face_gaussian_model
    
    def transform_cano2live(self, gaussian_vals, items, lbs=None):
        if lbs is None:
            lbs = self.lbs
        pt_mats = torch.einsum('nj,jxy->nxy', lbs, items['cano2live_jnt_mats'])
        gaussian_vals['positions'] = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], gaussian_vals['positions']) + pt_mats[..., :3, 3]
        rot_mats = pytorch3d.transforms.quaternion_to_matrix(gaussian_vals['rotations'])
        rot_mats = torch.einsum('nxy,nyz->nxz', pt_mats[..., :3, :3], rot_mats)
        gaussian_vals['rotations'] = pytorch3d.transforms.matrix_to_quaternion(rot_mats)

        return gaussian_vals

    def transform_cano2live_face(self, gaussian_vals, items):
        """Same as transform_cano2live but using KNN-interpolated LBS for face points.
        Uses a detached copy of cano2live_jnt_mats so face losses don't share graph
        with the body branch (avoids double-backward through shared joints)."""
        mats = items['cano2live_jnt_mats'].detach()
        pt_mats = torch.einsum('nj,jxy->nxy', self.face_lbs, mats)
        gaussian_vals['positions'] = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], gaussian_vals['positions']) + pt_mats[..., :3, 3]
        rot_mats = pytorch3d.transforms.quaternion_to_matrix(gaussian_vals['rotations'])
        rot_mats = torch.einsum('nxy,nyz->nxz', pt_mats[..., :3, :3], rot_mats)
        gaussian_vals['rotations'] = pytorch3d.transforms.matrix_to_quaternion(rot_mats)
        return gaussian_vals

    def get_positions(self, pose_map, return_map = False):
        position_map, _ = self.position_net([self.position_style], pose_map[None], randomize_noise = False)
        front_position_map, back_position_map = torch.split(position_map, [3, 3], 1)
        position_map = torch.cat([front_position_map, back_position_map], 3)[0].permute(1, 2, 0)
        delta_position = 0.05 * position_map[self.cano_smpl_mask]
        # delta_position = position_map[self.cano_smpl_mask]

        positions = delta_position + self.cano_gaussian_model.get_xyz
        if return_map:
            return positions, position_map
        else:
            return positions
    
    def get_face_positions(self, position_map):
        positions_map, _ = self.position_net_face([self.position_style_face], position_map[None], randomize_noise = False)
        front_position_map, back_position_map = torch.split(positions_map, [3, 3], 1)
        positions_map = torch.cat([front_position_map, back_position_map], 3)[0].permute(1, 2, 0)
        positions = 0.05 *positions_map[self.face_cano_smpl_mask]
        positions = positions + self.face_gaussian_model.get_xyz
        return positions, positions_map
         
    def get_others(self, pose_map):
        other_map, _ = self.other_net([self.other_style], pose_map[None], randomize_noise = False)
        front_map, back_map = torch.split(other_map, [8, 8], 1)
        other_map = torch.cat([front_map, back_map], 3)[0].permute(1, 2, 0)
        others = other_map[self.cano_smpl_mask]  # (N, 8)
        opacity, scales, rotations = torch.split(others, [1, 3, 4], 1)
        opacity = self.cano_gaussian_model.opacity_activation(opacity + self.cano_gaussian_model.get_opacity_raw)
        scales = self.cano_gaussian_model.scaling_activation(scales + self.cano_gaussian_model.get_scaling_raw)
        rotations = self.cano_gaussian_model.rotation_activation(rotations + self.cano_gaussian_model.get_rotation_raw)

        return opacity, scales, rotations
    
    def get_face_others(self, position_map):
        other_map, _ = self.other_net_face([self.other_style_face], position_map[None], randomize_noise = False)
        front_map, back_map = torch.split(other_map, [8, 8], 1)
        other_map = torch.cat([front_map, back_map], 3)[0].permute(1, 2, 0)
        others = other_map[self.face_cano_smpl_mask]  # (N, 8)
        opacity, scales, rotations = torch.split(others, [1, 3, 4], 1)
        opacity = self.face_gaussian_model.opacity_activation(opacity + self.face_gaussian_model.get_opacity_raw)
        scales = self.face_gaussian_model.scaling_activation(scales + self.face_gaussian_model.get_scaling_raw)
        # Prevent face Gaussians from becoming too large in canonical space to keep
        # the face region sharp and avoid unstable kernels. We softly clamp the
        # activated scales to at most 0.15 per dimension.
        scales = torch.clamp(scales, max=0.15)
        rotations = self.face_gaussian_model.rotation_activation(rotations + self.face_gaussian_model.get_rotation_raw)
        return opacity, scales, rotations

    def get_colors(self, pose_map, front_viewdirs = None, back_viewdirs = None):
        color_style = torch.rand_like(self.color_style) if self.random_style and self.training else self.color_style
        color_map, _ = self.color_net([color_style], pose_map[None], randomize_noise = False, view_feature1 = front_viewdirs, view_feature2 = back_viewdirs)
        front_color_map, back_color_map = torch.split(color_map, [3, 3], 1)
        color_map = torch.cat([front_color_map, back_color_map], 3)[0].permute(1, 2, 0)
        colors = color_map[self.cano_smpl_mask]
        return colors, color_map

    def get_face_colors(self, position_map):
        color_map, _ = self.color_net_face([self.color_style_face], position_map[None], randomize_noise = False)
        front_color_map, back_color_map = torch.split(color_map, [3, 3], 1)
        color_map = torch.cat([front_color_map, back_color_map], 3)[0].permute(1, 2, 0)
        colors = color_map[self.face_cano_smpl_mask]
        return colors, color_map

    def get_viewdir_feat(self, items):
        with torch.no_grad():
            pt_mats = torch.einsum('nj,jxy->nxy', self.lbs, items['cano2live_jnt_mats'])
            live_pts = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], self.init_points) + pt_mats[..., :3, 3]
            live_nmls = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], self.cano_nmls)
            cam_pos = -torch.matmul(torch.linalg.inv(items['extr'][:3, :3]), items['extr'][:3, 3])
            viewdirs = F.normalize(cam_pos[None] - live_pts, dim = -1, eps = 1e-3)
            if self.training:
                viewdirs += torch.randn(*viewdirs.shape).to(viewdirs) * 0.1
            viewdirs = F.normalize(viewdirs, dim = -1, eps = 1e-3)
            viewdirs = (live_nmls * viewdirs).sum(-1)

            viewdirs_map = torch.zeros(*self.cano_nml_map.shape[:2]).to(viewdirs)
            viewdirs_map[self.cano_smpl_mask] = viewdirs

            viewdirs_map = viewdirs_map[None, None]
            viewdirs_map = F.interpolate(viewdirs_map, None, 0.5, 'nearest')
            front_viewdirs, back_viewdirs = torch.split(viewdirs_map, [512, 512], -1)

        front_viewdirs = self.opt.get('weight_viewdirs', 1.) * self.viewdir_net(front_viewdirs)
        back_viewdirs = self.opt.get('weight_viewdirs', 1.) * self.viewdir_net(back_viewdirs)
        return front_viewdirs, back_viewdirs

    def get_pose_map(self, items):
        pt_mats = torch.einsum('nj,jxy->nxy', self.lbs, items['cano2live_jnt_mats_woRoot'])
        live_pts = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], self.init_points) + pt_mats[..., :3, 3]
        live_pos_map = torch.zeros_like(self.cano_smpl_map)
        live_pos_map[self.cano_smpl_mask] = live_pts
        live_pos_map = F.interpolate(live_pos_map.permute(2, 0, 1)[None], None, [0.5, 0.5], mode = 'nearest')[0]
        live_pos_map = torch.cat(torch.split(live_pos_map, [512, 512], 2), 0)
        items.update({
            'smpl_pos_map': live_pos_map
        })
        return live_pos_map

    def render(self, items, bg_color=(0., 0., 0.), use_pca=False, use_vae=False, render=True):
        """
        Build body gaussians and optionally render. When render=True, transform to live and call render3;
        return dict with rgb_map, mask_map, offset, pos_map, posed_gaussians (and cano_tex_map when not training).
        When render=False, skip render3; return same keys with rgb_map=None, mask_map=None, posed_gaussians canonical.
        Note that no batch index in items.
        """
        bg_color = torch.from_numpy(np.asarray(bg_color)).to(torch.float32).to(config.device)
        pose_map = items['smpl_pos_map'][:3]
        assert not (use_pca and use_vae), "Cannot use both PCA and VAE!"
        if use_pca:
            pose_map = items['smpl_pos_map_pca'][:3]
        if use_vae:
            pose_map = items['smpl_pos_map_vae'][:3]

        cano_pts, pos_map = self.get_positions(pose_map, return_map=True)
        opacity, scales, rotations = self.get_others(pose_map)
        if self.with_viewdirs:
            front_viewdirs, back_viewdirs = self.get_viewdir_feat(items)
        else:
            front_viewdirs, back_viewdirs = None, None
        colors, color_map = self.get_colors(pose_map, front_viewdirs, back_viewdirs)

        if not self.training and config.opt['test'].get('fix_hand', False) and config.opt['mode'] == 'test':
            import utils.geo_util as geo_util
            cano_xyz = self.init_points
            wl = torch.sigmoid(2.5 * (geo_util.normalize_vert_bbox(items['left_cano_mano_v'], attris = cano_xyz, dim = 0, per_axis = True)[..., 0:1] + 2.0))
            wr = torch.sigmoid(-2.5 * (geo_util.normalize_vert_bbox(items['right_cano_mano_v'], attris = cano_xyz, dim = 0, per_axis = True)[..., 0:1] - 2.0))
            wl[cano_xyz[..., 1] < items['cano_smpl_center'][1]] = 0.
            wr[cano_xyz[..., 1] < items['cano_smpl_center'][1]] = 0.

            s = torch.maximum(wl + wr, torch.ones_like(wl))
            wl, wr = wl / s, wr / s

            w = wl + wr
            cano_pts = w * self.hand_positions + (1.0 - w) * cano_pts
            opacity = w * self.hand_opacity + (1.0 - w) * opacity
            scales = w * self.hand_scales + (1.0 - w) * scales
            rotations = w * self.hand_rotations + (1.0 - w) * rotations

        gaussian_vals = {
            'positions': cano_pts,
            'opacity': opacity,
            'scales': scales,
            'rotations': rotations,
            'colors': colors,
            'max_sh_degree': self.max_sh_degree
        }

        nonrigid_offset = gaussian_vals['positions'] - self.init_points
        canonical_gaussians = gaussian_vals.copy()

        if not render:
            ret = {
                'rgb_map': None,
                'mask_map': None,
                'offset': nonrigid_offset,
                'pos_map': pos_map,
                'canonical_gaussians': canonical_gaussians
            }
            if not self.training:
                ret['cano_tex_map'] = color_map
            return ret

        gaussian_vals = self.transform_cano2live(gaussian_vals, items)
        render_ret = render3(
            gaussian_vals,
            bg_color,
            items['extr'],
            items['intr'],
            items['img_w'],
            items['img_h']
        )
        rgb_map = render_ret['render'].permute(1, 2, 0)
        mask_map = render_ret['mask'].permute(1, 2, 0)

        ret = {
            'rgb_map': rgb_map,
            'mask_map': mask_map,
            'offset': nonrigid_offset,
            'pos_map': pos_map,
            'posed_gaussians': gaussian_vals
        }

        if not self.training:
            ret.update({
                'cano_tex_map': color_map,
                'posed_gaussians': gaussian_vals
            })

        return ret

    def render_face(self, pose_map, items, bg_color=(0., 0., 0.), render=True):
        """
        Build face gaussians from pose_map (same logic for render or not). When render=True,
        transform to live and call render3; return (rgb_map, mask_map, positions_face, gaussian_vals_live).
        When render=False, return (None, None, positions_face, gaussian_vals_canonical) for e.g. pretrain loss.
        """
        # Break any dependency on the body graph: face branch must not use body activations.
        pos_map = pose_map.detach().clone()
        #pose_map = items['smpl_pos_map'][:3]
        if not torch.is_tensor(bg_color):
            bg_color = torch.from_numpy(np.asarray(bg_color)).to(torch.float32).to(config.device)
        else:
            bg_color = bg_color.detach().clone().to(torch.float32).to(config.device)
        #_, pos_map = self.get_positions(pose_map, return_map=True)
        patch1 = pos_map[self.face_U-self.face_d:self.face_U+self.face_d, self.face_V1-self.face_d:self.face_V1+self.face_d]
        patch2 = pos_map[self.face_U-self.face_d:self.face_U+self.face_d, self.face_V2-self.face_d:self.face_V2+self.face_d]
        face_pos_map = torch.concat([patch1, patch2], 1)
        face_trained_map = interpolate_image_masked(
            face_pos_map,
            target_h=2 * face_pos_map.shape[0],
            target_w=2 * face_pos_map.shape[1],
            eps=1e-6,
            threshold=0.5,
            fill=0.0,
        )
        # save the face_pos_map as png file
        face_pos_map = face_trained_map.permute(2, 0, 1)[:, :, :256]
        positions_face, _ = self.get_face_positions(face_pos_map)
        opacity_face, scales_face, rotations_face = self.get_face_others(face_pos_map)
        colors_face, _ = self.get_face_colors(face_pos_map)


        gaussian_vals_cano = {
            'positions': positions_face,
            'opacity': opacity_face,
            'scales': scales_face,
            'rotations': rotations_face,
            'colors': colors_face,
            'max_sh_degree': self.max_sh_degree
        }
        # Sanitize face gaussians: replace NaNs/Infs to avoid CUDA illegal memory access in rasterizer.
        had_nonfinite = False
        for key in ['opacity', 'scales', 'rotations']:
            t = gaussian_vals_cano[key]
            finite_mask = torch.isfinite(t)
            if not finite_mask.all():
                had_nonfinite = True
                t = t.clone()
                t[~finite_mask] = 0.0
                gaussian_vals_cano[key] = t
        if not render:
            ret = {
                'rgb_map': None,
                'mask_map': None,
                'positions_face': positions_face,
                'canonical_gaussians': gaussian_vals_cano,
                'had_nonfinite': had_nonfinite
            }
            return ret
        nonrigid_offset = gaussian_vals_cano['positions'] - self.face_init_points
        gaussian_vals = self.transform_cano2live_face(gaussian_vals_cano, items)
        # Sanitize again after transform in case joint transforms introduce non-finite values.
        for key in ['positions', 'opacity', 'scales', 'rotations']:
            t = gaussian_vals[key]
            finite_mask = torch.isfinite(t)
            if not finite_mask.all():
                t = t.clone()
                t[~finite_mask] = 0.0
                gaussian_vals[key] = t
        render_ret = render3(
            gaussian_vals,
            bg_color,
            items['crops_extr_mats'],
            items['crops_intr_mats'],
            items['crops_img_widths'],
            items['crops_img_heights']
        )
        rgb_map = render_ret['render'].permute(1, 2, 0)
        mask_map = render_ret['mask'].permute(1, 2, 0)
        ret = {
            'rgb_map': rgb_map,
            'mask_map': mask_map,
            'offset': nonrigid_offset,
            'positions_face': positions_face,
            'posed_gaussians': gaussian_vals,
            'had_nonfinite': had_nonfinite
        }
        return ret

    def body_to_face_gaussians(self, body_gaussians):
        """
        Given body gaussians (dict with: positions, opacity, scales, rotations, colors, max_sh_degree),
        returns face_gaussians (same format) interpolated at the canonical face positions (2x upsampled
        face map, same as load_canonical_face_gaussian_model). Uses chunked cdists to avoid OOM.

        Args:
            body_gaussians: dict as returned by get_deformed_gaussians_body()
        Returns:
            face_gaussians: dict (positions, opacity, scales, rotations, colors, max_sh_degree)
        """
        cano_map = self.cano_smpl_map
        face_U, face_V1, face_V2, face_d = self.face_U, self.face_V1, self.face_V2, self.face_d
        body_positions_cano = body_gaussians['positions']  # (Nb, 3)

        # Same 2x upsampled face positions as load_canonical_face_gaussian_model (4x count)
        face_cano_raw = torch.concat([
            cano_map[face_U - face_d:face_U + face_d, face_V1 - face_d:face_V1 + face_d],
            cano_map[face_U - face_d:face_U + face_d, face_V2 - face_d:face_V2 + face_d],
        ], 1)
        face_cano_smpl_map = interpolate_image_masked(
            face_cano_raw,
            target_h=2 * face_cano_raw.shape[0],
            target_w=2 * face_cano_raw.shape[1],
            eps=1e-6,
            threshold=0.5,
            fill=0.0,
        )
        face_cano_smpl_mask = torch.linalg.norm(face_cano_smpl_map, dim=-1) > 0.0
        face_positions_cano = face_cano_smpl_map[face_cano_smpl_mask]  # (Nf_face, 3)

        chunk_size = 2048
        nearest_chunks = []
        with torch.no_grad():
            for i in range(0, face_positions_cano.shape[0], chunk_size):
                chunk = face_positions_cano[i:i + chunk_size]
                dists_chunk = torch.cdist(chunk, body_positions_cano)
                nearest_chunks.append(dists_chunk.argmin(dim=1))
            face_body_indices = torch.cat(nearest_chunks, dim=0)

        face_body_subset = {}
        for key in ['positions', 'opacity', 'scales', 'rotations', 'colors']:
            face_body_subset[key] = body_gaussians[key][face_body_indices]

        src_pos = face_body_subset['positions']
        tgt_pos = face_positions_cano
        k = 4
        knn_dists_list = []
        knn_idx_list = []
        with torch.no_grad():
            for i in range(0, tgt_pos.shape[0], chunk_size):
                tgt_chunk = tgt_pos[i:i + chunk_size]
                dists_chunk = torch.cdist(tgt_chunk, src_pos)
                d_chunk, idx_chunk = dists_chunk.topk(k, dim=1, largest=False)
                knn_dists_list.append(d_chunk)
                knn_idx_list.append(idx_chunk)
            knn_dists = torch.cat(knn_dists_list, dim=0)
            knn_idx = torch.cat(knn_idx_list, dim=0)
            weights = 1.0 / (knn_dists + 1e-8)
            weights = weights / weights.sum(dim=1, keepdim=True)

        def interp(attr):
            vals = attr[knn_idx]
            while vals.dim() < 3:
                vals = vals.unsqueeze(-1)
            w = weights.unsqueeze(-1)
            while w.dim() < vals.dim():
                w = w.unsqueeze(-1)
            out = (w * vals).sum(1)
            return out  # keep (N, 1) for opacity so it matches render_face(..., render=False)

        face_gaussians = {
            'positions': tgt_pos.detach().clone(),
            'opacity': interp(face_body_subset['opacity']),
            'scales': interp(face_body_subset['scales']),
            'rotations': interp(face_body_subset['rotations']),
            'colors': interp(face_body_subset['colors']),
            'max_sh_degree': body_gaussians['max_sh_degree']
        }
        return face_gaussians

    def load_canonical_face_gaussian_model(self, items):
        """
        Set the canonical face gaussian model in self.face_gaussian_model by masking and interpolating the canonical body gaussians.
        Uses 2x upsampled face map (same as __init__) so face has 4x the gaussians as the raw face patch.
        This overwrites the gaussians in self.face_gaussian_model in-place.
        """
        # Load canonical body gaussians (in canonical space)
        cano_map = self.cano_smpl_map              # [H, W, 3]
        face_U, face_V1, face_V2, face_d = self.face_U, self.face_V1, self.face_V2, self.face_d
        body_gaussians = self.render(items, render=False)['canonical_gaussians']
        # Build face region and upsample 2x (same as __init__) to get 4x the number of face positions
        face_cano_raw = torch.concat([
            cano_map[face_U - face_d:face_U + face_d, face_V1 - face_d:face_V1 + face_d],
            cano_map[face_U - face_d:face_U + face_d, face_V2 - face_d:face_V2 + face_d],
        ], 1)
        face_cano_smpl_map = interpolate_image_masked(
            face_cano_raw,
            target_h=2 * face_cano_raw.shape[0],
            target_w=2 * face_cano_raw.shape[1],
            eps=1e-6,
            threshold=0.5,
            fill=0.0,
        )
        face_cano_smpl_mask = torch.linalg.norm(face_cano_smpl_map, dim=-1) > 0.0
        face_positions_cano = face_cano_smpl_map[face_cano_smpl_mask]  # (Nf_face, 3)
        body_positions_cano = body_gaussians['positions']  # (Nb, 3)
        # For each face_position, find the nearest body gaussian (chunked to avoid OOM on full cdist)
        chunk_size = 2048
        nearest_chunks = []
        with torch.no_grad():
            for i in range(0, face_positions_cano.shape[0], chunk_size):
                chunk = face_positions_cano[i:i + chunk_size]  # (chunk_size, 3)
                dists_chunk = torch.cdist(chunk, body_positions_cano)  # (chunk_size, Nb)
                nearest_chunks.append(dists_chunk.argmin(dim=1))
            face_body_indices = torch.cat(nearest_chunks, dim=0)  # (Nf_face,)

        # Gather those belonging to the face from the body gaussians
        face_body_gaussians = {}
        for key in ['positions', 'opacity', 'scales', 'rotations', 'colors']:
            attr = body_gaussians[key][face_body_indices]
            face_body_gaussians[key] = attr   # (Nf_face, ...)

        # For each face_position, interpolate or copy params from the extracted subset above (knn); chunked to avoid OOM
        src_pos = face_body_gaussians['positions']
        tgt_pos = face_positions_cano
        k = 4
        knn_dists_list = []
        knn_idx_list = []
        with torch.no_grad():
            for i in range(0, tgt_pos.shape[0], chunk_size):
                tgt_chunk = tgt_pos[i:i + chunk_size]
                dists_chunk = torch.cdist(tgt_chunk, src_pos)
                d_chunk, idx_chunk = dists_chunk.topk(k, dim=1, largest=False)
                knn_dists_list.append(d_chunk)
                knn_idx_list.append(idx_chunk)
            knn_dists = torch.cat(knn_dists_list, dim=0)   # (Nf_face, k)
            knn_idx = torch.cat(knn_idx_list, dim=0)       # (Nf_face, k)
            weights = 1.0 / (knn_dists + 1e-8)
            weights = weights / weights.sum(dim=1, keepdim=True)  # (Nf_face, k)

        def interp(attr):
            vals = attr[knn_idx]              # (Nf_face, k, ...)
            while vals.dim() < 3:
                vals = vals.unsqueeze(-1)
            # Expand weights to broadcast with vals
            w = weights.unsqueeze(-1)
            while w.dim() < vals.dim():
                w = w.unsqueeze(-1)
            out = (w * vals).sum(1)
            return out.squeeze(-1) if out.shape[-1] == 1 else out

        # Interpolating attributes (except colors)
        face_attrs = {
            'positions': tgt_pos.detach().clone(),
            'opacity': interp(face_body_gaussians['opacity']).unsqueeze(-1),
            'scales': interp(face_body_gaussians['scales']),
            'rotations': interp(face_body_gaussians['rotations']),
            'max_sh_degree': body_gaussians['max_sh_degree']
        }

        # Colors: use DC + higher frequency from nearest body point only
        nearest_idxs = knn_idx[:, 0]  # (Nf_face,)
        face_attrs['colors'] = face_body_gaussians['colors'][nearest_idxs]

        max_sh_degree = body_gaussians['max_sh_degree']

        # Set self.face_gaussian_model values accordingly
        self.face_gaussian_model.set_gaussians(
            xyz=face_attrs['positions'],
            opacity=face_attrs['opacity'],
            scaling=face_attrs['scales'],
            rotation=face_attrs['rotations'],
            colors=face_attrs['colors'].unsqueeze(1),
            sh_degree=max_sh_degree
        )

        # Optionally, return face_attrs dict
        return face_attrs