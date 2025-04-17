import os
import os.path as osp
import random
from typing import Any, Dict

import torch
import torch.cuda.amp as amp
import torch.nn.functional as F

from video_to_video.modules import *
from video_to_video.utils.config import cfg
from video_to_video.diffusion.diffusion_sdedit import GaussianDiffusion
from video_to_video.diffusion.schedules_sdedit import noise_schedule
from video_to_video.utils.logger import get_logger

from diffusers import AutoencoderKLTemporalDecoder

logger = get_logger()

class VideoToVideo_sr():
    def __init__(self, opt, device=torch.device('cuda:0')):
        # Define primary (GPU 0) and secondary (GPU 1) devices
        device0 = torch.device('cuda:0')
        device1 = torch.device('cuda:1')
        self.device = device0            # U-Net and diffusion on GPU 0
        self.device1 = device1          # VAE and CLIP on GPU 1

        self.opt = opt

        # Text encoder (CLIP) on GPU 1
        text_encoder = FrozenOpenCLIPEmbedder(device=self.device1, 
                                             pretrained="laion2b_s32b_b79k")
        text_encoder.model.to(self.device1)         # move CLIP model to GPU 1
        self.text_encoder = text_encoder
        logger.info(f'Build encoder with FrozenOpenCLIPEmbedder')

        # U-Net with ControlNet on GPU 0
        generator = ControlledV2VUNet()
        generator = generator.to(self.device)       # U-Net to GPU 0
        generator.eval()
        cfg.model_path = opt.model_path
        load_dict = torch.load(cfg.model_path, map_location='cpu')
        if 'state_dict' in load_dict:
            load_dict = load_dict['state_dict']
        ret = generator.load_state_dict(load_dict, strict=False)
        self.generator = generator.half()           # U-Net in FP16
        logger.info(f'Load model path {cfg.model_path}, with local status {ret}')

        # Noise scheduler (no change)
        sigmas = noise_schedule(schedule='logsnr_cosine_interp', n=1000, 
                                 zero_terminal_snr=True, scale_min=2.0, scale_max=4.0)
        diffusion = GaussianDiffusion(sigmas=sigmas)
        self.diffusion = diffusion
        logger.info('Build diffusion with GaussianDiffusion')

        # Temporal VAE on GPU 1
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid", subfolder="vae", variant="fp16"
        )
        vae.eval().requires_grad_(False)
        vae.to(self.device1)                       # move VAE to GPU 1
        self.vae = vae
        logger.info('Build Temporal VAE')
        # Free cached memory on both GPUs after loading models
        torch.cuda.empty_cache()
        with torch.cuda.device(self.device1):
            torch.cuda.empty_cache()

        # Prepare negative text embedding on GPU 1, then move it to GPU 0 for later use
        self.negative_prompt = cfg.negative_prompt
        self.positive_prompt = cfg.positive_prompt
        negative_y = text_encoder(self.negative_prompt).detach()      # on GPU 1
        negative_y = negative_y.to(self.device)                      # move to GPU 0
        self.negative_y = negative_y
        logger.info('Negative prompt encoded and transferred to GPU 0')


    def test(self, input: Dict[str, Any], total_noise_levels=1000, 
             steps=50, solver_mode='fast', guide_scale=7.5, max_chunk_len=32):
        video_data = input['video_data']        # input frames (tensor)
        y = input['y']                          # input prompt (string or embedding)
        (target_h, target_w) = input['target_res']

        # Preprocess video frames (resize and pad)
        video_data = F.interpolate(video_data, [target_h, target_w], mode='bilinear')
        logger.info(f'video_data shape: {video_data.shape}')
        frames_num, _, h, w = video_data.shape
        padding = pad_to_fit(h, w)
        video_data = F.pad(video_data, padding, 'constant', 1)
        video_data = video_data.unsqueeze(0)

        bs = 1
        # Move input video to GPU 1 for VAE encoding
        video_data = video_data.to(self.device1)
        video_data_feature = self.vae_encode(video_data)        # encode frames on GPU 1
        # Transfer VAE latents to GPU 0 for diffusion
        video_data_feature = video_data_feature.to(self.device)
        # Free VAE encode caches on both GPUs
        torch.cuda.empty_cache()
        with torch.cuda.device(self.device1):
            torch.cuda.empty_cache()

        # Encode text prompt on GPU 1, then move to GPU 0 for guidance
        y = self.text_encoder(y).detach()       # CLIP text encoding on GPU 1 (no grad)
        y = y.to(self.device)                   # move text embedding to GPU 0

        # Prepare diffusion (U-Net) inputs
        with torch.cuda.amp.autocast(enabled=True):
            t = torch.LongTensor([total_noise_levels - 1]).to(self.device)
            noised_lr = self.diffusion.diffuse(video_data_feature, t)  # add noise to latent

            # Model kwargs: positive and negative text embeddings, plus hint (latent frames)
            model_kwargs = [{'y': y}, {'y': self.negative_y}]
            model_kwargs.append({'hint': video_data_feature})
        # Free cached memory before diffusion sampling (both GPUs)
        torch.cuda.empty_cache()
        with torch.cuda.device(self.device1):
            torch.cuda.empty_cache()

        # Diffusion sampling on GPU 0 (U-Net in FP16)
        chunk_inds = make_chunks(frames_num, interp_f_num=0, max_chunk_len=max_chunk_len) \
                     if frames_num > max_chunk_len else None
        solver = 'dpmpp_2m_sde'  # sampler type
        gen_vid = self.diffusion.sample_sr(
            noise=noised_lr,
            model=self.generator,
            model_kwargs=model_kwargs,
            guide_scale=guide_scale,
            guide_rescale=0.2,
            solver=solver,
            solver_mode=solver_mode,
            return_intermediate=None,
            steps=steps,
            t_max=total_noise_levels - 1,
            t_min=0,
            discretization='trailing',
            chunk_inds=chunk_inds,
        )
        # Free diffusion caches on both GPUs
        torch.cuda.empty_cache()
        with torch.cuda.device(self.device1):
            torch.cuda.empty_cache()
        logger.info('sampling finished.')

        # Move generated latents to GPU 1 for VAE decoding
        gen_vid = gen_vid.to(self.device1)
        vid_tensor_gen = self.vae_decode_chunk(gen_vid, chunk_size=3)   # decode on GPU 1
        logger.info('temporal VAE decoding finished.')

        # Remove padding from the output frames
        w1, w2, h1, h2 = padding
        vid_tensor_gen = vid_tensor_gen[:, :, h1 : h + h1, w1 : w + w1]
        # Reshape back to (batch, channels, frames, height, width)
        gen_video = rearrange(vid_tensor_gen, '(b f) c h w -> b c f h w', b=bs)
        # Free decode caches on both GPUs
        torch.cuda.empty_cache()
        with torch.cuda.device(self.device1):
            torch.cuda.empty_cache()

        # Return the result on CPU (convert to float32 for safety)
        return gen_video.type(torch.float32).cpu()


    def temporal_vae_decode(self, z, num_f):
        return self.vae.decode(z/self.vae.config.scaling_factor, num_frames=num_f).sample

    def vae_decode_chunk(self, z, chunk_size=3):
        z = rearrange(z, "b c f h w -> (b f) c h w")
        video = []
        for ind in range(0, z.shape[0], chunk_size):
            num_f = z[ind:ind+chunk_size].shape[0]
            video.append(self.temporal_vae_decode(z[ind:ind+chunk_size],num_f))
        video = torch.cat(video)
        return video

    def vae_encode(self, t, chunk_size=1):
        num_f = t.shape[1]
        t = rearrange(t, "b f c h w -> (b f) c h w")
        z_list = []
        for ind in range(0,t.shape[0],chunk_size):
            z_list.append(self.vae.encode(t[ind:ind+chunk_size]).latent_dist.sample())
        z = torch.cat(z_list, dim=0)
        z = rearrange(z, "(b f) c h w -> b c f h w", f=num_f)
        return z * self.vae.config.scaling_factor
    

def pad_to_fit(h, w):
    BEST_H, BEST_W = 720, 1280

    if h < BEST_H:
        h1, h2 = _create_pad(h, BEST_H)
    elif h == BEST_H:
        h1 = h2 = 0
    else: 
        h1 = 0
        h2 = int((h + 48) // 64 * 64) + 64 - 48 - h

    if w < BEST_W:
        w1, w2 = _create_pad(w, BEST_W)
    elif w == BEST_W:
        w1 = w2 = 0
    else:
        w1 = 0
        w2 = int(w // 64 * 64) + 64 - w
    return (w1, w2, h1, h2)

def _create_pad(h, max_len):
    h1 = int((max_len - h) // 2)
    h2 = max_len - h1 - h
    return h1, h2


def make_chunks(f_num, interp_f_num, max_chunk_len, chunk_overlap_ratio=0.5):
    MAX_CHUNK_LEN = max_chunk_len
    MAX_O_LEN = MAX_CHUNK_LEN * chunk_overlap_ratio
    chunk_len = int((MAX_CHUNK_LEN-1)//(1+interp_f_num)*(interp_f_num+1)+1)
    o_len = int((MAX_O_LEN-1)//(1+interp_f_num)*(interp_f_num+1)+1)
    chunk_inds = sliding_windows_1d(f_num, chunk_len, o_len)
    return chunk_inds


def sliding_windows_1d(length, window_size, overlap_size):
    stride = window_size - overlap_size
    ind = 0
    coords = []
    while ind<length:
        if ind+window_size*1.25>=length:
            coords.append((ind,length))
            break
        else:
            coords.append((ind,ind+window_size))
            ind += stride  
    return coords
