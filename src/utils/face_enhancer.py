import os
import torch 

from gfpgan import GFPGANer

from tqdm import tqdm

from src.utils.videoio import load_video_to_cv2

import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


class GeneratorWithLen(object):
    """ From https://stackoverflow.com/a/7460929 """

    def __init__(self, gen, length):
        self.gen = gen
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.gen

def enhancer_list(images, method='gfpgan', bg_upsampler='realesrgan'):
    gen = enhancer_generator_no_len(images, method=method, bg_upsampler=bg_upsampler)
    return list(gen)

def enhancer_generator_with_len(images, method='gfpgan', bg_upsampler='realesrgan'):
    """ Provide a generator with a __len__ method so that it can passed to functions that
    call len()"""

    if os.path.isfile(images): # handle video to images
        # TODO: Create a generator version of load_video_to_cv2
        images = load_video_to_cv2(images)

    gen = enhancer_generator_no_len(images, method=method, bg_upsampler=bg_upsampler)
    gen_with_len = GeneratorWithLen(gen, len(images))
    return gen_with_len

def enhancer_generator_no_len(images, method='gfpgan', bg_upsampler='realesrgan'):
    """ Provide a generator function so that all of the enhanced images don't need
    to be stored in memory at the same time. This can save tons of RAM compared to
    the enhancer function. """

    outscale = 2 # TODO mj hard code
    print('face enhancer....')
    if not isinstance(images, list) and os.path.isfile(images): # handle video to images
        images = load_video_to_cv2(images)

    # ------------------------ set up GFPGAN restorer ------------------------
    if method != None:
        if  method.lower() == 'gfpgan':
            arch = 'clean'
            channel_multiplier = 2
            model_name = 'GFPGANv1.4'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
        elif method.lower() == 'restoreformer':
            arch = 'RestoreFormer'
            channel_multiplier = 2
            model_name = 'RestoreFormer'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
        elif method.lower() == 'codeformer': # TODO:
            arch = 'CodeFormer'
            channel_multiplier = 2
            model_name = 'CodeFormer'
            url = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'
        else:
            arch = None
            channel_multiplier = 2
            model_name = None
            url = None
            # raise ValueError(f'Wrong model version {method}.')


    # ------------------------ set up background upsampler ------------------------
    # if bg_upsampler == 'realesrgan':
    if bg_upsampler != None and bg_upsampler.lower().startswith('realesr'):
        
        if not torch.cuda.is_available():  # CPU
            import warnings
            warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                          'If you really want to use it, please modify the corresponding codes.')
            upsampler = None
        else:
            # model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            # upsampler = RealESRGANer(
            #     scale=2,
            #     model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
            #     model=model,
            #     tile=400,
            #     tile_pad=10,
            #     pre_pad=0,
            #     half=True)  # need to set False in CPU mode
            upsampler = upsamplerModel(bg_upsampler)
    else:
        upsampler = None

    if model_name != None:
        # determine model paths
        model_path = os.path.join('gfpgan/weights', model_name + '.pth')
        
        if not os.path.isfile(model_path):
            model_path = os.path.join('checkpoints', model_name + '.pth')
        
        if not os.path.isfile(model_path):
            # download pre-trained models from url
            model_path = url

        restorer = GFPGANer(
            model_path=model_path,
            upscale=2,
            arch=arch,
            channel_multiplier=channel_multiplier,
            bg_upsampler=upsampler)

        # ------------------------ restore ------------------------
        for idx in tqdm(range(len(images)), 'Face Enhancer:'):
            
            img = cv2.cvtColor(images[idx], cv2.COLOR_RGB2BGR)
            
            # restore faces and background if necessary
            cropped_faces, restored_faces, r_img = restorer.enhance(
                img,
                has_aligned=False,
                only_center_face=False,
                paste_back=True)
            
            r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)
            yield r_img
    # elif bg_upsampler == 'realesrgan':
    elif bg_upsampler != None and bg_upsampler.lower().startswith('realesr'):
        for idx in tqdm(range(len(images)), 'Full image Enhancer:'):
            
            img = cv2.cvtColor(images[idx], cv2.COLOR_RGB2BGR)
            
            # restore faces and background if necessary
            r_img, _ = upsampler.enhance(img, outscale=outscale)
            # cropped_faces, restored_faces, r_img = restorer.enhance(
            #     img, 
            #     has_aligned=False,
            #     only_center_face=False,
            #     paste_back=True)
            
            r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)
            yield r_img
            
        # else:
        #     raise ValueError(f'Wrong enhancer model version {method} and background_enhancer model {bg_upsampler}.')
     
def upsamplerModel(model_name):
    if model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    elif model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    elif model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
    elif model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    elif model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
    elif model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]
    
    model_path = os.path.join('gfpgan/weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        for url in file_url:
            # model_path will be updated
            model_path = load_file_from_url(
                url=url, model_dir='gfpgan/weights', progress=True, file_name=None)
            

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        # dni_weight=dni_weight,
        model=model,
        tile=400,
        tile_pad=10,
        pre_pad=0,
        half=True) # need to set False in CPU mode
    
    return upsampler