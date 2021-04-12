import sys, numpy as np, os
from cellpose import models as cp # cellpose has to be installed in the environment, and the models have to be downloaded in cellpose's favorite directory: ~/.cellpose/models
import pathlib
from urllib.parse import urlparse

class Segmentor2D:
    """ Segments images of nuclei using cellpose"""
    def __init__(self):
#         self.models_dir = model_directory
#         cpmodel_path = [os.path.join(self.models_dir, '{}_{}'.format('nuclei', i)) for i in range(4)]
#         szmodel_path = os.path.join(self.models_dir, 'size_nuclei_0.npy')
#         if not os.path.exists(cpmodel_path[0]):
#             print("Cellpose weights not found in {}.".format(self.models_dir))
#             self.download_model_weights(self.models_dir)

        self.base_model = cp.Cellpose(model_type = 'nuclei')
        
        
    def segment(self, imgs, diameters = 40, out_files = None, **kwargs):
        """ Takes a list of images, one or a list of average diameters 
        of nuclei in each image and runs Cellpose and returns masks in a list
        If diameter is None, run cellpose with automatic diameter detection and also returns estimated diameters
        Optional: out_files is a list of addresses to save the masks. """
        
        if len(imgs[0].shape) == 3:
            print("Images are 3D. TODO: 3D erosion")
            raise TypeError('3D image loaded instead of 2D')
        initial_masks = []
        
        if diameters is None:
            estim_diams = []
            for img in imgs:
                mask, _, _, diam = self.base_model.eval([img], channels = [0, 0])
                if mask[0].max() < 2**16 - 1:
                    initial_masks.append(mask[0].astype(np.uint16))
                estim_diams.append(diam)
            toReturn = initial_masks, estim_diams
        else:
            if not isinstance(diameters, list):
                diameters = len(imgs) * [diameters]
            for i, img in enumerate(imgs):
                mask, _, _, _ = self.base_model.eval([img], channels = [0, 0], diameter = diameters[i], **kwargs)
                if mask[0].max() < 2**16 - 1:
                    initial_masks.append(mask[0].astype(np.uint16))
            toReturn = initial_masks

        if out_files is not None:
            self.save_masks(initial_masks, out_files)
            
        return toReturn

    def save_masks(self, masks, files):
        for i, (file, mask) in enumerate(zip(files, masks)):
            print('Saving mask {0} in {1}.'.format(i, file))
            np.save(file, mask)
            
    def download_model_weights(self, models_dir, urls = cp.urls):
        print(cp.urls)
        print(urls)
        """ modified version of cellpose/models/download_model_weights """
        # cellpose directory
        #     cp_dir = pathlib.Path.home().joinpath('.cellpose')
        #     cp_dir.mkdir(exist_ok=True)
        #     model_dir = cp_dir.joinpath('models')
        #     model_dir.mkdir(exist_ok=True)
        if not os.path.exists(models_dir):
            os.mkdir(models_dir)    
        for url in urls:
            parts = urlparse(url)
            filename = os.path.basename(parts.path)
            cached_file = os.path.join(models_dir, filename)
            if not os.path.exists(cached_file):
                sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
                cp.download_url_to_file(url, cached_file, progress=True)
