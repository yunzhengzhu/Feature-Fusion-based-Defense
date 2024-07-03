import numpy as np
from sklearn.model_selection import train_test_split
import random
import os
from scipy import ndimage
import torch
from torch.utils.data import Dataset
from feature_utils2d import mindssc
from PIL import ImageEnhance, Image
import json
import SimpleITK as sitk

def to_categorical(labels, num_classes=None):
    labels = np.array(labels, dtype=int)
    if num_classes is None:
        num_classes = np.max(labels) + 1
    categorical = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    categorical[np.arange(labels.shape[0]), np.squeeze(labels)] = 1
    return categorical


class LIDC_Dataset(Dataset):
    def __init__(self, path, datalist, mode="Train", crop_type="roi_center", aug=False, orig=False, clip_range=None, crop_size=(224, 224), input_shape=(224, 224), feat_extract=None, feat_groups_categories=None): 
        self.path = path
        self.orig = orig
        self.datalist = self.csv_dataset_to_data(path, datalist, orig=orig, mode=mode)
        self.crop_type = crop_type
        self.crop_size = crop_size
        self.clip_range = clip_range
        self.aug = aug
        self.input_shape = input_shape
        self.feat_extract = feat_extract
        self.feat_groups_categories = feat_groups_categories

    def load_data(self, onehot=False):
        '''
        for attack loader only
        '''
        img_list, mask_list, label_list = [], [], []
        feat_groups_list = []
        for i in range(len(self.datalist['image'])):
            img_path = self.datalist['image'][i]
            mask_path = self.datalist['mask'][i]
            label = self.datalist['label'][i]

            img = np.load(img_path)
            orig_img_shape = img.shape
            mask = np.load(mask_path)

            cx, cy = self.center_of_mass(mask)
            img = self.data_preprocess(img, crop_size=self.crop_size, crop_type=self.crop_type, crop_center=(round(cx), round(cy)), normalize=True, clip_range=self.clip_range)
            mask = self.data_preprocess(mask, crop_size=self.crop_size, crop_type=self.crop_type, crop_center=(round(cx), round(cy)), normalize=False)

            if self.aug:
                img, aug_params = self.apply_augmentation(img)
                mask = self.apply_augmentation(mask, aug_params)[0]
            else:
                aug_params = None

            # convert integers to floats
            img = img.astype('float32')
            mask = mask.astype('float32')
            label = np.array(label).astype('float32')

            if img.shape != self.input_shape:
                img = ndimage.zoom(
                    img,
                    zoom=(
                        self.input_shape[0]/img.shape[0],
                        self.input_shape[1]/img.shape[1],
                    ),
                    order=0,
                )
        
            # feature extraction
            if self.feat_extract is not None:
                normalize = False
                feat_groups = {}
                for feat_ext in self.feat_extract:
                    if feat_ext == "mindssc":
                        this_feat_groups = {"mindssc": mindssc(torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float(), delta=1, sigma=1).squeeze(0).numpy()}
                        this_feat_groups = self.listdict2torchdict(this_feat_groups)
                    if feat_ext == "grad":
                        dxx, dxy = np.gradient(img)
                        if normalize:
                            min_vals_dxx = np.min(dxx, axis=(0, 1), keepdims=True)
                            max_vals_dxx = np.max(dxx, axis=(0, 1), keepdims=True)
                            dxx = (dxx - min_vals_dxx) / (max_vals_dxx - min_vals_dxx)

                            min_vals_dxy = np.min(dxy, axis=(0, 1), keepdims=True)
                            max_vals_dxy = np.max(dxy, axis=(0, 1), keepdims=True)
                            dxy = (dxy - min_vals_dxy) / (max_vals_dxy - min_vals_dxy)

                        this_feat_groups = {"grad": torch.cat((torch.from_numpy(dxx[None, ...]), torch.from_numpy(dxy[None, ...])), dim=0).numpy()}
                        this_feat_groups = self.listdict2torchdict(this_feat_groups)
                    if feat_ext == "pyradiomic":
                        this_feat_groups = self.load_pyradiomic_features(os.path.splitext(img_path.replace('data', 'radiomic_features_Params_orig'))[0], (cx, cy), crop_size=self.crop_size, shape=orig_img_shape, normalize=normalize, aug_params=aug_params, feat_groups_categories=self.feat_groups_categories)
                    if feat_ext == "pyradiomic-wavelet":
                        this_feat_groups = self.load_pyradiomic_features(os.path.splitext(img_path.replace('data', 'radiomic_features_Params_wavelet'))[0], (cx, cy), crop_size=self.crop_size, shape=orig_img_shape, normalize=normalize, aug_params=aug_params, feat_groups_categories=self.feat_groups_categories)
                    feat_groups.update(this_feat_groups)
                feat_groups = torch.cat([v for k, v in feat_groups.items()], dim=0)
            
            x = torch.from_numpy(img[None, ...])
            if self.feat_extract is not None:
                x = torch.cat((x, feat_groups), dim=0)
            
            img = x.detach().numpy()

            img_list.append(img)
            mask_list.append(mask[None, ...])
            label_list.append(label)

        if onehot:
            label_list = to_categorical(label_list)

        if self.feat_extract:
            return np.array(img_list), np.array(label_list), np.array(mask_list), np.array(feat_groups_list)
        else:
            return np.array(img_list), np.array(label_list), np.array(mask_list)



    def __getitem__(self, idx):
        img_path = self.datalist['image'][idx]
        mask_path = self.datalist['mask'][idx]
        label = self.datalist['label'][idx]
        
        img = np.load(img_path)
        orig_img_shape = img.shape
        mask = np.load(mask_path)
        
        cx, cy = self.center_of_mass(mask)
        img = self.data_preprocess(img, crop_size=self.crop_size, crop_type=self.crop_type, crop_center=(round(cx), round(cy)), normalize=True, clip_range=self.clip_range)

        mask = self.data_preprocess(mask, crop_size=self.crop_size, crop_type=self.crop_type, crop_center=(round(cx), round(cy)), normalize=False)

        if self.aug:
            img, aug_params = self.apply_augmentation(img)
            mask = self.apply_augmentation(mask, aug_params)[0]
        else:
            aug_params = None

        # convert integers to floats
        img = img.astype('float32')
        mask = mask.astype('float32')
        label = np.array(label).astype('float32')

        if img.shape != self.input_shape:
            img = ndimage.zoom(
                img, 
                zoom=(
                    self.input_shape[0]/img.shape[0],
                    self.input_shape[1]/img.shape[1],
                ),
                order=0,
            )
        
        
        # feature extraction
        if self.feat_extract is not None:
            normalize = False
            feat_groups = {}
            for feat_ext in self.feat_extract:
                if feat_ext == "mindssc":
                    this_feat_groups = {"mindssc": mindssc(torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float(), delta=1, sigma=1., normalize=normalize).squeeze(0).numpy()}
                    this_feat_groups = self.listdict2torchdict(this_feat_groups)
                    
                if feat_ext == "grad":
                    dxx, dxy = np.gradient(img)
                    if normalize:
                        min_vals_dxx = np.min(dxx, axis=(0, 1), keepdims=True)
                        max_vals_dxx = np.max(dxx, axis=(0, 1), keepdims=True)
                        dxx = (dxx - min_vals_dxx) / (max_vals_dxx - min_vals_dxx)

                        min_vals_dxy = np.min(dxy, axis=(0, 1), keepdims=True)
                        max_vals_dxy = np.max(dxy, axis=(0, 1), keepdims=True)
                        dxy = (dxy - min_vals_dxy) / (max_vals_dxy - min_vals_dxy)

                    this_feat_groups = {"grad": torch.cat((torch.from_numpy(dxx[None, ...]), torch.from_numpy(dxy[None, ...])), dim=0).numpy()}
                    this_feat_groups = self.listdict2torchdict(this_feat_groups)
                if feat_ext == "pyradiomic":
                    this_feat_groups = self.load_pyradiomic_features(os.path.splitext(img_path.replace('data', 'radiomic_features_Params_orig'))[0], (cx, cy), crop_size=self.crop_size, shape=orig_img_shape, normalize=normalize, aug_params=aug_params, feat_groups_categories=self.feat_groups_categories)
                if feat_ext == "pyradiomic-wavelet":
                    this_feat_groups = self.load_pyradiomic_features(os.path.splitext(img_path.replace('data', 'radiomic_features_Params_wavelet'))[0], (cx, cy), crop_size=self.crop_size, shape=orig_img_shape, normalize=normalize, aug_params=aug_params, feat_groups_categories=self.feat_groups_categories)
                feat_groups.update(this_feat_groups)
        
        x = torch.from_numpy(img[None, ...])

        y = torch.from_numpy(label)
        mask = torch.from_numpy(mask[None, ...])
        
        if self.feat_extract:
            return x, y, mask, feat_groups
        else:
            return x, y, mask

    def __len__(self):
        return len(self.datalist['image'])
  
    def apply_augmentation(self, img, aug_params=None):
        if aug_params is None:
            aug_params = {
                "flip": random.randint(0, 2),
                "rotate_angle": np.random.uniform(-20 ,20),
            }

        # flipping
        if aug_params["flip"] == 1:
            img = np.flip(img, 0).copy()
        elif aug_params["flip"] == 2:
            img = np.flip(img, 1).copy()

        # rotation
        img = ndimage.rotate(img, aug_params["rotate_angle"], reshape=False, mode="nearest")

        return img, aug_params

    def load_pyradiomic_features(self, path, orig_center, crop_size=(40, 40), shape=(512, 512), normalize=True, aug_params=None, feat_groups_categories=None):
        feat_paths = [f for f in os.listdir(path) if f.endswith('.nrrd')]

        feat_json = json.load(open(os.path.join(path, 'info.json'), 'r'))
        centerofmass = feat_json['diagnostics_Mask-original_CenterOfMassIndex']
        cy = round(centerofmass[0])
        cx = round(centerofmass[1])
        if feat_groups_categories is None:
            groups = {'firstorder': [], 'glcm': [], 'gldm': [], 'glrlm': [], 'glszm': [], 'ngtdm': []}
        else:
            groups = {k: [] for k in feat_groups_categories}

        for fp in feat_paths:
            this_feat_group = fp.split("_")[-2]
            if this_feat_group in groups.keys():
                precropped_feat = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(path, fp)))[0]
                crop_x, crop_y = precropped_feat.shape
                # resample feat
                this_feat = np.zeros(shape)
                this_feat[
                    cx - crop_x // 2:cx + (crop_x - crop_x // 2),
                    cy - crop_y // 2:cy + (crop_y - crop_y // 2)
                ] = precropped_feat
                
                crop_feat = self.data_preprocess(this_feat, crop_size=(40, 40), crop_type="roi_center", crop_center=(round(orig_center[0]), round(orig_center[1])), normalize=normalize, clip_range=None)
                if aug_params is not None:
                    crop_feat = self.apply_augmentation(crop_feat, aug_params)[0]
                
                if crop_feat.shape != self.input_shape:
                    crop_feat = ndimage.zoom(
                        crop_feat, 
                        zoom=(
                            self.input_shape[0]/crop_feat.shape[0],
                            self.input_shape[1]/crop_feat.shape[1],
                        ),
                        order=0,
                    )
            
                groups[this_feat_group].append(crop_feat)
            else:
                continue

        # convert group of features into torch tensors
        groups_th = self.listdict2torchdict(groups)
        return groups_th


    def data_preprocess(self, img, crop_size=(224, 224), crop_type="rand", crop_center=None, normalize=False, clip_range=None):
        h, w = img.shape
        ch, cw = crop_size
        
        if clip_range is not None:
            min_bound = clip_range[0]
            max_bound = clip_range[1]
            img[img < min_bound] = min_bound
            img[img > max_bound] = max_bound

        if normalize:
            img = (img - img.min()) / (img.max() - img.min())

        # center crop and resize
        if crop_size != None and (h > ch or w > cw):
            img = self._crop(img, crop_size, crop_type=crop_type, crop_center=crop_center)

        return img

    def _crop(self, img, crop_size, crop_type="rand", crop_center=None):
        h, w = img.shape
        crop_h, crop_w = crop_size

        if crop_h > h or crop_w > w:
            raise ValueError("Crop size should be smaller than the image size")

        if crop_center is not None:
            roi_h, roi_w = crop_center

        if crop_type == "rand":
            start_h = np.random.randint(crop_h, h - crop_h + 1)
            start_w = np.random.randint(crop_w, w - crop_w + 1)
        elif crop_type == "center":
            start_h = (h - crop_h) // 2
            start_w = (w - crop_w) // 2
        elif crop_type == "roi_center":
            start_h = max(roi_h - crop_h // 2, 0)
            start_w = max(roi_w - crop_w // 2, 0)
            end_h = min(roi_h + crop_h // 2, h)
            end_w = min(roi_w + crop_w // 2, w)
        else:
            raise NotImplementedError

        if crop_type == "rand" or crop_type == "center":
            cropped_img = img[
                start_h:start_h + crop_h,
                start_w:start_w + crop_w,
            ]
        else:
            cropped_img = img[
                start_h:end_h,
                start_w:end_w,
            ]
            if cropped_img.shape != crop_size:
                pad_start_h = max(0, crop_h // 2 - roi_h)
                pad_end_h = max(0, roi_h + crop_h // 2 - h)
                pad_start_w = max(0, crop_w // 2 - roi_w)
                pad_end_w = max(0, roi_w + crop_w // 2 - w)
                padded_img = np.pad(
                    cropped_img, (
                        (pad_start_h, pad_end_h), 
                        (pad_start_w, pad_end_w),
                    ), 
                    mode="constant", 
                    constant_values=0
                )
                cropped_img = padded_img
            
        return cropped_img

    def center_of_mass(self, mask):
        mask = mask.astype(int)
        return np.argwhere(mask).mean(axis=0)

    def csv_dataset_to_data(self, path, csv_dict, orig=False, mode="Train"):
        data_dict = {}
        data_dict['image'], data_dict['mask'], data_dict['label'] = [], [], []
        with open(csv_dict, "r") as f:
            f.readline()
            i = 0
            for line in f:
                elements = line.strip().split(",")
                img = elements[3]
                pid = img.split("_")[0]
                mask = elements[4]
                label = 1 if elements[-4] == "True" else 0
                part = elements[-1]
                
                if orig:
                    img_path = os.path.join(path, 'data', 'Original_Image', f'LIDC-IDRI-{pid}', img + '.npy')
                else:
                    img_path = os.path.join(path, 'data', 'Image', f'LIDC-IDRI-{pid}', img + '.npy')

                mask_path = os.path.join(path, 'data', 'Mask', f'LIDC-IDRI-{pid}', mask + '.npy')

                if part == mode:
                    data_dict['image'].append(img_path)
                    data_dict['label'].append(label)
                    data_dict['mask'].append(mask_path)
               
                i += 1

        return data_dict


    def listdict2torchdict(self, d):
        num_feats = {k: len(v) for k, v in d.items()}
        d_torch = {}
        for feat_name, feat in d.items():
            combined_tensor = torch.empty((num_feats[feat_name], self.input_shape[0], self.input_shape[1]))

            for i, array in enumerate(feat):
                combined_tensor[i] = torch.tensor(array)

            d_torch[feat_name] = combined_tensor

        return d_torch
