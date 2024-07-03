import radiomics
from radiomics import featureextractor, getTestCase, logging
import SimpleITK as sitk
import six
import os
import numpy as np
import json

radiomics.logger.setLevel(60) 

path = './'
csv_dict = 'meta_622n_balanced6k_filterissuecase'
orig = False
parameters = 'myParams.yaml'
output_path = 'radiomic_features_Params_orig'
parts = ['train', 'val', 'test'] #['train', 'val', 'test']

data_dict = {}
data_dict['train'], data_dict['val'], data_dict['test'] = {}, {}, {}
data_dict['train']['images'], data_dict['val']['images'], data_dict['test']['images'] = [], [], []
data_dict['train']['masks'], data_dict['val']['masks'], data_dict['test']['masks'] = [], [], []

# data loading
datalist = os.path.join(path, 'data', 'Meta', f'{csv_dict}.csv')
with open(datalist, "r") as f:
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

        if part == 'Train':
            data_dict['train']['images'].append(img_path)
            data_dict['train']['masks'].append(mask_path)
        elif part == "Validation":
            data_dict['val']['images'].append(img_path)
            data_dict['val']['masks'].append(mask_path)
        elif part == "Test":
            data_dict['test']['images'].append(img_path)
            data_dict['test']['masks'].append(mask_path)

        i += 1

print(f"Total images: Train {len(data_dict['train']['images'])} Val {len(data_dict['val']['images'])} Test {len(data_dict['test']['images'])}")
print(f"Total masks: Train {len(data_dict['train']['masks'])} Val {len(data_dict['val']['masks'])} Test {len(data_dict['test']['masks'])}")

# extracting features
extractor = featureextractor.RadiomicsFeatureExtractor(parameters)
for part in parts:
    print(f"Processing {part}...")

    image_paths = data_dict[f'{part}']['images']
    mask_paths = data_dict[f'{part}']['masks']
    i = 0
    for ip, mp in zip(image_paths, mask_paths):
        print(f"Case {i+1} at {ip}...")
        if ip.endswith('.npy') or mp.endswith('.npy'):
            if ip.endswith('.npy'):
                image = sitk.GetImageFromArray(np.load(ip))
                image.SetSpacing((1.0, 1.0, 1.0))
                image = sitk.JoinSeries(image)
            if mp.endswith('.npy'): 
                mask = sitk.GetImageFromArray(np.load(mp).astype(np.uint8))
                mask.SetSpacing((1.0, 1.0, 1.0))
                mask = sitk.JoinSeries(mask)
                
            try:
                result = extractor.execute(image, mask, voxelBased=True)
            except:
                print(f"Only contain {(np.load(mp).astype(np.uint8)).sum()} voxels for mask!")
        else:
            try:
                result = extractor.execute(ip, mp, voxelBased=True)
            except:
                print(f"Only contain {(np.load(mp).astype(np.uint8)).sum()} voxels for mask!")

        info = {}
        for k, v in six.iteritems(result):
            save_path = os.path.splitext(ip)[0].replace('data', output_path)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if isinstance(v, sitk.Image):
                sitk.WriteImage(v, os.path.join(save_path, k + '.nrrd'), True)
            else:
                print(f"{k}: {v}")
                info[k] = v
                continue
        
        with open(os.path.join(save_path, "info.json"), "w") as f:
            json.dump(info, f, indent=4)

        i += 1
