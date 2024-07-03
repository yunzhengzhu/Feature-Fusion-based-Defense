import os
import numpy as np
from torchvision import models, datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

from art.estimators.classification import PyTorchClassifier
from torchmetrics import AUROC
from torch.utils.data import DataLoader
import json
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from configs import path, path_fig, path_csv
from data_loader import LIDC_Dataset
from plot_utils import plot_tv, plot_roc_curve, plot_attacks_all
from train_utils import set_seed, seed_worker
from attacks import create_attack, partial_attack
from models import Medical_Vgg_Model

set_seed(1234)

class Options():
    def __init__(self):
        ## data setup
        self.dataset = 'lidc' #'lidc' # dataset options: mnist / lidc
        self.datalist = 'meta_622n_balanced6k_filterissuecase' # datalist file
        self.pretrain = 'imagenet' # pretrain options: 'imagenet' / None
        self.original_data = True # original data or preprocessed data
        self.aug = True # augmentation or not
        self.crop_size = (40, 40) # cropping size for the data default: (40, 40) 
        self.input_shape = (224, 224) # data input size for the model
        self.clip_range = (-1000, 500) # if preprocessed == True, None; o/w default: (-1000, 500)

        self.feat_extract = ['mindssc', 'grad', 'pyradiomic'] #['mindssc', 'grad', 'pyradiomic'] #'mindssc' # mindssc / grad / pyradiomic / pyradiomic-wavelet
        self.feat_groups_categories = ['gldm', 'glszm'] # specify the feature selections for pyradiomics: firstorder / glcm / gldm / glrlm / glszm / ngtdm (gldm and glszm)
        # crop_size and clip_range are referenced from following literature:
        # https://www.sciencedirect.com/science/article/pii/S0957417419300545
        ##################### hyparameter tuning ###################
        ## model setup 
        self.init_lr = 0.0002 # initial learning rate
        self.momentum = 0.9
        self.nesterov = True
        self.feat_embedding_layer = True
        self.image_free = False
        
        ## training setup
        self.exp_name = 'lidc_baseline_orig_ep200_clip-1k500_crop40x40_reshape224x224_aug'
        #self.exp_name = 'lidc_baseline_processed_ep200_clip-1k500_crop40x40_reshape224x224_aug'
        self.exp_name = 'ptfrclean_lidc_adv_pgd_eps3e-3_r0.25_iter10_randinit5_rand_bs128_ep30'
        self.exp_name = 'lidc_baseline_orig_ep200_clip-1k500_crop40x40_reshape224x224_aug_addfeatextractlayer'
        self.exp_name = 'lidc_baseline_processed_ep200_clip-1k500_crop40x40_reshape224x224_aug_addfeatextractlayer'
        self.exp_name = 'lidc_baseline_orig_ep200_clip-1k500_crop40x40_reshape224x224_aug_addfeatextractlayer_mindsscgradgldmglszm'
        self.exp_name = 'ptfr_lidc_baseline_orig_ep200_clip-1k500_crop40x40_reshape224x224_aug_addfeatextractlayer_mindsscgradgldmglszm_adv_pgd_eps3e-3_r0.25_iter10_randint5_rand_bs64_ep30'
       
        ## attack setup
        self.attack = True
        self.attack_portion = None #['img'] #['mindssc']
        self.attack_type = 'pgd' # options: 'fgsm' / 'pgd' / 'bim'
        self.epsilons = [0.001, 0.002, 0.004, 0.006] #, 1/255, 2/255, 4/255, 8/255] #[0.001, 0.002, 0.004, 0.006, 0.012]
        self.eps_step_ratio = 0.25
        self.max_iter = 10
        self.num_random_init = 20
        self.plot_all_attacks = False
        self.save_baseline = True
        self.save_adv_results = True
        ############################################################
        ## evaluation setup
        self.evaluation = True
        self.eval_exp_name = self.exp_name #'lidc_baseline_orig_ep200_clip-1k500_crop40x40_reshape224x224_aug'
        self.load_model = 'best_auroc' # Options: 'last'/'best_acc'/'best_auroc'
                
opt = Options()

# Evaluate on Test data
# load dataset
if opt.dataset == "lidc":
    datalist = os.path.join(path, 'data', 'Meta', f'{opt.datalist}.csv')
    val_dataset = LIDC_Dataset(
        path=path,
        datalist=datalist,
        mode="Validation",
        crop_size=opt.crop_size,
        crop_type="roi_center",
        aug=False,
        orig=opt.original_data,
        clip_range=opt.clip_range,
        input_shape=opt.input_shape,
        feat_extract=opt.feat_extract,
        feat_groups_categories=opt.feat_groups_categories,
    )
    test_dataset = LIDC_Dataset(
        path=path,
        datalist=datalist,
        mode="Test",
        crop_size=opt.crop_size,
        crop_type="roi_center",
        aug=False,
        orig=opt.original_data,
        clip_range=opt.clip_range,
        input_shape=opt.input_shape,
        feat_extract=opt.feat_extract,
        feat_groups_categories=opt.feat_groups_categories,
    )
    num_classes = 2
elif opt.dataset == 'mnist':
    transform = transforms.Compose([
        transforms.Pad(padding=2, fill=0, padding_mode='constant'),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    num_classes = 10
    
if opt.dataset == "lidc":
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    print("Validation dataset size:", len(val_dataset))

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
print("Test dataset size:", len(test_dataset))

for batch in val_loader:
    images, labels = batch[:2]
    feats = batch[-1]
    if opt.feat_extract:
        feat_names = [k for k, v in feats.items()]
        num_feats_each_group = [v.shape[1] for k, v in feats.items()]
        num_feat_groups = len(num_feats_each_group)
    else:
        feat_names = None
        num_feats_each_group = images.shape[1]
        num_feat_groups = None
    break
print("Batch shape:", images.shape)
print("Batch label shape:", labels.shape)
if opt.feat_extract:
    print("Batch feats shape:", {k: v.shape for k, v in feats.items()})
    print("Number feats for each group:", num_feats_each_group)
    print("Number of feature groups:", num_feat_groups)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Medical_Vgg_Model(pretrain=opt.pretrain, num_feat_layers=num_feat_groups, feat_embedding_layer=opt.feat_embedding_layer, in_feats=num_feats_each_group, num_classes=num_classes, image_free=opt.image_free)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
if opt.momentum == 0:
    nesterov = False
else:
    nesterov = opt.nesterov
optimizer = optim.SGD(model.parameters(), lr=opt.init_lr, momentum=opt.momentum, nesterov=nesterov)
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")

load_save_path = os.path.join(path, 'exp', opt.eval_exp_name, opt.dataset)
if opt.load_model == "last":
    model_name = 'vgg16_model.pth'
elif opt.load_model == "best_acc":
    model_name = 'vgg16_bestaccmodel.pth'
elif opt.load_model == "best_auroc":
    model_name = 'vgg16_bestroaucmodel.pth'
else:
    raise ValueError('Incorrect load model type!')

try:
    model_weights = torch.load(
        os.path.join(load_save_path, opt.dataset + f"_{model_name}"),
        map_location='cuda:0',
    )

    model.load_state_dict(model_weights)
    print("loading pretrain model weights Done")
except:
    model = torch.load(
        os.path.join(load_save_path, opt.dataset + f"_{model_name}"),
        map_location='cuda:0',
    )
    print("loading pretrain model Done")

if opt.attack:
    #val_baseline_samples = os.path.join(load_save_path, f"val_baseline.npy")
    test_baseline_samples = os.path.join(load_save_path, f"test_baseline.npy")
    #val_baseline_labels = os.path.join(load_save_path, f"val_baseline_label.npy")
    test_baseline_labels = os.path.join(load_save_path, f"test_baseline_label.npy")
    if opt.dataset == "lidc":
        #if not (os.path.exists(val_baseline_samples) and os.path.exists(val_baseline_labels)):
            #val_data = val_dataset.load_data(onehot=True)
            #x_val, y_val = val_data[:2]
            #if opt.save_baseline:
            #    np.save(val_baseline_samples, x_val)
            #    np.save(val_baseline_labels, y_val)
        #else:
        #    x_val = np.load(val_baseline_samples)
        #    y_val = np.load(val_baseline_labels)

        if not (os.path.exists(test_baseline_samples) and os.path.exists(test_baseline_labels)):
            test_data = test_dataset.load_data(onehot=True)
            x_test, y_test = test_data[:2]
            if opt.save_baseline:
                np.save(test_baseline_samples, x_test)
                np.save(test_baseline_labels, y_test)
        else:
            x_test = np.load(test_baseline_samples)
            y_test = np.load(test_baseline_labels)

            
    else:
        raise NotImplementedError

    #print(f"Valid: data {x_val.shape} label: {y_val.shape}")
    print(f"Test: data {x_test.shape} label: {y_test.shape}")
    classifier = PyTorchClassifier(
        model=model,
        clip_values=opt.clip_range,
        loss=criterion,
        optimizer=optimizer,
        input_shape=opt.input_shape,
        nb_classes=num_classes,
    )
    
    # val
    #predictions_val = classifier.predict(x_val)
    #accuracy_val = np.sum(np.argmax(predictions_val, axis=1) == np.argmax(y_val, axis=1)) / len(y_val)
    #print("\t\t Val Acc (clean): {:.3f}".format(accuracy_val))
   
    #auroc_val = AUROC(task="multiclass", num_classes=num_classes)(
    #    nn.Softmax(dim=1)(torch.from_numpy(predictions_val)), # (1200, 2)
    #    torch.from_numpy(np.where(y_val)[1]).long(), # (1200)
    #)
    #print("\t\t Val auroc (clean): {:.3f}".format(auroc_val))
    
    # test
    predictions_test = classifier.predict(x_test)
    accuracy_test = np.sum(np.argmax(predictions_test, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("\t\t Test Acc (clean): {:.3f}".format(accuracy_test))
    
    auroc_test = AUROC(task="multiclass", num_classes=num_classes)(
        nn.Softmax(dim=1)(torch.from_numpy(predictions_test)), # (1200, 2)
        torch.from_numpy(np.where(y_test)[1]).long(), # (1200)
    )
    print("\t\t Test auroc (clean): {:.3f}".format(auroc_test))
    if opt.save_baseline:
    #    np.save(os.path.join(load_save_path, f"val_baseline_pred.npy"), nn.Softmax(dim=1)(torch.from_numpy(predictions_val)).numpy())
        np.save(os.path.join(load_save_path, f"test_baseline_pred.npy"), nn.Softmax(dim=1)(torch.from_numpy(predictions_test)).numpy())

    if opt.plot_all_attacks:
        if not os.path.exists(os.path.join(path_fig, opt.dataset)):
            os.makedirs(os.path.join(path_fig, opt.dataset))
        if not os.path.exists(os.path.join(path_csv, opt.dataset)):
            os.makedirs(os.path.join(path_csv, opt.dataset))
        #plot_attacks_all(classifier, x_val, y_val, path_fig, path_csv, opt.dataset, 'vgg16_attacks', num_classes)
        plot_attacks_all(classifier, x_test, y_test, path_fig, path_csv, opt.dataset, 'vgg16_attacks', num_classes)
    else:
        for eps in opt.epsilons:
            print(f"\nFor {opt.attack_type} attack with eps {eps}:")
            #val_attacked_samples = os.path.join(load_save_path, f"val_{opt.attack_type}_{eps}.npy")
            test_attacked_samples = os.path.join(load_save_path, f"test_{opt.attack_type}_{eps}.npy")
            #if not (os.path.exists(val_attacked_samples) and os.path.exists(test_attacked_samples)):
            #if not os.path.exists(test_attacked_samples):
            attacker = create_attack(
                opt.attack_type,
                classifier,
                eps,
                eps * opt.eps_step_ratio,
                opt.max_iter,
                opt.num_random_init,
            )

            #if not os.path.exists(val_attacked_samples):
            #    print("Val attacked samples do not exist. Generating...")
                #import time
                #val_start_time = time.time()
                #x_val_adv = attacker.generate(x=x_val)
                #val_attack_time = time.time() - val_start_time
                #print(f"attack time: {val_attack_time}")
            #    if opt.save_adv_results:
            #        np.save(val_attacked_samples, x_val_adv)
            #else:
            #    print(f"Loading attacked samples {val_attacked_samples}...")
            #    x_val_adv = np.load(val_attacked_samples)

            #if not os.path.exists(test_attacked_samples):
            #    print("Test attacked samples do not exist. Generating...")
            x_test_adv = attacker.generate(x=x_test)
            if opt.save_adv_results:
                np.save(test_attacked_samples, x_test_adv)
            #else:
            #    print(f"Loading attacked samples {test_attacked_samples}...")
            #    x_test_adv = np.load(test_attacked_samples)

            # partial attack
            #if opt.attack_portion is not None:
            #    print(f"Val attack Portion: {opt.attack_portion}")
            #    assert sum(num_feats_each_group) == x_val.shape[1] - 1
            #    x_val = partial_attack(x_val, x_val_adv, opt.attack_portion, num_feats_each_group, feat_names)
            #    attack_portion_name = ''.join(opt.attack_portion)
            #else:
            #    print(f"Val attack All")
            #    x_val = x_val_adv
            #    attack_portion_name = "All"

            #predictions_val_adv = classifier.predict(x_val)
            #accuracy_val_adv = np.sum(np.argmax(predictions_val_adv, axis=1) == np.argmax(y_val, axis=1)) / len(y_test)
            #print("\t\t Val Acc after Attack: {:.3f}".format(accuracy_val_adv))
            #auroc_val_adv = AUROC(task="multiclass", num_classes=num_classes)(
            #    nn.Softmax(dim=1)(torch.from_numpy(predictions_val_adv)), # (1200, 2)
            #    torch.from_numpy(np.where(y_val)[1]).long(), # (1200)
            #)

            #print("\t\t Val auroc after Attack: {:.3f}".format(auroc_val_adv))
            #fpr_val_adv, tpr_val_adv, _ = roc_curve(
            #    np.argmax(y_val, axis=1),
            #    nn.Softmax(dim=1)(torch.from_numpy(predictions_val_adv))[:, 1],
            #)
            #plot_roc_curve(fpr_val_adv, tpr_val_adv, auroc_val_adv, load_save_path, f"val_{opt.attack_type}_{eps}_{attack_portion_name}")

            # partial attack
            if opt.attack_portion is not None:
                print(f"Test attack Portion: {opt.attack_portion}")
                assert sum(num_feats_each_group) == x_test.shape[1] - 1
                x_test = partial_attack(x_test, x_test_adv, opt.attack_portion, num_feats_each_group, feat_names)
                attack_portion_name = ''.join(opt.attack_portion)
            else:
                print(f"Test attack All")
                x_test = x_test_adv
                attack_portion_name = "All"
            
            predictions_test_adv = classifier.predict(x_test)
            accuracy_test_adv = np.sum(np.argmax(predictions_test_adv, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
            print("\t\t Test Acc after Attack: {:.3f}".format(accuracy_test_adv))
            auroc_test_adv = AUROC(task="multiclass", num_classes=num_classes)(
                nn.Softmax(dim=1)(torch.from_numpy(predictions_test_adv)), # (1200, 2)
                torch.from_numpy(np.where(y_test)[1]).long(), # (1200)
            )
            print("\t\t Test auroc after Attack: {:.3f}".format(auroc_test_adv))
            fpr_test_adv, tpr_test_adv, _ = roc_curve(
                np.argmax(y_test, axis=1),
                nn.Softmax(dim=1)(torch.from_numpy(predictions_test_adv))[:, 1],
            )
            plot_roc_curve(fpr_test_adv, tpr_test_adv, auroc_test_adv, load_save_path, f"test_{opt.attack_type}_{eps}_{attack_portion_name}")
            if opt.save_adv_results:
                #np.save(os.path.join(load_save_path, f"test_{opt.attack_type}_{eps}_{opt.attack_portion}.npy"), x_test_adv)
                np.save(os.path.join(load_save_path, f"test_{opt.attack_type}_{eps}_{attack_portion_name}_pred.npy"), nn.Softmax(dim=1)(torch.from_numpy(predictions_test_adv)).numpy())        
