# Adversarial Learning
# import dependencies
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

from configs import path
from data_loader import LIDC_Dataset
from plot_utils import plot_tv, plot_roc_curve
from train_utils import set_seed, seed_worker
from sklearn.metrics import roc_curve
from attacks import create_attack, partial_attack
from models import Medical_Vgg_Model

class Options():
    def __init__(self):
        ## data setup
        self.dataset = 'lidc' #'lidc' # dataset options: mnist/lidc
        self.datalist = 'meta_622n_balanced6k_filterissuecase' # datalist file
        self.pretrain = 'imagenet' # pretrain options: 'imagenet' / None
        self.batchsize = 128 #64
        self.original_data = True # original data or preprocessed data
        self.aug = True # augmentation or not
        self.crop_size = (40, 40) # cropping size for the data default: (40, 40) 
        self.input_shape = (224, 224) # data input size for the model
        self.clip_range = (-1000, 500) # if preprocessed == True, None; o/w default: (-1000, 500)
        self.feat_extract = ['mindssc', 'grad', 'pyradiomic'] # mindssc / grad / pyradiomic / pyradiomic-wavelet
        self.feat_groups_categories = ['gldm', 'glszm'] # specify the feature selections for pyradiomics: firstorder / glcm / gldm / glrlm / glszm / ngtdm (gldm and glszm)
        # crop_size and clip_range are referenced from following literature:
        # https://www.sciencedirect.com/science/article/pii/S0957417419300545
        ##################### hyparameter tuning ###################
        ## model setup
        self.training = True
        self.init_lr = 0.0002 # initial learning rate
        self.momentum = 0.9
        self.nesterov = True
        self.lr_dc = True #learning rate decay
        self.feat_embedding_layer = True
        self.image_free = False

        ## pretrain setup
        self.pretrain_model = True #True
        self.pretrain_exp_name = 'lidc_baseline_orig_ep200_clip-1k500_crop40x40_reshape224x224_aug_addfeatextractlayer_mindsscgradgldmglszm' 
        #'lidc_baseline_orig_ep200_clip-1k500_crop40x40_reshape224x224_aug'
        #'lidc_baseline_processed_ep200_clip-1k500_crop40x40_reshape224x224_aug_addfeatextractlayer'
        self.pretrain_load_model = 'best_auroc'

        ## training setup
        self.exp_name = 'ptfr_lidc_baseline_orig_ep200_clip-1k500_crop40x40_reshape224x224_aug_addfeatextractlayer_mindsscgradgldmglszm_adv_pgd_eps3e-3_r0.25_iter10_randint5_rand_bs64_ep30'
        #'ptfr_lidc_baseline_orig_ep200_clip-1k500_crop40x40_reshape224x224_aug_adv_pgd_eps3e-3_r0.25_iter10_randint5_rand_bs128_ep30'
        #'lidc_baseline_processed_ep200_clip-1k500_crop40x40_reshape224x224_aug_addfeatextractlayer' #_gldmglszm'
        #'ptfr_lidc_baseline_orig_ep200_clip-1k500_crop40x40_reshape224x224_aug_addfeatextractlayer_mindsscgradgldmglszm_adv_pgd_eps3e-3_r0.25_iter10_randint5_rand_bs64_ep30'
        #'lidc_baseline_orig_ep200_clip-1k500_crop40x40_reshape224x224_aug_addfeatextractlayer_mindsscgradgldmglszm_imgfree'
        #'lidc_baseline_orig_ep200_clip-1k500_crop40x40_reshape224x224_aug_addfeatextractlayer_pyradiomic_glszm_imgfree'
        self.total_epoch = 30 #200
        self.patience = 50
        self.print_every = 5
        self.val_print_every = 1000 #100

        ## attack setup
        self.attack = True #True
        self.attack_portion = None
        self.attack_type = 'pgd' # options: 'fgsm' / 'pgd' / 'bim' 
        self.epsilon = 0.003
        self.eps_step_ratio = 0.25
        self.max_iter = 10
        self.num_random_init = 5
        self.adv_train_type = "random" # options: 'random' / 'random_batch'
        ############################################################
        ## evaluation setup
        self.evaluation = True
        self.eval_exp_name = self.exp_name
        self.load_model = 'best_auroc' # Options: 'last'/'best_acc'/'best_auroc'

def main(path):
    set_seed(1234)
    opt = Options()
    save_path = os.path.join(path, 'exp', opt.exp_name, opt.dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if opt.training:
        with open(os.path.join(save_path, "args.json"), "w") as f:
            json.dump(vars(opt), f, indent=4)

    # load dataset
    if opt.dataset == "lidc":
        datalist = os.path.join(path, 'data', 'Meta', f'{opt.datalist}.csv')
        train_dataset = LIDC_Dataset(
            path=path,
            datalist=datalist,
            mode="Train",
            crop_size=opt.crop_size,
            crop_type="roi_center",
            aug=opt.aug,
            orig=opt.original_data,
            clip_range=opt.clip_range,
            input_shape=opt.input_shape,
            feat_extract=opt.feat_extract,
            feat_groups_categories=opt.feat_groups_categories,
        )
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
        num_classes = 2
    elif opt.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.Pad(padding=2, fill=0, padding_mode='constant'),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
        
    train_loader = DataLoader(train_dataset, batch_size=opt.batchsize, shuffle=True, num_workers=4, worker_init_fn=seed_worker, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    print("Training dataset size:", len(train_dataset))
    print("Validation dataset size:", len(val_dataset))

    for batch in train_loader:
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


    if opt.training:
        # load pretrained model
        if opt.pretrain_model and opt.pretrain_exp_name is not None:
            load_pretrain_path = os.path.join(path, 'exp', opt.pretrain_exp_name, opt.dataset)
            if opt.pretrain_load_model == "last":
                pretrain_model_name = 'vgg16_model.pth'
            elif opt.pretrain_load_model == "best_acc":
                pretrain_model_name = 'vgg16_bestaccmodel.pth'
            elif opt.pretrain_load_model == "best_auroc":
                pretrain_model_name = 'vgg16_bestroaucmodel.pth'
            else:
                raise ValueError('Incorrect load model type!')
            
            try:
                pretrain_model_weights = torch.load(
                    os.path.join(load_pretrain_path, opt.dataset + f"_{pretrain_model_name}"),
                    map_location='cuda:0',
                )
                model.load_state_dict(pretrain_model_weights)
                print("loading pretrain model weights Done")
            except:
                model = torch.load(
                    os.path.join(load_pretrain_path, opt.dataset + f"_{pretrain_model_name}"),
                    map_location='cuda:0',
                )
                print("loading pretrain model Done")
        
        if opt.attack:
            classifier = PyTorchClassifier(
                model=model,
                clip_values=opt.clip_range,
                loss=criterion,
                optimizer=optimizer,
                input_shape=opt.input_shape,
                nb_classes=num_classes,
            )
            attacker = create_attack(
                opt.attack_type,
                classifier,
                opt.epsilon,
                opt.epsilon * opt.eps_step_ratio,
                opt.max_iter,
                opt.num_random_init,
            )

        best_acc = 0.0
        best_auroc = 0.0
        no_improve = 0
        train_losses, train_accs, train_aucs = [], [], []
        val_losses, val_accs, val_aucs = [], [], []

        for epoch in range(opt.total_epoch):
            training_loss = 0.0
            # Train the model
            model.train()
            results_list = []
            labels_list = []
            for i, batch in enumerate(train_loader):
                x, y = batch[:2]
                if opt.feat_extract and opt.feat_embedding_layer:
                    feats = batch[-1]
                
                # adversarial training
                if opt.attack:
                    if opt.feat_extract:
                        feats = [v for k, v in feats.items()]
                        feats = torch.cat(feats, dim=1)
                        x = torch.cat((x, feats), dim=1)
                    
                    if opt.adv_train_type == "random":
                        if torch.rand(1) >= 0.5:
                            x = x.detach().cpu().numpy()
                            x_adv = attacker.generate(x=x)
                            if opt.attack_portion is not None:
                                print(f"Adversarial Training Portion: {opt.attack_portion}")
                                x = partial_attack(x, x_adv, opt.attack_portion, num_feats_each_group, feat_names) 
                            x = torch.from_numpy(x).to(device)
                        else:
                            x = x.to(device)

                    elif opt.adv_train_type == "random_batch":
                        # x (50, 1, 224, 224)
                        selected_batch_ids = torch.randint(0, x.shape[0], (x.shape[0]//2,))
                        attack_x = x[selected_batch_ids] # (25, 1, 224, 224)
                        attack_x = attack_x.detach().cpu().numpy()
                        attacked_x = attacker.generate(x=attack_x)
                        if opt.attack_portion is not None:
                            print(f"Adversarial Training Portion: {opt.attack_portion}")
                            attacked_x = partial_attack(attack_x, attacked_x, opt.attack_portion, num_feats_each_group, feat_names)
                        x[selected_batch_ids] = torch.from_numpy(attacked_x)
                        x = x.to(device)
                    else:
                        x = x.to(device)
                else:
                    x = x.to(device)
                    if opt.feat_extract:
                        feats = [v.to(device=device) for k, v in feats.items()]
                        feats = torch.cat(feats, dim=1)
                        x = torch.cat((x, feats), dim=1)

                y = y.to(device).long()

                optimizer.zero_grad()
                
                outputs = model(x)

                loss = criterion(
                    outputs,
                    y,
                ).to(device)
                loss.backward()
                optimizer.step()
                training_loss += loss.item()

                if opt.print_every != 0 and i % opt.print_every == 0:
                    print(f'batch {i+1}/{len(train_loader)}: loss: {training_loss/((i+1) * opt.batchsize * opt.print_every):.5f}')

                outputs = outputs.detach().cpu().numpy().astype(np.float32)
                if len(outputs.shape) == 1:
                    outputs = np.expand_dims(outputs, axis=1).astype(np.float32)

                results_list.append(outputs)
                y = y.detach().cpu().numpy()
                if len(y.shape) == 1:
                    y = np.expand_dims(y, axis=1)

                labels_list.append(y)
            predictions_tr = np.vstack(results_list)
            labels_tr = np.vstack(labels_list)

            # predict on train
            logits = nn.Softmax(dim=1)(torch.from_numpy(predictions_tr))
            pred_class = torch.argmax(logits, dim=1)
            tr_acc = (pred_class == torch.from_numpy(labels_tr).squeeze(1)).sum() / len(labels_tr)
            tr_auroc = AUROC(task="multiclass", num_classes=num_classes)(
                logits, #logits[torch.arange(len(logits)), torch.from_numpy(labels_tr).squeeze(1)], 
                torch.from_numpy(labels_tr).squeeze(1).long(),
            )

            # Evaluation
            model.eval()
            val_results_list = []
            val_labels_list = []
            validation_loss = 0.0
            for i, val_batch in enumerate(val_loader):
                val_x, val_y = val_batch[:2]
                if opt.feat_extract and opt.feat_embedding_layer:
                    val_feats = val_batch[-1]
                val_x = val_x.to(device)
                val_y = val_y.to(device).long()

                with torch.no_grad():
                    if opt.feat_extract:
                        val_feats = [v.to(device=device) for k, v in val_feats.items()]
                        val_feats = torch.cat(val_feats, dim=1)
                        val_x = torch.cat((val_x, val_feats), dim=1)
                    val_outputs = model(val_x)

                val_loss = criterion(
                    val_outputs,
                    val_y,
                ).to(device)
                validation_loss += val_loss.item()

                if opt.val_print_every != 0 and i % opt.val_print_every == 0:
                    print(f'val batch {i+1}/{len(val_loader)}: loss: {validation_loss/((i+1) * 1 * opt.val_print_every):.5f}')

                val_outputs = val_outputs.detach().cpu().numpy().astype(np.float32)
                if len(val_outputs.shape) == 1:
                    val_outputs = np.expand_dims(val_outputs, axis=1).astype(np.float32)

                val_results_list.append(val_outputs)
                val_y = val_y.detach().cpu().numpy()
                if len(val_y) == 1:
                    val_y = np.expand_dims(val_y, axis=1)

                val_labels_list.append(val_y)
            predictions_val = np.vstack(val_results_list)
            labels_val = np.vstack(val_labels_list)

            # Evaluate classifier on train set
            val_logits = nn.Softmax(dim=1)(torch.from_numpy(predictions_val))
            val_pred_class = torch.argmax(val_logits, dim=1)
            val_acc = (val_pred_class == torch.from_numpy(labels_val).squeeze(1)).sum() / len(labels_val)
            val_auroc = AUROC(task="multiclass", num_classes=num_classes)(
                val_logits, #val_logits[torch.arange(len(val_logits)), torch.from_numpy(np.array(val_labels_list))], 
                torch.from_numpy(labels_val).squeeze(1).long(),
            )

            print("Epoch [{}/{}] lr: {:.7f} Training loss {:.3f} acc {:.3f} auroc {:.3f} Valid loss {:.3f} acc {:.3f} auroc {:.3f}".format(
                    epoch, opt.total_epoch, optimizer.param_groups[0]['lr'],
                    training_loss/len(train_loader), tr_acc, tr_auroc, 
                    validation_loss/len(val_loader), val_acc, val_auroc
                )
            )

            if opt.lr_dc is not None:
                update_lr = opt.init_lr - (epoch + 1) * 1e-6
                for param_group in optimizer.param_groups:
                    param_group['lr'] = update_lr

            train_losses.append(training_loss/len(train_loader))
            train_accs.append(tr_acc)
            train_aucs.append(tr_auroc)
            val_losses.append(validation_loss/len(val_loader))
            val_accs.append(val_acc)
            val_aucs.append(val_auroc)

            if val_acc > best_acc or val_auroc > best_auroc:
                if val_acc > best_acc:
                    best_accuracy = val_acc
                    torch.save(model.state_dict(), os.path.join(save_path, opt.dataset + "_vgg16_bestaccmodel.pth"))
                if val_auroc > best_auroc:
                    best_auroc = val_auroc
                    torch.save(model.state_dict(), os.path.join(save_path, opt.dataset + "_vgg16_bestroaucmodel.pth"))
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= opt.patience:
                    print("Early Stopping after epoch {}.".format(epoch))
                    break

        # save model
        torch.save(model.state_dict(), os.path.join(save_path, opt.dataset + f'_vgg16_model.pth'))

        # Plot training and validation curves of classifier model
        plot_tv(train_losses, val_losses, save_path, 'VGG16', 'tr_curve', 'Loss')
        plot_tv(train_accs, val_accs, save_path, 'VGG16', 'tv_curve', 'Accuracy')
        plot_tv(train_aucs, val_aucs, save_path, 'VGG16', 'tv_curve', 'AUC')

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
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        print("Validation dataset size:", len(val_dataset))

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    print("Test dataset size:", len(test_dataset))

    for test_batch in test_loader:
        images, labels = test_batch[:2]
        break
    print("Batch shape:", images.shape)
    print("Batch label shape:", labels.shape)

    # Evaluation
    if opt.evaluation == True:
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
            eval_model_weights = torch.load(
                os.path.join(load_save_path, opt.dataset + f"_{model_name}"), 
                map_location='cuda:0',
            )
            model.load_state_dict(eval_model_weights)
        except:
            model = torch.load(
                os.path.join(load_save_path, opt.dataset + f"_{model_name}"), 
                map_location='cuda:0',
            )
            
        val_loss, val_acc, val_auroc, val_fpr, val_tpr = evaluate_model(
            model, val_loader, opt.val_print_every, num_classes, device, opt
        )
        print("Validation loss {:.3f} acc {:.3f} auroc {:.3f}".format( 
                val_loss, val_acc, val_auroc
            )
        )
        plot_roc_curve(val_fpr, val_tpr, val_auroc, load_save_path, "val")
        
        test_loss, test_acc, test_auroc, test_fpr, test_tpr = evaluate_model(
            model, test_loader, opt.val_print_every, num_classes, device, opt
        )

        print("Test loss {:.3f} acc {:.3f} auroc {:.3f}".format( 
                test_loss, test_acc, test_auroc
            )
        )
        plot_roc_curve(test_fpr, test_tpr, test_auroc, load_save_path, "test")

def evaluate_model(model, dataloader, print_every, num_classes, device, opt):
    eval_criterion = nn.CrossEntropyLoss()
    model.eval()
    results_list, labels_list, test_loss = [], [], 0.0
    for i, batch in enumerate(dataloader):
        x, y = batch[:2]
        if opt.feat_extract and opt.feat_embedding_layer:
            feats = batch[-1]
        x = x.to(device)
        y = y.to(device).long()
        with torch.no_grad():
            if opt.feat_extract:
                feats = [v.to(device=device) for k, v in feats.items()]
                feats = torch.cat(feats, dim=1)
                x = torch.cat((x, feats), dim=1)
            outputs = model(x)
        
        loss = eval_criterion(
            outputs,
            y,
        ).to(device)
        
        test_loss += loss.item()

        if print_every != 0 and i % print_every == 0:
            print(f'batch {i+1}/{len(dataloader)}: loss: {test_loss/((i + 1) * 1 * print_every):.5f}')

        outputs = outputs.detach().cpu().numpy().astype(np.float32)
        if len(outputs.shape) == 1:
            outputs = np.expand_dims(outputs, axis=1).astype(np.float32)

        results_list.append(outputs)
        y = y.detach().cpu().numpy()
        if len(y) == 1:
            y = np.expand_dims(y, axis=1)

        labels_list.append(y)
    predictions = np.vstack(results_list)
    labels = np.vstack(labels_list)

    # Evaluate classifier on train set
    logits = nn.Softmax(dim=1)(torch.from_numpy(predictions))
    pred_class = torch.argmax(logits, dim=1)
    acc = (pred_class == torch.from_numpy(labels).squeeze(1)).sum() / len(labels)
    auroc = AUROC(task="multiclass", num_classes=num_classes)(
        logits, #val_logits[torch.arange(len(val_logits)), torch.from_numpy(np.array(val_labels_list))], 
        torch.from_numpy(labels).squeeze(1).long(),
    )
    
    # ROCAUC Curve
    fpr, tpr, _ = roc_curve(
        labels, 
        nn.Softmax(dim=1)(torch.from_numpy(predictions))[:, 1],
    )
    
    return test_loss/len(dataloader), acc, auroc, fpr, tpr

if __name__ == "__main__":
    main(path)
