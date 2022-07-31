import time
import torch
import json
import os
from torch.utils.data import DataLoader
from loguru import logger
import pandas as pd
import argparse
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
import configparser
from src import model,data,loss,utils
from pathlib import Path
import datetime


def main(config):
    name = config['name']
    d = datetime.datetime.now()
    timestamp = d.strftime("%m_%d_%Y_%H_%M")
    model_dir = f'models/{name}/{timestamp}'
    Path(model_dir).mkdir(exist_ok=True, parents=True)

    torch.backends.cudnn.benchmark = True
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # train dataloader
    path_to_train = config["train_data_loader"]["path"]
    face = getattr(data,config['train_data_loader']['type'])
    use_aug = config['train_data_loader']['use_augmentation']
    face_dataset_train = face(path_to_train,mode="train",aug = use_aug)
    dataloader_train = DataLoader(face_dataset_train,batch_size=config["train_data_loader"]["batch_size"],shuffle=True, num_workers=0)
    # val dataloader
    path_to_validation = config["validation_data_loader"]["path"]
    face_dataset_val = face(path_to_validation,mode="validation", aug = use_aug)
    dataloader_val = DataLoader(face_dataset_val,batch_size=config["validation_data_loader"]["batch_size"],shuffle=False,num_workers=0)
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model
    logger.info(f"Initializing model..")
    dropout = config['arch']['dropout']
    arch = getattr(model,config['arch']['type'])(dropout).to(device)

    # loss
    criterion = getattr(loss,config['loss'])()

    # metric
    metric = getattr(loss,config['metrics'])(sigma=3)
    # optimizer
    logger.info(f"Initializing Optimizer..")
    params = list(arch.parameters())
    opt = getattr(torch.optim,config['optimizer']['type'])
    optimizer = opt(params=params, **config['optimizer']['args'])
    # scheduler
    sch = getattr(torch.optim.lr_scheduler,config['lr_scheduler']['type'])
    scheduler = sch(optimizer, **config['lr_scheduler']['args'])
    # tensorboard

    writer = SummaryWriter(logdir=model_dir)

    total_i = 0
    is_best = False
    best_val_error = 1

    logger.info(f"Training started..")
    epochs = config['trainer']['epochs']
    for epoch in range(epochs):
        if epoch==9:
            print("here")
        # training loop
        arch.train()
        pbar = tqdm(enumerate(dataloader_train),
                    total=len(dataloader_train))

        start_time = time.time()
        training_loss = 0
        training_error = 0
        for i, batch  in pbar:
            img, age = batch
            img = img.to(device)
            age = age.to(device)
            prepare_time = start_time - time.time()
            # forward the image batch through the network to get predicted age
            predicted_age = arch(img, mode="train")
            # calculate KL divergence based loss
            c_loss = criterion(predicted_age,age.flatten())
            if c_loss.isnan():
                print("here ")
            training_loss+= c_loss
            # calculate error
            error = metric(predicted_age.argmax(1),age.flatten()).mean()
            training_error+=error
            # take step using gradient
            optimizer.zero_grad()
            c_loss.backward()
            optimizer.step()
            # scheduler.step()
            total_i += 1
            writer.add_scalar("Training Loss", training_loss/(i+1), total_i)

            process_time = start_time - time.time() - prepare_time
            compute_efficiency = process_time / (process_time + prepare_time)

            pbar.set_description(
                f'Compute efficiency: {compute_efficiency:.2f}, '
                f'loss: {training_loss / (i + 1):.4f},  epoch: {epoch}/{epochs}')
            start_time = time.time()
        writer.add_scalar('Training Error', training_error/total_i, epoch)
        logger.info(f'Training Error:{training_error/total_i}')
        pbar = tqdm(enumerate(dataloader_val),
                    total=len(dataloader_val))
        # validation
        with torch.no_grad():
            arch.eval()
            validation_loss = 0
            validation_error = 0
            total_i = 0
            for i, batch in pbar:
                total_i+=1
                # data preparation
                img, age = batch
                img = img.to(device)
                age = age.to(device)
                # forward pass for validation
                predicted_age = arch(img, mode="eval")
                c_loss  = criterion(predicted_age,age.flatten())
                validation_loss += c_loss
                error = metric(predicted_age.argmax(1), age.flatten()).mean()
                validation_error += error

            avg_val_error =  validation_error/total_i
            if avg_val_error < best_val_error:
                logger.info(f'Found new best at epoch ' + str(epoch))
                is_best = True
                best_val_error = avg_val_error
            writer.add_scalar('Validation Error', avg_val_error, epoch)
            writer.add_scalar('Validation Loss', avg_val_error, epoch)
            logger.info(f'Validation Error:{avg_val_error}')

        ckpt_folder = os.path.join(model_dir, "checkpoints")
        Path(ckpt_folder).mkdir(parents=True, exist_ok=True)

        cpkt = {
            'net': arch.state_dict(),
            'epoch': epoch,
            'optim': optimizer.state_dict(),
            'val_stats': avg_val_error
        }
        save_path = os.path.join(ckpt_folder, 'model.ckpt')
        utils.save_checkpoint(cpkt, save_path, is_best)
        is_best = False





if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=False, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-bd', '--dataset_base_dir', default='.', type=str,
                      help='base path for dataloader (default: None)')
    args.add_argument('-b', '--workspace_base_dir', default='.', type=str,
                      help='base path for loading checkpoints (default: None)')
    # custom cli options to modify configuration from default values given in json file.
    args = args.parse_args()
    with open(args.config, 'r') as f:
        config = json.load(f)
    main(config)






