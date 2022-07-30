import time
import torch
import os
from torch.utils.data import DataLoader
from loguru import logger
import pandas as pd
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter

from src.data import Face
from src.model import AgeEstimation, EstimationError

epochs = 50

torch.backends.cudnn.benchmark = True
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

path_to_train = "data/train.csv"
path_to_validation = "data/validation.csv"
learning_rate = 0.00001
# face dataset
face_dataset_train = Face(path_to_train,mode="train")
face_dataset_val = Face(path_to_validation,mode="validation")

# dataloader
dataloader_train = DataLoader(face_dataset_train,batch_size=8,shuffle=True,num_workers=0)
dataloader_val = DataLoader(face_dataset_val,batch_size=8,shuffle=False,num_workers=0)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model
logger.info(f"Initializing model..")
model = AgeEstimation().to(device)

# loss
criterion = torch.nn.NLLLoss()
criterion_L1 = EstimationError(sigma=3)
# optimizer
logger.info(f"Initializing Optimizer..")
params = list(model.parameters())
optimizer = torch.optim.Adam(params=params, lr = learning_rate)

# tensorboard
tensorboard_path  = os.path.join("model")
writer = SummaryWriter()

total_i = 0
is_best = False

for epoch in range(epochs):

    model.train()
    pbar = tqdm(enumerate(dataloader_train),
                total=len(dataloader_train))

    start_time = time.time()
    training_loss = 0
    training_acc = 0
    for i, batch  in pbar:
        img, age = batch
        img = img.to(device)
        age = age.to(device)
        prepare_time = start_time - time.time()
        predicted_age = model(img,mode="train")
        c_loss = criterion(predicted_age,age.flatten())
        training_loss+= c_loss

        l1_metric = criterion_L1(predicted_age.argmax(1),age.flatten()).mean()
        training_acc+=l1_metric

        optimizer.zero_grad()
        c_loss.backward()
        optimizer.step()
        total_i += 1
        writer.add_scalar("Training Loss", training_loss/(i+1), total_i)

        process_time = start_time - time.time() - prepare_time
        compute_efficiency = process_time / (process_time + prepare_time)

        pbar.set_description(
            f'Compute efficiency: {compute_efficiency:.2f}, '
            f'loss: {training_loss / (i + 1):.4f},  epoch: {epoch}/{epochs}')
        start_time = time.time()
    writer.add_scalar('Training Accuracy', training_acc/total_i, epoch)
    print('Training Accuracy', training_acc/total_i)
    pbar = tqdm(enumerate(dataloader_val),
                total=len(dataloader_val))

    with torch.no_grad():
        model.eval()
        validation_loss = 0
        validation_acc = 0
        total_i = 0
        for i, batch in pbar:
            total_i+=1
            # data preparation
            img, age = batch
            img = img.to(device)
            age = age.to(device)
            # forward and backward pass
            predicted_age = model(img, mode="eval")
            c_loss  = criterion(predicted_age,age.flatten())
            validation_loss += c_loss
            l1_metric = criterion_L1(predicted_age.argmax(1), age.flatten()).mean()
            validation_acc += l1_metric
        writer.add_scalar('Validation Accuracy', validation_acc/total_i, epoch)
        writer.add_scalar('Validation Loss', validation_loss/total_i, epoch)
        print('Validation Accuracy',  validation_acc/total_i)



print("training_done")









