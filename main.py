import copy
import os
import utils
import argparse
import joblib
from tqdm import tqdm
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from custom_types import Basic, TrainConfig
from modules import MATconv as MAT
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument("--hidden", type=int, default=32, help="encoder decoder hidden size")
    parser.add_argument("--logname", action="store", default='root', help="name for log")
    parser.add_argument("--subName", action="store", type=str, default='test', help="name of the directory of current run")
    parser.add_argument("--inputLength", type=int, default=864, help="input length for the model")
    parser.add_argument("--outputLength", type=int, default=864, help="output length for the model")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--dataAug", action="store_true", help="data augmentation mode")
    parser.add_argument("--prob0", type=float, default=0.3, help="augment probability for Dishwasher")
    parser.add_argument("--prob1", type=float, default=0.6, help="weight")
    parser.add_argument("--prob2", type=float, default=0.3, help="weight")
    parser.add_argument("--prob3", type=float, default=0.3, help="weight")
    return parser.parse_args()


def train(t_net, train_Dataloader, vali_Dataloader, config, criterion, modelDir, epo=200):
    iter_loss = []
    vali_loss = []
    early_stopping_all = utils.EarlyStopping(logger, patience=30, verbose=True)

    if config.dataAug:
        sigClass = utils.sigGen(config)

    path_all = os.path.join(modelDir, "All_best_onoff.ckpt")

    for e_i in range(epo):

        logger.info(f"# of epoches: {e_i}")
        for t_i, (_, _, X_scaled, Y_scaled, Y_of) in enumerate(tqdm(train_Dataloader)):
            if config.dataAug:
                X_scaled, Y_scaled, Y_of = utils.dataAug(X_scaled.clone(), Y_scaled.clone(), Y_of.clone(), sigClass, config)

            t_net.model_opt.zero_grad(set_to_none=True)

            X_scaled = X_scaled.type(torch.FloatTensor).to(device, non_blocking=True)
            Y_scaled = Y_scaled.type(torch.FloatTensor).to(device, non_blocking=True)
            Y_of = Y_of.type(torch.FloatTensor).to(device, non_blocking=True)

            y_pred_dish_r, y_pred_dish_c = t_net.model(X_scaled)

            loss_r = criterion[0](y_pred_dish_r,Y_scaled)
            loss_c = criterion[1](y_pred_dish_c, Y_of)

            loss=loss_r+loss_c
            loss.backward()

            t_net.model_opt.step()
            iter_loss.append(loss.item())

        epoch_losses = np.average(iter_loss)

        logger.info(f"Validation: ")
        maeScore, y_vali_ori, y_vali_pred_d_update, _, _, _ = utils.evaluateResult(net, config, vali_Dataloader, logger)
        val_loss = criterion[0](y_vali_ori, y_vali_pred_d_update)
        logger.info(f"Epoch {e_i:d}, train loss: {epoch_losses:3.3f}, val loss: {val_loss:3.3f}.")
        vali_loss.append(val_loss)

        if e_i % 10 == 0:
            checkpointName = os.path.join(modelDir, "checkpoint_" + str(e_i) + '.ckpt')
            utils.saveModel(logger, net, checkpointName)

        logger.info(f"Early stopping overall: ")
        early_stopping_all(np.mean(maeScore), net, path_all)
        if early_stopping_all.early_stop:
            print("Early stopping")
            break

    net_all = copy.deepcopy(net)
    checkpoint_all = torch.load(path_all, map_location=device)
    utils.loadModel(logger, net_all, checkpoint_all)
    net_all.model.eval()
    
    return net_all

if __name__ == '__main__':
    args = get_args()
    utils.mkdir("log/" + args.subName)
    logger = utils.setup_log(args.subName, args.logname)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using computation device: {device}")
    logger.info(args)
    if args.debug:
        epo = 2
    else:
        epo = 200

    # splitLoss = False
    # trainFull = True

    # Dataloder
    logger.info(f"loading data")
    train_data, val_data, test_data = utils.data_loader(args)

    logger.info(f"loading data finished")

    config_dict = {
        "input_size": 1,
        "batch_size": args.batch,
        "hidden": args.hidden,
        "lr": args.lr,
        "dropout": args.dropout,
        "logname": args.logname,
        "outputLength": args.outputLength,
        "inputLength" : args.inputLength,
        "subName": args.subName,
        "dataAug": args.dataAug,
        "prob0": args.prob0,
        "prob1": args.prob1,
        "prob2": args.prob2,
        "prob3": args.prob3,
    }

    config = TrainConfig.from_dict(config_dict)
    modelDir = utils.mkdirectory(config.subName, saveModel=True)
    joblib.dump(config, os.path.join(modelDir, "config.pkl"))


    logger.info(f"Training size: {train_data.cumulative_sizes[-1]:d}.")

    index = np.arange(0,train_data.cumulative_sizes[-1])
    train_subsampler = torch.utils.data.SubsetRandomSampler(index)
    train_Dataloader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        sampler=train_subsampler,
        num_workers=1,
        pin_memory=True)

    sampler = utils.testSampler(val_data.cumulative_sizes[-1], config.outputLength)
    sampler_test = utils.testSampler(test_data.cumulative_sizes[-1], config.outputLength)

    vali_Dataloader = DataLoader(
        val_data,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=1,
        pin_memory=True)

    test_Dataloader = DataLoader(
        test_data,
        batch_size=config.batch_size,
        sampler=sampler_test,
        num_workers=1,
        pin_memory=True)

    logger.info("Initialize model")
    model = MAT(config).to(device)
    logger.info("Model MAT")

    optim = optim.Adam(params=[p for p in model.parameters() if p.requires_grad], lr=config.lr)
    net = Basic(model, optim)
    criterion_r = nn.MSELoss()
    criterion_c = nn.BCELoss()
    criterion = [criterion_r, criterion_c]

    logger.info("Training start")
    net_all = train(net, train_Dataloader, vali_Dataloader, config, criterion, modelDir, epo=epo)
    logger.info("Training end")

    logger.info("validation start")
    utils.evaluateResult(net_all, config, vali_Dataloader, logger)
    logger.info("test start")
    utils.evaluateResult(net_all, config, test_Dataloader, logger)

