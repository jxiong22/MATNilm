import logging
import logging.handlers
import os
from datetime import datetime, date
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from numpy import save
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Sampler
import torch
from sklearn import metrics
from scipy.interpolate import interp1d
from scipy.stats import truncnorm
import pickle as pkl
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def setup_log(subName='', tag='root'):
    # create logger
    logger = logging.getLogger(tag)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)

    # file log
    log_name = tag + datetime.now().strftime('log_%Y_%m_%d.log')

    log_path = os.path.join('log', subName, log_name)
    fh = logging.handlers.RotatingFileHandler(
        log_path, mode='a', maxBytes=100 * 1024 * 1024, backupCount=1, encoding='utf-8'
    )

    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    return logger

def mkdir(dirName):
    if not os.path.exists(dirName):
        if os.name == 'nt':
            os.system('mkdir {}'.format(dirName.replace('/', '\\')))
        else:
            os.system('mkdir -p {}'.format(dirName))


def mkdirectory(subName, saveModel):
    dirName_data = "data/" + subName
    dirName_log = "log/" + subName
    mkdir(dirName_data)
    mkdir(dirName_log)

    if saveModel is True:
        model_name = "/s0"
        dirName_model = "history_model/" + subName + model_name
        mkdir(dirName_model)
        return dirName_model

def sae(true, pred, N):
    num_period = int(len(true) / N)
    diff = 0
    for i in range(num_period):
        diff += abs(np.sum(true[i * N: (i + 1) * N]) - np.sum(pred[i * N: (i + 1) * N]))
    return diff / (N * num_period)


def evaluate_score(y_real, y_predict, y_real_c, y_pred_c, logger):

    maeScore = np.mean(np.abs(y_predict - y_real))

    logger.info(f"MAE: {maeScore}")
    # SAE
    SAE = sae(y_real, y_predict, 1200)
    logger.info(f"SAE: {SAE}")

    f1s = f1_score(y_real_c, np.round(y_pred_c))
    logger.info(f"F1: {f1s}")

    return maeScore

def evaluate_score_multi(y_real, y_predict, y_real_c, y_pred_c, logger):
    listOfAppliance = ['dish washer', 'fridge', 'microwave', 'wash']
    mapeScore = []
    for i in range(y_predict.shape[1]):
        logger.info(f"Evaluate {listOfAppliance[i]}: ")
        mapeScore.append(evaluate_score(y_real[:,i], y_predict[:,i], y_real_c[:,i], y_pred_c[:,i], logger))
    return mapeScore
    


class testSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, length, step):
        self.length = length
        self.step = step

    def __iter__(self):
        return iter(range(0,self.length,self.step))

    def __len__(self) -> int:
        return len(range(0,self.length,self.step))


def saveModel(logger, net, path):
    torch.save({
        'model_state_dict': net.model.state_dict(),
        'model_optimizer_state_dict': net.model_opt.state_dict(),
    }, path)
    logger.info(f'Model saved')


def loadModel(logger, net, checkpoint):

    net.model.load_state_dict(checkpoint['model_state_dict'])
    net.model_opt.load_state_dict(checkpoint['model_optimizer_state_dict'])
    logger.info(f'Model loaded')
    return net

class EarlyStopping:
    def __init__(self, logger, patience=7, verbose=False, delta=0, best_score=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = best_score
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.logger = logger

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
            self.early_stop = False

    def save_checkpoint(self, val_loss, net, path):
        if self.verbose:
            self.logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        saveModel(self.logger, net, path)
        self.val_loss_min = val_loss


def data_loader(args):
    path = "data/redd/"
    train_arrays = pkl.load(open(os.path.join(path,"train_small.pkl"),'rb'))
    val_arrays = pkl.load(open(os.path.join(path,"val_small.pkl"),'rb'))
    test_arrays = pkl.load(open(os.path.join(path,"test_small.pkl"),'rb'))

    length_input = args.inputLength
    length_output = args.outputLength

    if args.debug:
        train_arrays[0] = train_arrays[0][:2000]

    ListTrain = [SubSet(array.values, length_input=length_input, length_output=length_output) for array in train_arrays]
    ListVal = [SubSet(array.values, length_input=length_input, length_output=length_output) for array in val_arrays]
    ListTest = [SubSet(array.values, length_input=length_input, length_output=length_output) for array in test_arrays]

    return ConcatDataset(ListTrain), ConcatDataset(ListVal), ConcatDataset(ListTest)

class SubSet(Dataset):
    def __init__(self, x, length_input=400, length_output=400):
        super(SubSet, self).__init__()
        self.outputs = x[:, 1:]
        self.mains = x[:, :1]
        self.inLen = length_input
        self.outLen = length_output

    def __getitem__(self, index):
        in_begin = index
        in_end = in_begin + self.inLen
        out_begin = in_begin + int((self.inLen-self.outLen)/2)
        out_end = out_begin + self.outLen

        X = self.mains[in_begin:in_end,:]
        # Y = self.outputs[out_begin:out_end,:]
        Y = self.outputs[in_begin:in_end,:]

        X_scaled = X/612
        Y_scaled = Y/612
        Y_of = np.where(Y > 15, 1, 0)

        return X, Y, X_scaled, Y_scaled, Y_of

    def __len__(self):
        return len(self.outputs)-self.inLen + 1


class sigGen():
    def __init__(self, config):
        self.inLen = config.inputLength
        self.outLen = config.outputLength

        self.pool1 = pkl.load(open("data/redd/REDD_pool.pkl",'rb'))
        self.pool2 = pkl.load(open("data/redd/poolx.pkl",'rb'))
        self.pool = self.pool1 + self.pool2
        self.off = pkl.load(open("data/redd/offduration.pkl",'rb'))
        self.offduration = [[x for house in self.off for x in house[i]] for i in range(len(self.off[0]))]
        self.offduration = self.offduration + [[]] * len(self.pool2)

        self.offInt = [[x for x in list if x <= self.inLen] for list in self.offduration]


    def getMore(self, sigCh, y_of_ori, appNum):
        appSig = self.pool[appNum]

        houseNum2 = np.random.randint(len(appSig))
        samNum2 = np.random.randint(len(appSig[houseNum2]))
        sigCh2 = appSig[houseNum2][samNum2]
        y_of_ori2 = np.ones_like(sigCh2)
        if len(self.offInt[appNum]) > 100:
            zeroLength = self.offInt[appNum][np.random.randint(0,len(self.offInt[appNum]))]
        elif len(self.offInt[appNum]) > 10 and np.random.rand(1) < 0.7:
            zeroLength = self.offInt[appNum][np.random.randint(0,len(self.offInt[appNum]))]
        else:
            zeroLength = np.random.randint(0, self.length/4)

        zeroBetween = np.zeros(zeroLength)
        sigCh = np.concatenate((sigCh, zeroBetween, sigCh2), axis=None)
        y_of_ori = np.concatenate((y_of_ori, zeroBetween, y_of_ori2), axis=None)

        return sigCh, y_of_ori

    def getSignal(self, appNum, length):
        self.length = length
        appSig = self.pool[appNum]
        houseNum = np.random.randint(len(appSig))
        samNum = np.random.randint(len(appSig[houseNum]))
        sigCh = appSig[houseNum][samNum]
        y_of_ori = np.ones_like(sigCh)

        while len(sigCh) < self.outLen * 2:
            if np.random.rand(1) < 0.5:
                break
            sigCh, y_of_ori = self.getMore(sigCh, y_of_ori, appNum)

        signal = torch.from_numpy(sigCh)
        signal_scaled = signal/612
        y_of = torch.from_numpy(y_of_ori)
        return signal_scaled, signal, y_of


def vertScale2(x):
    # scale = np.random.uniform(0.8, 1.2, 1)
    mu, sigma = 1, 0.2
    lower, upper = mu - 2*sigma, mu + 2*sigma
    tn = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    scale = tn.rvs(1)
    return x*torch.from_numpy(scale)

def horiScale2(input):
    olength = int(len(input)/2)
    # scale = np.random.uniform(0.8, 1.2, 1)
    mu, sigma = 1, 0.2
    lower, upper = mu - 2*sigma, mu + 2*sigma
    tn = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    scale = tn.rvs(1)
    y = input.reshape(-1)
    x = np.arange(0, len(y))
    f = interp1d(x, y)

    xnew = np.arange(0, len(y) - 1, scale)
    ynew = f(xnew)
    if len(ynew) > olength:
        return torch.from_numpy(ynew[:olength])
    else:
        diffSize = olength - len(ynew)
        ynew = np.pad(ynew, (0, diffSize), 'constant')
        return torch.from_numpy(ynew)


def insertSig(sig, length, config, start_index):
    signal = torch.zeros(length)
    if start_index < 0:
        signal[:min(len(sig) + start_index, length)] = sig[-start_index:min(len(sig), length - start_index)]
    else:
        signal[start_index:min(len(sig) + start_index, length)] = sig[:min(len(sig), length - start_index)]
    
    return signal

def selectPortion(config, sig, length):
    insertIdxLrange = int((config.inputLength - config.outputLength)/2) - len(sig)
    insertIdxRrange = int((config.inputLength + config.outputLength)/2)
    idx = np.random.randint(insertIdxLrange, insertIdxRrange)
    signal = insertSig(sig, length, config, idx)
    # y_of = torch.where(signal > 15/612, torch.Tensor([1]), torch.Tensor([0]))
    return signal

def dataAug( X_scaled, Y_scaled, Y_of, sigClass, config):
    orilen = Y_scaled.shape[-1]
    prob=[config.prob0, config.prob1, config.prob2, config.prob3]
    xlen = len(sigClass.pool) - orilen
    prob.extend([0.1] * xlen)
    llen = Y_scaled.shape[1]

    # prob=[config.prob0, config.prob1, config.prob2]
    for i in range(X_scaled.shape[0]):
        minX = min(X_scaled[i,:,0]).item()
        for j in range(len(prob)):
            # onOff = sum(Y_of[i,:,j])
            p = np.random.rand(1)
            if p < prob[j]: # do data_aug
                if j < orilen:
                    X_scaled[i,:,0] -= Y_scaled[i,:,j]
                    Y_scaled[i, :, j] -= Y_scaled[i,:,j]
                sig, sig_ori, y_of = sigClass.getSignal(j, Y_scaled.shape[1]*2)

                if j==0:
                    mode = np.random.choice(4, 1, p=[.25, .25, .25, .25])
                elif j==1:
                    mode = np.random.choice(4, 1, p=[.25, .25, .25, .25])
                else:
                    mode = np.random.choice(4, 1, p=[.25, .25, .25, .25])
                ## 0: origin, 1:vertical scaled, 2: horizon scaled, 3: v&h scaled
                if mode == 1:
                    sig = vertScale2(sig)

                elif mode == 2:
                    sig = horiScale2(sig)

                elif mode == 3:
                    sig = horiScale2(sig)
                    sig = vertScale2(sig)

                sig = selectPortion(config, sig, llen)
                y_of = torch.where(sig > 15/612, torch.Tensor([1]), torch.Tensor([0]))
                if j < orilen:
                    Y_scaled[i,:,j] += sig
                    Y_of[i, :, j] = y_of
                X_scaled[i,:,0] += sig

    return X_scaled, Y_scaled, Y_of


def genList(appliance, j):
    thre = [50,10,100, 100, 30]  #dishwasher, fridge, microwave, kettle, washmachine
    ontol = [150, 10, 10, 10, 150]
    offtol = [10, 10, 5, 5, 10]
    on, off, onduration, offduration = [], [], [], []
    start, end = 0, 0
    onflag = (appliance[0] > thre[j])
    tol = 0
    applLen = len(appliance)
    for i, value in enumerate(appliance):
        if onflag:
            if i == applLen - 1:
                on.append([start, i])
                onduration.append(i - start)
            else:
                if value > thre[j]:
                    end = i
                    tol = 0
                elif tol < ontol[j]:
                    tol += 1
                else:
                    # if end > start:
                    on.append([start, end])
                    onduration.append(end - start)
                    start = end + 1
                    onflag = False
                    tol = 0
        else: # off period
            if i == applLen - 1:
                off.append([start, i])
                offduration.append(i - start)
            else:
                if value < thre[j]:
                    end = i
                else:
                    l = min(i+offtol[j]*2, applLen)
                    future = appliance[i:l]
                    numofOn = (future > thre[j]).sum()
                    if numofOn > offtol[j]:
                        if end < start:
                            end = i - 1
                        off.append([start, end])
                        offduration.append(end - start)
                        start = end + 1
                        onflag = True
    return on, off, onduration, offduration


def evaluateResult(net, config, vali_Dataloader, logger, mode=-1):
    y_vali_pred, y_vali, y_vali_ori, y_vali_pred_c, y_vali_ori_c, truex = predict(net, config, vali_Dataloader, mode=mode)

    y_vali_pred = y_vali_pred.reshape(-1,y_vali_pred.shape[-1])
    y_vali_pred_c = y_vali_pred_c.reshape(-1,y_vali_pred.shape[-1])
    y_vali_ori = y_vali_ori.reshape(-1,y_vali_pred.shape[-1])
    y_vali_ori_c = y_vali_ori_c.reshape(-1,y_vali_pred.shape[-1])
    y_vali_pred[y_vali_pred < 0] = 0
    y_vali_pred_d = y_vali_pred * 612
    y_vali_pred_d_update = y_vali_pred_d

    if mode>=0:
        maeScore = evaluate_score(y_vali_ori.numpy(), y_vali_pred_d_update.numpy(), y_vali_ori_c.numpy(), y_vali_pred_c.numpy(), logger)
    else:
        maeScore = evaluate_score_multi(y_vali_ori.numpy(), y_vali_pred_d_update.numpy(), y_vali_ori_c.numpy(), y_vali_pred_c.numpy(), logger)

    return maeScore, y_vali_ori, y_vali_pred_d_update, y_vali_ori_c, y_vali_pred_c, truex.reshape(-1, 1)

def predict(t_net, t_cfg, vali_Dataloader, mode=-1):
    '''
    :param t_net: pa_rnn_net(encoder, decoder, encoder_optimizer, decoder_optimizer)
    :param t_dat: data include train and evaluation and test.
    :param t_cfg: config file for train
    :param on_train: when on_train, predict will evaluate the result using train sample, otherwise, use test sample
    :return: y_pred:
    '''
    y_pred_r = []  # (test_size)
    y_true_scaled_r = []
    y_true_r = []
    y_pred_c = []  # (test_size)
    y_true_c = []
    x_true = []
    start = int((t_cfg.inputLength-t_cfg.outputLength)/2)
    end = start + t_cfg.outputLength

    with torch.no_grad():
        for _, (X, Y, X_scaled, Y_scaled, Y_of) in enumerate(vali_Dataloader):
            if mode>=0:
                Y = Y[:,start:end,[mode]]
                Y_scaled = Y_scaled[:,start:end,[mode]]
                Y_of = Y_of[:,start:end,[mode]]
            else:
                Y = Y[:,start:end,:]
                Y_scaled = Y_scaled[:,start:end,:]
                Y_of = Y_of[:,start:end,:]

            X_scaled = X_scaled.type(torch.FloatTensor).to(device, non_blocking=True)
            Y_scaled = Y_scaled.type(torch.FloatTensor).to(device, non_blocking=True)
            Y = Y.type(torch.FloatTensor).to(device, non_blocking=True)
            Y_of = Y_of.type(torch.FloatTensor).to(device, non_blocking=True)

            output_r, output_c = t_net.model(X_scaled)
            y_pred_r.append(output_r.cpu())
            y_true_scaled_r.append(Y_scaled.cpu())
            y_true_r.append(Y.cpu())
            x_true.append(X[:,start:end,:])

            y_pred_c.append(output_c.cpu())
            y_true_c.append(Y_of.cpu())

        out_pred_scaled_r = torch.vstack(y_pred_r)
        out_true_scaled_r = torch.vstack(y_true_scaled_r)
        out_true_r = torch.vstack(y_true_r)
        out_true_x = torch.vstack(x_true)
        out_pred_scaled_c = torch.vstack(y_pred_c)
        out_true_c = torch.vstack(y_true_c)

    return out_pred_scaled_r, out_true_scaled_r, out_true_r, out_pred_scaled_c, out_true_c, out_true_x

