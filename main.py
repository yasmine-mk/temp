import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import importlib
import random
import torch
from FedUtils.models.utils import read_data, CusDataset, medmnist_to_json_train, medmnist_to_json_test
import json
from torch.utils.data import DataLoader
from loguru import logger
from functools import partial
import os
torch.backends.cudnn.deterministic = True


def allocate_memory():
    total, used = os.popen(
        '"nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader'
    ).read().split('\n')[0].split(',')
    total = int(total)
    total = int(total * 0.7)
    n = torch.cuda.device_count()
    for _ in range(n):
        x = torch.rand((256, 1024, total)).cuda(_)
        del x

def plot_loss_acc(log_path, ds_name):
    train_acc_regex = r"- Accuracy: (\d\.\d\d\d\d\d)"
    test_acc_regex = r"\d Accuracy: (\d\.\d\d\d\d\d)"
    train_loss_regex = r"Loss: (\d\.\d\d\d\d\d)"
    with open(log_path, "r") as f:
        logs = " ".join(f)
    train_accuracy = re.findall(train_acc_regex, logs)  # extracts the last train accuracy value
    test_accuracy = re.findall(test_acc_regex, logs)
    train_loss = re.findall(train_loss_regex, logs) # extracts the last test accuracy value
    train_accuracy, test_accuracy, train_loss=  [float(i) for i in train_accuracy], [float(i) for i in test_accuracy], [float(i) for i in train_loss]  
    if not os.path.exists("./results"):
        os.mkdir("results")

    plt.plot(train_accuracy)
    plt.title(f"{ds_name} Training Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.savefig(f"./results/{ds_name}_Training_Accuracy", dpi=300)
    
    plt.plot(test_accuracy)
    plt.title(f"{ds_name} Test Accuracy")
    plt.savefig(f"./results/{ds_name}_Test_Accuracy", dpi=300)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.plot(train_loss)
    plt.title(f"{ds_name} Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(f"./results/{ds_name}_Train_Loss", dpi=300)

def makedir(path):
    cwd = os.getcwd()
    path = path.split("/")
    print(path)
    temp = ""
    for folder in path:
        temp += "/"+ folder
        os.mkdir(os.path.join(cwd, temp))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="The config file")
    args = parser.parse_args()
    config = importlib.import_module(args.config.replace("/", "."))
    config = config.config
    logger.add(config["log_path"])

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])

    Model = config["model"]
    inner_opt = config["inner_opt"]
    ds_name = config["dataset_name"]
    train_path = os.path.join(config["train_path"], ds_name)
    test_path = os.path.join(config["test_path"], ds_name)
    if os.path. exists(train_path):
        clients, groups, train_data, eval_data = read_data(train_path, test_path)
    else:
        os.makedirs(train_path)
        os.makedirs(test_path)
        with open(os.path.join(test_path, "test.json"),"w") as f:
            data_test = medmnist_to_json_test(ds_name, num_clients=config["clients_per_round"])
            json.dump(data_test, f)
            del data_test
        with open(os.path.join(train_path, "train.json"), "w") as f:
            data_train = medmnist_to_json_train(ds_name, num_clients=config["clients_per_round"], iid=config["iid"]) 
            json.dump(data_train, f)
            del data_train
        clients, groups, train_data, eval_data = read_data(train_path, test_path)
    
    # if not os.path.exists(os.path.join(config["train_path"], ds_name)):
    #     clients, groups, train_data, eval_data = read_data(os.path.join(config["train_path"], ds_name), os.path.join(config["test_path"], ds_name))
    # else:
    #     if not os.path.exists(os.path.join(config["test_path"], ds_name)):
    #         os.mkdir(os.path.join(config["test_path"], ds_name))
    #     if not os.path.exists(os.path.join(config["train_path"], ds_name)):
    #         os.mkdir(os.path.join(config["train_path"], ds_name))
    #     with open(os.path.join(config["test_path"], ds_name, "test.json"),"w") as f:
    #         data_test = medmnist_to_json_test(ds_name, num_clients=config["clients_per_round"])
    #         json.dump(data_test, f)
    #         del data_test
    #     with open(os.path.join(config["train_path"], ds_name, "train.json"), "w") as f:
    #         data_train = medmnist_to_json_train(ds_name, num_clients=config["clients_per_round"], iid=config["iid"]) 
    #         json.dump(data_train, f)
    #         del data_train
    #     clients, groups, train_data, eval_data = read_data(os.path.join(config["train_path"], ds_name), os.path.join(config["test_path"], ds_name))
    log_file = open(config["log_path"],"r+")
    log_file.truncate(0)
    log_file.close()
    Dataset = CusDataset

    if config["use_fed"]:
        Optimizer = config["optimizer"]
        t = Optimizer(config, Model, [clients, groups, train_data, eval_data], train_transform=config["train_transform"],
                      test_transform=config['test_transform'], traincusdataset=Dataset, evalcusdataset=Dataset)
        t.train()
    else: # in the non IID setting
        train_data_total = {"x": [], "y": []}
        eval_data_total = {"x": [], "y": []}
        for t in train_data:
            train_data_total["x"].extend(train_data[t]["x"])
            train_data_total["y"].extend(train_data[t]["y"])
        for t in eval_data:
            eval_data_total["x"].extend(eval_data[t]["x"])
            eval_data_total["y"].extend(eval_data[t]["y"])
        train_data_size = len(train_data_total["x"])
        eval_data_size = len(eval_data_total["x"])
        train_data_total_fortest = DataLoader(Dataset(train_data_total, config["test_transform"]), batch_size=config["batch_size"], shuffle=False,)
        train_data_total = DataLoader(Dataset(train_data_total, config["train_transform"]), batch_size=config["batch_size"], shuffle=True, )
        eval_data_total = DataLoader(Dataset(eval_data_total, config["test_transform"]), batch_size=config["batch_size"], shuffle=False,)
        model = Model(*config["model_param"], optimizer=inner_opt)
        for r in range(config["num_rounds"]):
            model.solve_inner(train_data_total)
            stats = model.test(eval_data_total)
            train_stats = model.test(train_data_total_fortest)
            logger.info("-- Log At Round {} --".format(r))
            logger.info("-- TEST RESULTS --")
            logger.info("Accuracy: {}".format(stats[0]*1.0/eval_data_size))
            logger.info("-- TRAIN RESULTS --")
            logger.info(
                "Accuracy: {} Loss: {}".format(train_stats[0]/train_data_size, train_stats[1]/train_data_size))
    plot_loss_acc(config["log_path"], ds_name=config["dataset_name"])


    
if __name__ == "__main__":
    main()