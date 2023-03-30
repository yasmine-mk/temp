from thop import profile
import os
import json
from torch.utils.data import TensorDataset
import numpy as np
import torch
from loguru import logger
import medmnist
from medmnist import INFO
# PIL is now renamed to Pillow
from PIL import Image
import h5py


def FSGM(model, inp, label, iters, eta):
    '''
    the function implements the FGSM attack to generate adversarial examples 
    for the given model, inp, and label
    this function is usually called from the generate fake method from a Model class
    this usually the model argument will take the value of self, called from Model 
    '''
    
    inp.requires_grad = True
    minv, maxv = float(inp.min().detach().cpu().numpy()), float(inp.max().detach().cpu().numpy())
    for _ in range(iters):
        loss = model.loss(model.forward(inp), label).mean()
        dp = torch.sign(torch.autograd.grad(loss, inp)[0])
        inp.data.add_(eta*dp.detach()).clamp(minv, maxv)
    return inp

# traincusdataset(train_data, transform=train_transform)
class CusDataset(TensorDataset):
    def __init__(self, data, transform=None):
        assert "x" in data
        assert "y" in data
        self.data = {}
        self.data["x"] = (data["x"])
        self.data["y"] = (data["y"])
        self.transform = transform

    def __getitem__(self, item):
        if self.transform is None:
            ret = torch.tensor(self.data['x'][item])
        else:
            ret = np.array(self.data["x"][item]).astype("uint8")
            if ret.shape[-1] == 3:
                ret = ret
            elif ret.shape[0] == 3:
                ret = ret.transpose(1, 2, 0)
            else:
                ret = ret
            ret = self.transform(Image.fromarray(ret))

        return [ret, torch.tensor(self.data["y"][item])]

    def __len__(self):
        return len(self.data["x"])


class ImageDataset(TensorDataset):
    def __init__(self,  data, transform=None, image_path=None):
        self.transform = transform

        assert "x" in data
        assert "y" in data
        self.data = {}
        self.data["x"] = (data["x"])
        self.data["y"] = (data["y"])
        if len(self.data["x"]) < 20000:
            File = h5py.File(image_path, "r")
            self.image_path = {}
            for name in self.data["x"]:
                name = name.replace(".png", "")
                self.image_path[name+"_X"] = np.array(File[name+"_X"])
                self.image_path[name+"_Y"] = np.array(File[name+"_Y"])
            File.close()
        else:
            self.image_path = h5py.File(image_path, "r")

    def __getitem__(self, item):
        path = self.data["x"][item]
        path = path.replace(".png", "")
        image, y = Image.fromarray((np.array(self.image_path[path+"_X"])*255).transpose(1, 2, 0).astype(np.uint8)), self.image_path[path+"_Y"]
        if self.transform is None:
            ret = torch.tensor(image)
        else:
            try:
                assert image.mode == "RGB"
            except:
                image = image.convert("RGB")
            ret = self.transform(image)

        return [ret, torch.tensor(self.data["y"][item])]

    def __len__(self):
        return len(self.data["x"])


def Flops(model, inp):
    return profile(model, inputs=(inp,), verbose=False)[0]


def read_data(train_data_path, test_data_path):
    if not isinstance(test_data_path, list):
        test_data_path = [test_data_path, ]
    groups = []
    train_data = {}
    test_data = [{} for _ in test_data_path]
    train_files = os.listdir(train_data_path)
    train_files = [f for f in train_files if f.endswith(".json")]
    for f in train_files:
        file_path = os.path.join(train_data_path, f)
        with open(file_path, "r") as inf:
            cdata = json.load(inf)
        if "hierarchies" in cdata:
            groups.extend(cdata["hierarchies"])
        train_data.update(cdata["user_data"])
    for F, td in zip(test_data_path, test_data):
        test_files = os.listdir(F)
        test_files = [f for f in test_files if f.endswith(".json")]
        for f in test_files:
            file_path = os.path.join(F, f)
            with open(file_path, "r") as inf:
                cdata = json.load(inf)
            td.update(cdata["user_data"])
    clients = list(sorted(train_data.keys()))
    return clients, groups, train_data, test_data


def decode_stat(stat):
    # global train_loss, train_acc, test_loss, test_acc
    if len(stat) == 4:
        ids, groups, num_samples, tot_correct = stat
        if isinstance(num_samples[0], list):
            assert len(num_samples) == len(tot_correct)
            idx = 0
            for a, b in zip(tot_correct, num_samples):
                logger.info("Test_{} Accuracy: {}".format(idx, sum(a) * 1.0 / sum(b)))
                # test_acc.append(sum(a) * 1.0 / sum(b))
                idx += 1
        else:
            # train_acc.append(sum(tot_correct) / sum(num_samples))
            logger.info("Accuracy: {}".format(sum(tot_correct) / sum(num_samples)))
    elif len(stat) == 5:
        ids, groups, num_samples, tot_correct, losses = stat
        logger.info("Accuracy: {} Loss: {}".format(sum(tot_correct) / sum(num_samples), sum(losses) / sum(num_samples)))
    else:
        raise ValueError

def non_iid_partition(dataset, num_clients):
    """
    non I.I.D parititioning of data over clients
    Sort the data by the digit label
    Divide the data into N shards of size S
    Each of the clients will get X shards

    params:
      - dataset (torch.utils.Dataset): Dataset containing the pathMNIST Images
      - num_clients (int): Number of Clients to split the data between
      - total_shards (int): Number of shards to partition the data in
      - shards_size (int): Size of each shard 
      - num_shards_per_client (int): Number of shards of size shards_size that each client receives

    returns:
      - Dictionary of image indexes for each client
    """
    shards_size = 9
    total_shards = len(dataset)// shards_size
    num_shards_per_client = total_shards // num_clients
    shard_idxs = [i for i in range(total_shards)]
    client_dict = {i: np.array([], dtype='int64') for i in range(num_clients)}
    idxs = np.arange(len(dataset))
    # get labels as a numpy array
    data_labels = np.array([np.array(target).flatten() for _, target in dataset]).flatten()
    # sort the labels
    label_idxs = np.vstack((idxs, data_labels))
    label_idxs = label_idxs[:, label_idxs[1,:].argsort()]
    idxs = label_idxs[0,:]

    # divide the data into total_shards of size shards_size
    # assign num_shards_per_client to each client
    for i in range(num_clients):
        rand_set = set(np.random.choice(shard_idxs, num_shards_per_client, replace=False))
        shard_idxs = list(set(shard_idxs) - rand_set)

        for rand in rand_set:
            client_dict[i] = np.concatenate((client_dict[i], idxs[rand*shards_size:(rand+1)*shards_size]), axis=0)
    return client_dict # client dict has [idx: list(datapoint indices)


def iid_partition(dataset, clients):
    """
    I.I.D paritioning of data over clients
    Shuffle the data
    Split it between clients
    params:
      - dataset (torch.utils.Dataset): Dataset containing the PathMNIST Images 
      - clients (int): Number of Clients to split the data between
    returns:
      - Dictionary of image indexes for each client
    """
    num_items_per_client = int(len(dataset)/clients)
    client_dict = {}
    image_idxs = [i for i in range(len(dataset))]
    for i in range(clients):
        client_dict[i] = set(np.random.choice(image_idxs, num_items_per_client, replace=False))
        image_idxs = list(set(image_idxs) - client_dict[i])
    return client_dict

def medmnist_to_json_train(ds_name, num_clients,iid=False) -> dict:
    info = INFO[ds_name]
    DataClass = getattr(medmnist, info["python_class"])
    dataset = DataClass(root="./data", download=True,split="train")    
    len_dataset = len(dataset)
    return_dict = {"users": [], "user_data":{}, "num_samples":0}
    users = []
    for i in range(num_clients):
        users.append(f"f_{i}")
    if iid:
        clients_dict = iid_partition(dataset, num_clients)
    else:
        clients_dict = non_iid_partition(dataset, num_clients)
    return_dict["num_samples"] = [len(i) for i in clients_dict.values()]
    # users = ["f_0", "f_1"..]
    # client_dict = {0: [1, 2, 5,12,..]}
    for user, client_dict in zip(users, clients_dict.values()):
        # print(user)
        # print(client_dict)
        return_dict["user_data"][user] = {"x":[[int(i)/255 for i in list(np.array(dataset[image][0]).reshape((28*28)))] for image in client_dict],
                                          "y":[float(dataset[image][1].item()) for image in client_dict]}
    return_dict["users"] = users
    return return_dict

def medmnist_to_json_test(ds_name, num_clients) -> dict:
    info = INFO[ds_name]
    DataClass = getattr(medmnist, info["python_class"])
    dataset = DataClass(root="./data", download=True,split="test")    
    len_dataset = len(dataset)
    return_dict = {"users": [], "user_data":{}, "num_samples":[]}
    users = []
    for i in range(num_clients):
        users.append(f"f_{i}")
    
    return_dict["num_samples"] = [0 for i in range(num_clients)]
    return_dict["num_samples"][0] = len_dataset
    # users = ["f_0", "f_1"..]
    # client_dict = {0: [1, 2, 5,12,..]}
    clients_dict = iid_partition(dataset, num_clients)
    for user, client_dict in zip(users, clients_dict.values()):
        # print(user)
        # print(client_dict)
        return_dict["user_data"][user] = {"x":[[int(i)/255 for i in list(np.array(dataset[image][0]).reshape((28*28)))] for image in client_dict],
                                          "y":[float(dataset[image][1].item()) for image in client_dict]}
    return_dict["users"] = users
    return return_dict