from .server import Server
from loguru import logger
import numpy as np
from FedUtils.models.utils import decode_stat
import torch

def _aggregate(self, wstate_dicts):
    old_params = self.get_param()
    state_dict = {x: 0.0 for x in self.get_param()}
    wtotal = 0.0
    for w, st in wstate_dicts:
        wtotal += w
        for name in state_dict.keys():
            if "bn" in name:  # skip batch normalization parameters
                continue
            assert name in state_dict
            state_dict[name] += st[name]*w
    state_dict = {x: state_dict[x]/wtotal for x in state_dict}
    return state_dict

def aggregate(self, wstate_dicts):
    state_dict = self._aggregate(wstate_dicts)
    state_dict = {k: v for k, v in state_dict.items() if "bn" not in k}  # remove batch normalization parameters from the state_dict
    return self.set_param(state_dict)

def step_func(model, data):
    lr = model.learning_rate
    parameters = list(model.parameters())
    flop = model.flop

    def func(d):
        nonlocal flop, lr
        model.train()
        model.zero_grad()
        x, y = d
        pred = model.forward(x)
        loss = model.loss(pred, y).mean()
        grad = torch.autograd.grad(loss, parameters)
        for p, g in zip(parameters, grad):
            p.data.add_(-lr*g)
        return flop*len(x)
    return func


class FedBN(Server):
    step = 0

    def train(self):
        logger.info("Train with {} workers...".format(self.clients_per_round))
        for r in range(self.num_rounds):
            if r % self.eval_every == 0:
                logger.info("-- Log At Round {} --".format(r))
                stats = self.test()
                if self.eval_train:
                    stats_train = self.train_error_and_loss()
                else:
                    stats_train = stats
                logger.info("-- TEST RESULTS --")
                decode_stat(stats)
                logger.info("-- TRAIN RESULTS --")
                decode_stat(stats_train)

            indices, selected_clients = self.select_clients(r, num_clients=self.clients_per_round)
            np.random.seed(r)
            active_clients = np.random.choice(selected_clients, round(self.clients_per_round*(1.0-self.drop_percent)), replace=False)
            csolns = {}
            w = 0

            for idx, c in enumerate(active_clients):
                c.set_param(self.model.get_param())
                soln, stats = c.solve_inner(num_epochs=self.num_epochs, step_func=step_func)  # stats has (byte w, comp, byte r)
                soln = [1.0, soln[1]]
                w += soln[0]
                if len(csolns) == 0:
                    csolns = {x: soln[1][x].detach()*soln[0] for x in soln[1]}
                else:
                    for x in csolns:
                        csolns[x].data.add_(soln[1][x]*soln[0])
                del c
            csolns = [[w, {x: csolns[x]/w for x in csolns}]]

            self.latest_model = self.aggregate(csolns)

        logger.info("-- Log At Round {} --".format(r))
        stats = self.test()
        if self.eval_train:
            stats_train = self.train_error_and_loss()
        else:
            stats_train = stats
        logger.info("-- TEST RESULTS --")
        decode_stat(stats)
        logger.info("-- TRAIN RESULTS --")
        decode_stat(stats_train)