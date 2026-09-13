import argparse
import logging
import random
import pickle
from thop import profile
from thop import clever_format


import numpy as np
np.set_printoptions(threshold=np.inf)
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.nn import functional as F


import datasets.datasetfactory as df
import model.learner as learner
import model.modelfactory as mf
import utils
from experiment.experiment import experiment
from model.fseml_model import FSEMLModel, FSSAEConfig

logger = logging.getLogger('experiment')

def pickle_dict(dictionary, filename): 
    p = pickle.Pickler(open("{0}.p".format(filename),"wb")) 
    p.fast = True 
    p.dump(dictionary) 


def is_fseml_model(model):
    return isinstance(model, FSEMLModel)


def get_eval_split_flags(args):
    if args.dataset == "omniglot":



        return True, False

    train_flag = not args.test
    return train_flag, train_flag


def create_scratch_model(args):
    if args.FSEML:
        return FSEMLModel(FSSAEConfig(num_classes=1000))

    if args.Neuromodulation:
        model_type = "Neuromodulation"
    elif args.AE:
        model_type = "AE"
    else:
        model_type = "OML"

    config = mf.ModelFactory.get_model(model_type, args.dataset)
    return learner.Learner(config)


def predict_seen_classes_only(logits, keep, device):
    keep_tensor = torch.as_tensor(keep, device=device, dtype=torch.long)
    visible_logits = logits.index_select(dim=1, index=keep_tensor)
    local_pred = visible_logits.argmax(dim=1)
    return keep_tensor[local_pred]


def keep_class_cross_entropy(logits, targets, keep, device):
    keep_tensor = torch.as_tensor(keep, device=device, dtype=torch.long)
    visible_logits = logits.index_select(dim=1, index=keep_tensor)

    target_map = {int(cls): idx for idx, cls in enumerate(keep)}
    local_targets = torch.as_tensor(
        [target_map[int(label)] for label in targets.detach().cpu().tolist()],
        device=device,
        dtype=torch.long,
    )
    return F.cross_entropy(visible_logits, local_targets)


def evaluate_iterator_accuracy(model, iterator, keep, device):
    correct = 0.0
    total = 0
    for img, target in iterator:
        with torch.no_grad():
            img = img.to(device)
            target = target.to(device)
            _, logits_q = model(img, vars=None, bn_training=False, feature=False)
            pred_q = predict_seen_classes_only(logits_q, keep, device)
            correct += torch.eq(pred_q, target).sum().item()
            total += len(img)

    if total == 0:
        return 0.0
    return correct / total


def configure_eval_model(model, args, frozen_layers, device):
    is_fseml = is_fseml_model(model)

    for name, param in model.named_parameters():
        param.learn = True

    if is_fseml:


        for name, param in model.named_parameters():
            if args.no_freeze:
                param.learn = True
            elif args.fseml_eval_mode == "output-only":
                param.learn = name.startswith("cpn.output.")
            else:
                param.learn = name.startswith("cpn.")
            if args.reset and param.learn:
                if len(param.shape) > 1:
                    torch.nn.init.kaiming_normal_(param)
                else:
                    param.data.zero_()

        torch.nn.init.kaiming_normal_(model.cpn.output.weight)
        model.cpn.output.bias.data.zero_()
        return

    for name, param in model.named_parameters():
        if name in frozen_layers:
            param.learn = False
        elif args.reset:
            w = nn.Parameter(torch.ones_like(param))
            if len(w.shape) > 1:
                torch.nn.init.kaiming_normal_(w)
            else:
                w = nn.Parameter(torch.zeros_like(param))
            param.data = w
            param.learn = True

    torch.nn.init.kaiming_normal_(model.parameters()[-2])
    w = nn.Parameter(torch.zeros_like(model.parameters()[-1]))
    model.parameters()[-1].data = w

    if args.Neuromodulation:
        weights2reset = ["vars_26"]
        biases2reset = ["vars_27"]
    elif args.AE:
        weights2reset = ["vars_14"]
        biases2reset = ["vars_15"]
    else:
        weights2reset = ["vars_14"]
        biases2reset = ["vars_15"]

    for n, a in model.named_parameters():
        n = n.replace(".", "_")
        if n in weights2reset:
            w = nn.Parameter(torch.ones_like(a)).to(device)
            torch.nn.init.kaiming_normal_(w)
            a.data = w
        if n in biases2reset:
            w = nn.Parameter(torch.zeros_like(a)).to(device)
            a.data = w

def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    my_experiment = experiment(args.name, args, "../results/", args.commit)
    writer = SummaryWriter(my_experiment.path + "tensorboard")

    logger = logging.getLogger('experiment')
    logger.setLevel(logging.INFO)


    def get_omniglot_eval_dataset(train):
        return df.DatasetFactory.get_dataset(
            "omniglot",
            train=train,
            background=False,
            path=args.dataset_path,
        )

    if args.dataset == "omniglot":
        eval_class_dataset = get_omniglot_eval_dataset(train=True)
        eval_class_ids = list(range(int(np.max(eval_class_dataset.targets)) + 1))
    else:
        eval_class_ids = None

    finetune_train_flag, eval_test_flag = get_eval_split_flags(args)

    frozen_layers = []
    for temp in range(args.rln * 2):
        frozen_layers.append("vars." + str(temp))
    logger.info("Frozen layers = %s", " ".join(frozen_layers))
    for v in range(4):
       frozen_layers.append("vars_bn.{0}".format(v))

    final_results_all = []
    temp_result = []
    total_clases = args.schedule
    for tot_class in total_clases:
        lr_list = args.lr_list

        lr_all = []

        for lr_search in range(1):
            keep = np.random.choice(eval_class_ids, tot_class, replace=False)




            dataset = utils.remove_classes_omni(
                get_omniglot_eval_dataset(train=finetune_train_flag), keep)
            iterator_sorted = torch.utils.data.DataLoader(
                utils.iterator_sorter_omni(dataset, False, classes=total_clases),
                batch_size=1,
                shuffle=args.iid, num_workers=args.num_workers)
            dataset = utils.remove_classes_omni(
                get_omniglot_eval_dataset(train=eval_test_flag), keep)
            iterator = torch.utils.data.DataLoader(dataset, batch_size=1,



                                                   shuffle=False, num_workers=args.num_workers)




            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')

            results_mem_size = {}

            for mem_size in [args.memory]:
                max_acc = -10
                max_lr = -10
                for lr in lr_list:


                    maml = torch.load(args.model, map_location='cpu', weights_only=False)

                    if args.scratch:
                        maml = create_scratch_model(args)


                    maml = maml.to(device)
                    configure_eval_model(maml, args, frozen_layers, device)
                       
                    filter_list = ["vars.{0}".format(v) for v in range(12)]

                    logger.info("Filter list = %s", ",".join(filter_list))




                    list_of_params = list(filter(lambda x: x.learn, maml.parameters()))
                    list_of_names = list(filter(lambda x: x[1].learn, maml.named_parameters()))




                    if args.scratch or args.no_freeze:
                        print("Empty filter list")
                        list_of_params = maml.parameters()

                    for x in list_of_names:
                        logger.info("Unfrozensss layer = %s", str(x[0]))
                    opt = torch.optim.Adam(list_of_params, lr=lr)

                    for _ in range(0, args.epoch):
                        for img, y in iterator_sorted:

                            img = img.to(device)
                            y = y.to(device)
                            y = y.long()

                            _, pred = maml(img)
                            opt.zero_grad()
                            if args.restrict_finetune_to_seen_classes:
                                loss = keep_class_cross_entropy(pred, y, keep, device)
                            else:
                                loss = F.cross_entropy(pred, y)
                            loss.backward()
                            opt.step()

                    train_acc = evaluate_iterator_accuracy(maml, iterator_sorted, keep, device)
                    test_acc = evaluate_iterator_accuracy(maml, iterator, keep, device)
                    logger.info("Meta-test train accuracy after fine-tune (lr=%f) = %s", lr, str(train_acc))
                    logger.info("Meta-test test accuracy after fine-tune (lr=%f) = %s", lr, str(test_acc))
                    logger.info(str(correct / len(iterator)))
                    if test_acc > max_acc:
                        max_acc = test_acc
                        max_lr = lr

                lr_all.append(max_lr)
                results_mem_size[mem_size] = (max_acc, max_lr)
                logger.info("Final Max Result = %s", str(max_acc))
                writer.add_scalar('/finetune/best_' + str(lr_search), max_acc, tot_class)
            temp_result.append((tot_class, results_mem_size))
            logger.info("Temp Results = %s", str(results_mem_size))

            my_experiment.results["Temp Results"] = temp_result
            my_experiment.store_json()
            print("LR RESULTS = ", temp_result)
            logger.info("LR RESULTS = %s", temp_result)

        from scipy import stats
        best_lr = float(np.asarray(stats.mode(lr_all, keepdims=True).mode).reshape(-1)[0])
        logger.info("BEST LR %s= ", str(best_lr))

        for aoo in range(args.runs):

            keep = np.random.choice(eval_class_ids, tot_class, replace=False)
            
            if args.dataset == "omniglot":

                dataset = utils.remove_classes_omni(
                    get_omniglot_eval_dataset(train=finetune_train_flag), keep)
                iterator_sorted = torch.utils.data.DataLoader(
                    utils.iterator_sorter_omni(dataset, False, classes=total_clases),
                    batch_size=1,
                    shuffle=args.iid, num_workers=args.num_workers)
                dataset = utils.remove_classes_omni(
                    get_omniglot_eval_dataset(train=eval_test_flag), keep)
                iterator = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                       shuffle=False, num_workers=args.num_workers)
            elif args.dataset == "CIFAR100":
                keep = np.random.choice(list(range(50, 100)), tot_class)
                dataset = utils.remove_classes(df.DatasetFactory.get_dataset(args.dataset, train=True), keep)
                iterator_sorted = torch.utils.data.DataLoader(
                    utils.iterator_sorter(dataset, False, classes=tot_class),
                    batch_size=16,
                    shuffle=args.iid, num_workers=args.num_workers)
                dataset = utils.remove_classes(df.DatasetFactory.get_dataset(args.dataset, train=False), keep)
                iterator = torch.utils.data.DataLoader(dataset, batch_size=128,
                                                       shuffle=False, num_workers=args.num_workers)
            print(args)

            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')

            results_mem_size = {}

            for mem_size in [args.memory]:
                max_acc = -10
                max_lr = -10

                lr = best_lr

                maml = torch.load(args.model, map_location='cpu', weights_only=False)

                if args.scratch:
                    maml = create_scratch_model(args)

                maml = maml.to(device)
                configure_eval_model(maml, args, frozen_layers, device)
                pre_epoch_train_acc = evaluate_iterator_accuracy(maml, iterator_sorted, keep, device)
                pre_epoch_test_acc = evaluate_iterator_accuracy(maml, iterator, keep, device)
                logger.info("Pre-epoch meta-test train accuracy %s", str(pre_epoch_train_acc))
                logger.info("Pre-epoch meta-test test accuracy %s", str(pre_epoch_test_acc))
                filter_list = ["vars.{0}".format(v) for v in range(12)]

                logger.info("Filter list = %s", ",".join(filter_list))
               



                list_of_params = list(filter(lambda x: x.learn, maml.parameters()))
                list_of_names = list(filter(lambda x: x[1].learn, maml.named_parameters()))
                if args.scratch or args.no_freeze:
                    print("Empty filter list")
                    list_of_params = maml.parameters()

                for x in list_of_names:
                    logger.info("Unfrozen layer = %s", str(x[0]))
                opt = torch.optim.Adam(list_of_params, lr=lr)

                for _ in range(0, args.epoch):
                    for img, y in iterator_sorted:
                        img = img.to(device)
                        y = y.to(device)
                        y = y.long()
                        _, pred = maml(img)

                        opt.zero_grad()
                        if args.restrict_finetune_to_seen_classes:
                            loss = keep_class_cross_entropy(pred, y, keep, device)
                        else:
                            loss = F.cross_entropy(pred, y)
                        loss.backward()
                        opt.step()

                post_epoch_train_acc = evaluate_iterator_accuracy(maml, iterator_sorted, keep, device)
                post_epoch_test_acc = evaluate_iterator_accuracy(maml, iterator, keep, device)
                logger.info("Post-epoch meta-test train accuracy %s", str(post_epoch_train_acc))
                logger.info("Post-epoch meta-test test accuracy %s", str(post_epoch_test_acc))
                if post_epoch_test_acc > max_acc:
                    max_acc = post_epoch_test_acc
                    max_lr = lr

                lr_list = [max_lr]
                results_mem_size[mem_size] = (max_acc, max_lr)
                logger.info("Final Max Result = %s", str(max_acc))
                writer.add_scalar('/finetune/best_' + str(aoo), max_acc, tot_class)
            final_results_all.append((tot_class, results_mem_size))
            print("A=  ", results_mem_size)
            logger.info("Final results = %s", str(results_mem_size))

            my_experiment.results["Final Results"] = final_results_all
            my_experiment.store_json()
            print("FINAL RESULTS = ", final_results_all)

    writer.close()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=5)
    argparser.add_argument('--seed', type=int, help='epoch number', default=222)



    argparser.add_argument('--schedule', type=int, nargs='+', default=[10, 50, 75, 100, 200, 300, 400, 500, 600],
                        help='Decrease learning rate at these epochs.')
    argparser.add_argument('--memory', type=int, help='epoch number', default=0)
    argparser.add_argument('--model', type=str, help='epoch number', default="../PreNet/FSEML_Model_20260422_lat128_from93.net")
    argparser.add_argument('--scratch', action='store_true', default=False)

    argparser.add_argument('--dataset', help='Name of experiment', default="omniglot")
    argparser.add_argument('--dataset-path', help='Name of experiment', default=None)
    argparser.add_argument('--num-workers', type=int, default=4)
    argparser.add_argument('--lr-list', type=float, nargs='+', default=[0.001, 0.0005, 0.0002, 0.0001])
    argparser.add_argument('--name', help='Name of experiment', default="eval_fseml_20260427_main")
    argparser.add_argument("--commit", action="store_true")
    argparser.add_argument("--no-freeze", action="store_true")
    argparser.add_argument('--reset', action="store_true")
    argparser.add_argument('--test', action="store_true")
    argparser.add_argument("--iid", action="store_true")
    argparser.add_argument("--rln", type=int, default=0)
    argparser.add_argument("--runs", type=int, default=10)
    argparser.add_argument("--AE", action="store_true")
    argparser.add_argument("--FSEML", action="store_true", default=True)
    argparser.add_argument("--fseml-eval-mode", choices=["full-cpn", "output-only"], default="full-cpn")
    argparser.add_argument("--restrict-finetune-to-seen-classes", action="store_true", default=True)
    argparser.add_argument("--Neuromodulation", action="store_true")



    args = argparser.parse_args()

    import os

    args.name = "/".join([args.dataset, "eval", str(args.epoch).replace(".", "_"), args.name])

    main(args)

"""
1）iterator_sorted:在元测试训练时是采样训练集。元测试测试时采样训练集。  （75,150,225...）
   iterator：在元测试训练时是采样训练集（75,150,225...）。元测试测试时采样测试集（25,50,75...）。
   一轮次用5个类，每类测试集：训练集=15：5
   
2) 代码结构：
①for tot_class in total_clases:对于660个测试任务，tot-class=10,50,75
    ②for lr_search in range(1):对于10个独立模型分别元测试
        ③for mem_size in [args.memory]:对重放区取出的大小默认=0
    ②for aoo in range(args.runs):对于10个独立模型分别元测试
        ③for mem_size in [args.memory]:对重放区取出的大小默认=0
"""
