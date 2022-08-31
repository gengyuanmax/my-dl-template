from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from matplotlib.pyplot import step


import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader


import os
import logging
import argparse
import time
import socket
import numpy as np
import mlflow
from scipy.linalg import block_diag
from torchsummary import summary


from common.config import Config
from common.util import parallel_apply
from model.model import TemplateModel
from dataloaders.dataloader import get_dataset
from model.layer import BertAdam


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', default=False, help='train the model')
    parser.add_argument('--resume', action='store_true', default=False, help='test the model')
    parser.add_argument('--test', action='store_true', default=False, help='test the model')
    parser.add_argument('--val', type=str, default=True, help='validate the model')

    parser.add_argument('--run_id', type=str, default=None, help='set run id to resume')

    parser.add_argument('--config', type=str, help='path to config file')
    parser.add_argument('--world_size', type=int, default=1, help='number of nodes')
    parser.add_argument('--rank', type=int, default=0, help='rank of node')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank of node')
    parser.add_argument('--backend', type=str, default='nccl', help='backend for distributed training')
    parser.add_argument('--init_method', type=str, default='env://', help='init method for distributed training')

    args = parser.parse_args()

    if not (args.train or args.resume or args.test):
        raise ValueError("Please specify either --train or --resume or --test")

    if args.train and args.resume:
        raise ValueError("--train and --resume cannot be used at the same time")

    return args

def init_device(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", config.args.local_rank)

    n_gpu = torch.cuda.device_count()    
    config.args.n_gpu = n_gpu

    if config.train.batch_size % n_gpu != 0 or config.val.batch_size % n_gpu != 0:
        raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
            config.train.batch_size, n_gpu, config.val.batch_size, n_gpu))

    return device, n_gpu

def init_model(config, device):
    # Train from scratch
    if config.args.train:
        model = TemplateModel.from_config(config)
        # model = PerceiverModel.from_config(config)

        if config.args.local_rank == 0:
            model.log_params()
        kwargs = None

    # Load from checkpoint
    if config.args.resume or config.args.test:
        model, kwargs = TemplateModel.from_pretrained(config)
        # model, kwargs = PerceiverModel.from_pretrained(config)

    if hasattr(model, 'module'):
        model = model.module

    model.to(device=device)

    return model, kwargs

def init_optimizer(config, model, num_train_optimization_steps):
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    weight_decay = 0.2
    optimizer_grouped_parameters = [
        {'params': [p for n, p in decay_param_tp], 'weight_decay': weight_decay},
        {'params': [p for n, p in no_decay_param_tp], 'weight_decay': 0.0},

    ]

    optimizer = BertAdam(optimizer_grouped_parameters, lr=config.train.lr, warmup=config.train.warmup_proportion,
                         schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                         t_total=num_train_optimization_steps, weight_decay=weight_decay,
                         max_grad_norm=1.0)
    scheduler = None

    return optimizer, scheduler


def train_epoch(config, model, optimizer, scheduler, train_dataloader, device, n_gpu, epoch):    
    torch.cuda.empty_cache()
    model.train()

    start_time = time.time()
    total_loss = 0.

    for step, batch in enumerate(train_dataloader):
        if n_gpu == 1:
            # multi-gpu does scattering it-self
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        ## ####################################
        # customized code 
        ## ####################################  
        vis_feat, text_feat, vis_mask, text_mask = batch
        logits = model(vis_feat, text_feat, vis_mask, text_mask) # [batch_size, max_len_text]

        # compute loss
        ind_gt = block_diag(*[[1] * int(nc) for nc in text_mask.sum(dim=1)])
        ind_gt = torch.tensor(ind_gt, dtype=logits.dtype, device=logits.device)

        loss1 = model.module.loss(logits, ind_gt)
        loss2 = model.module.loss(logits.T, ind_gt.T)
        loss = (loss1 + loss2)/2


        ## ####################################

        if n_gpu > 1:
            loss = loss.mean()
        if config.train.gradient_accumulation_steps > 1:
            loss = loss / config.train.gradient_accumulation_steps

        loss.backward()

        #######
        # debug
        #######
        if config.args.local_rank == 0:
            for n, p in model.named_parameters():
                if p.grad is not None and n.find('to_') != -1:
                    config.writer.add_histogram(n, p.data.cpu().numpy(), epoch)


        total_loss += loss.item()

        if (step + 1) % config.train.gradient_accumulation_steps == 0:
            if scheduler is not None:
                scheduler.step()
            optimizer.step()
            optimizer.zero_grad()

        n_step = epoch * len(train_dataloader) + step

        if config.args.local_rank == 0:
            if n_step % config.train.log_step == 0:
                logging.info("Epoch: {}/{}, Step: {}/{}, Loss: {:.4f}, Time: {:.4f}".format(
                    epoch+1, config.train.epochs, step+1, len(train_dataloader), loss.item(), time.time() - start_time))
                start_time = time.time()

            config.writer.add_scalar('train/step_loss', loss.item(), n_step)

    total_loss = total_loss / len(train_dataloader) # loss averaged on batch

    return total_loss





def eval(config, model, device, eval_dataloader, epoch):
    n_gpu = config.args.n_gpu

    # cache visual images
    batch_video_feat_list = []
    batch_text_feat_list = []
    batch_ncaption = []

    model.eval()
    with torch.no_grad():
        for bid, batch in enumerate(eval_dataloader):
            metrics = None



   

    logging.info("Video-to-Text:")

    return metrics




def run():
    ## ####################################
    # initialize experiments
    ## ####################################   
    
    args = get_args()
    config = Config.from_yaml(args.config)
    config.set_args(args)

    torch.distributed.init_process_group(backend=config.args.backend)
    config.args.world_size = torch.distributed.get_world_size()
    config.args.rank = torch.distributed.get_rank()

    config.set_experiment()
    config.set_seed()


    # Move to main.py
    if config.args.local_rank == 0:
        logging.info("======== ENVIRONMENT ========")
        logging.info("GPU number: {}".format(torch.cuda.device_count()))
        logging.info("Hostname: {}".format(socket.gethostname()))
        logging.info("Conda env: {}".format(os.environ['CONDA_DEFAULT_ENV']))
        logging.info("\n")

        if not config.args.resume:
            logging.info("======== SETUP ========")
            logging.info("MlFLow Experiment: {}".format(config.current_experiment().name))
            logging.info("MlFLow Experiment ID: {}".format(config.current_experiment().experiment_id))
            logging.info("MlFLow Tags {}".format(config.current_experiment().tags))
            logging.info("MlFLow Run Name: {}".format(config.current_run().data.tags.get('mlflow.runName')))
            logging.info("MlFLow Run ID: {}".format(config.current_run().info.run_id))
            logging.info("MlFLow Artifact Location: {}".format(config.current_run().info.artifact_uri))
            logging.info("\n")
        else:
            logging.info("======== RESTORE ========")
            logging.info("Resume MlFlow run: {}".format(config.current_run().info.run_id))
            logging.info("\n")

    ## ####################################
    # initialize model
    ## ####################################
    torch.cuda.set_device(config.args.local_rank)
    device, n_gpu = init_device(config)
    model, checkpoint = init_model(config, device)
    
    ## ####################################
    # training and evaluation
    ## ####################################
    if config.args.train or config.args.resume:
        if config.args.local_rank == 0:
            logging.info("Start training...")

            logging.info("======== MODEL ========")
            logging.info("{}".format(model))
            logging.info("Num of learnable parameters: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

        ## ####################################
        # dataloader loading
        ## ####################################
        # train_dataloader, train_sampler, train_len = get_dataloader(config, 'train')
        # val_dataloader, _, val_len = get_dataloader(config, 'val')
        train_dataset = get_dataset(config, 'train')
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=config.train.batch_size,
                                num_workers=config.data.num_workers,
                                shuffle=(train_sampler is None),
                                sampler=train_sampler,
                                drop_last=config.data.drop_last,
                                collate_fn=train_dataset.collate_fn if hasattr(train_dataset, 'collate_fn') else None)


        eval_dataset = get_dataset(config, 'val')
        eval_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=config.val.batch_size,
                                num_workers=config.data.num_workers,
                                shuffle=False,
                                drop_last=False,
                                collate_fn=train_dataset.collate_fn if hasattr(eval_dataset, 'collate_fn') else None)


        num_train_optimization_steps = (int(len(train_dataloader) + config.train.gradient_accumulation_steps - 1)
                                            / config.train.gradient_accumulation_steps) * config.train.epochs

        if config.args.local_rank == 0:
            logging.info("======== DATA ========")
            logging.info("Number of train samples: {}".format(len(train_dataset)))
            logging.info("Train batch size: {}".format(config.train.batch_size))
            logging.info("Number of total train steps: {}".format(num_train_optimization_steps * config.train.gradient_accumulation_steps))
            
            logging.info("Number of val samples: {}".format(len(eval_dataset)))
            logging.info("Val batch size: {}".format(config.val.batch_size))

            
        if config.args.local_rank == 0:
            logging.info("======== RUNNING ========")
            logging.info("Setting up model and optimizer...".format())

        optimizer, scheduler = init_optimizer(config, model, num_train_optimization_steps)

        if config.args.resume:
            last_epoch = checkpoint['last_epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            
            if scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler'])
            best_metrics = checkpoint['best_metrics']
        else:
            last_epoch = -1
            best_metrics = 0.

        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.args.local_rank], output_device=config.args.local_rank, find_unused_parameters=True)
        
        if config.args.local_rank == 0:
            logging.info("Start training...")

        for epoch in range(last_epoch + 1, config.train.epochs):
            # shuffle for different sampler
            train_sampler.set_epoch(epoch)

            # Train epoch
            loss = train_epoch(config, model, optimizer, scheduler, train_dataloader, device, n_gpu, epoch)

            if config.args.local_rank == 0:
                logging.info("Epoch {}/{} - Total loss {}".format(epoch+1, config.train.epochs, loss))  
                mlflow.log_metric('train.loss', loss, step=epoch)
                config.writer.add_scalar('train/epoch_loss', loss, epoch)

            if epoch % config.val.step == 0 and config.args.local_rank == 0:
                # Evaluate epoch
                if hasattr(model, 'module'):
                    model_ = model.module.to(device)
                else:
                    model_ = model.to(device)

                metrics = eval(config, model_, device, eval_dataloader, epoch)

                logging.info(">>> Val {}/{} - {}: {}".format(epoch+1, config.train.epochs, config.val.metrics, metrics['v2t']['mean']))
                model_.log_metrics(metrics, step=epoch)


                # Save best checkpoint
                if not((metrics['v2t']['mean'] > best_metrics) ^ config.val.higher_better):
                    best_metrics = metrics['v2t']['mean']
                    if config.args.local_rank == 0:
                        checkpoint = {
                            'last_epoch': epoch,
                            'model_state_dict': model_.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict() if scheduler is not None else None,
                            'best_metrics': best_metrics
                        }
                        model_.save_model(name='best', checkpoint=checkpoint, metrics=metrics)
                        # logging.info("Saved model to {}".format(config.artifacts_folder() + '/' + 'best'))
                        
            # Save latest checkpoint
            if config.args.local_rank == 0:
                checkpoint = {
                    'last_epoch': epoch,
                    'model_state_dict': model_.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict() if scheduler is not None else None,
                    'best_metrics': best_metrics
                }

                model_.save_model(name='latest', checkpoint=checkpoint, metrics=metrics)
                # logging.info("Saved model to {}".format(config.artifacts_folder() + '/' + 'latest'))

                if epoch % config.train.save_freq == 0:
                    model_.save_model(name='epoch_{}'.format(epoch+1), checkpoint=checkpoint, metrics=metrics)
                    # logging.info("Saved model to {}".format(config.artifacts_folder() + '/' + 'epoch_{}'.format(epoch+1)))


        if config.args.local_rank == 0:
            logging.info("Finished training.")
        config.close_experiment()



    ## ####################################
    # testing
    ## ####################################
    if config.args.test:
        if config.args.local_rank == 0:
            logging.info("Start testing...")

        # # load best model
        # model = init_model(config, device, n_gpu)
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.args.local_rank], output_device=config.args.local_rank, find_unused_parameters=True)

        # # evaluate model
        # eval(model, config, val_dataloader)

        if config.args.local_rank == 0:
            logging.info("Finished testing.")
        config.close_experiment()



if __name__ == "__main__":
    run()