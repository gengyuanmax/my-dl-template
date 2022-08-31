import logging
from matplotlib.pyplot import get
from matplotlib.style import context
import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod
import mlflow
from einops.layers.torch import Reduce
from einops import rearrange, repeat
from math import pi
from torchsummary import summary


class ConfigModel(metaclass=ABCMeta):
    def __init__(self):
        pass

    def set_config(self, config):
        self.config = config

    @classmethod 
    def from_config(cls, config):
        return cls()

    @classmethod 
    def from_pretrained(cls, config, name='latest'):
        # model = mlflow.pytorch.load_model(config.artifacts_folder() + '/' + name)
        model = cls.from_config(config)

        # restore other
        checkpoint = torch.load(config.artifacts_folder() + '/ckpt/checkpoint_{}.pth'.format(name))

        model.load_state_dict(checkpoint['model_state_dict'], strict=True)

        if config.args.local_rank == 0:
            logging.info("Restoring model from {}".format(config.artifacts_folder() + name))

        return model, checkpoint
    
    def save_model(self, name="latest", checkpoint=None, metrics=None):
        import os
        if not os.path.isdir(self.config.artifacts_folder() + '/' + 'ckpt'):
            os.mkdir(self.config.artifacts_folder() + '/' + 'ckpt')
        
        # Save model to MLFlow artifact uri
        mlflow.log_dict(metrics, "ckpt/metrics_{}.yaml".format(name))

        assert isinstance(self, nn.Module), "model must be a torch.nn.Module"
        # mlflow.pytorch.log_model(self, name)
        path = self.config.artifacts_folder() + '/ckpt/checkpoint_{}.pth'.format(name)

        torch.save(checkpoint, path)

    def log_params(self):
        mlflow.set_tag('model', self.__class__.__name__)

        # logging model parameters to MLFlow tracking uri when creating a new run
        def _log_param(kv, p=[]):
            prefix=p.copy()
            if isinstance(kv, dict):
                for k, v in kv.items():
                    _log_param(v, prefix+[k])
            else:
                mlflow.log_param('.'.join(prefix), kv)

        model_params = self.config.get_config_dict()['model']
        _log_param(model_params, p=['model'])

        train_params = self.config.get_config_dict()['train']
        _log_param(train_params, p=['train'])

    def log_metrics(self, metrics, step=None):
        # logging model metrics to MLFlow tracking uri when asked
        def _log_metric(kv, p=[]):
            prefix=p.copy()
            if isinstance(kv, dict):
                for k, v in kv.items():
                    _log_metric(v, prefix + [k])
            else:
                mlflow.log_metric('.'.join(prefix), kv, step=step)
                self.config.writer.add_scalar('/'.join(prefix), kv, step)
        
        _log_metric(metrics, p=['val'])


class TemplateModel(ConfigModel, torch.nn.Module):
    @classmethod
    def from_config(cls, config):
        latent_dim = config.model.latent_dim
        latent_heads = config.model.latent_heads
        latent_hidden_dim = config.model.latent_dim_head
        num_latents = config.model.num_latents
        depth = config.model.depth
        num_media_embeds = config.model.num_media_embeds
        ff_mult = config.model.ff_mult
        loss_fn = config.model.loss_fn
        chunk_size = config.model.chunk_size
        similarity_head = config.model.similarity_head

        ins = cls(latent_dim,
                  latent_heads,
                  latent_hidden_dim,
                  num_latents,
                  depth,
                  num_media_embeds,
                  loss_fn,
                  chunk_size,
                  similarity_head,
                  ff_mult)

        ins.set_config(config)

        return ins

    def __init__(self, latent_dim,
                       latent_heads,
                       latent_hidden_dim,
                       num_latents,
                       depth,
                       num_media_embeds,
                       loss_fn,
                       chunk_size,
                       similarity_head,
                       ff_mult=4):
        ConfigModel.__init__(self)
        torch.nn.Module.__init__(self)

        
   
    
    def forward(self, vis_feat, text_feat, vis_mask, text_mask, mode='train'):
        pass
