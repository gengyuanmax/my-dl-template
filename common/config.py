from turtle import pen
import yaml
import re
import logging
import torch
import random
import os
import mlflow
import time
import datetime
import tensorboardX

_LEVEL = {
    'critical': logging.CRITICAL,
    'error': logging.ERROR,
    'warning': logging.WARNING,
    'info': logging.INFO,
    'debug': logging.DEBUG,
    'notset': logging.NOTSET,
}

class Config:
    @classmethod
    def from_yaml(cls, yaml_path):
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
            assert isinstance(config, dict)
            config = cls(config)

        return config

    def __init__(self, config):
        self._config_dict = config
        self._logger = None

    def __getattr__(self, k):
        v = self._config_dict[k]
        if isinstance(v, dict):
            return Config(v)
        return v

    def set_args(self, args):
        self.args = args

    def set_experiment(self):
        self._experiment, self._run = self._set_mlflow()
        self._logger = self._set_logger()
        self.writer = tensorboardX.SummaryWriter(self.artifacts_folder() + '/tensorboard', flush_secs=60) 


    def current_experiment(self):
        return self._experiment
    
    def current_run(self):
        return self._run

    def close_experiment(self):
        mlflow.end_run()

        if self.args.local_rank == 0:
            mlflow.set_tag('progress', 'CLOSED')

    def set_seed(self):
        seed = self.seed.default_seed
        if seed:
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            if self.args.local_rank == 0:
                mlflow.log_param('seed', seed)
                logging.info('Set random seed to {}'.format(seed))
        else:
            if self.args.local_rank == 0:
                mlflow.log_param('seed', 'None')
                logging.info('No random seed set')

    def _set_mlflow(self):
        # TODO: if exist, just create on master process (rank 0)
        mlflow.set_experiment(self.experiment)
        experiment = mlflow.get_experiment_by_name(self.experiment)

        if self.args.resume:
            run = mlflow.start_run(run_id=self.args.run_id)
        else:
            if self.args.local_rank == 0:
                run = mlflow.start_run(run_name=os.environ['START_TIME'])
            else:
                wait_ = 0
                while True:
                    runs = mlflow.search_runs(filter_string="tags.mlflow.runName LIKE '{}'".format(os.environ['START_TIME']))
                    if len(runs) > 0:
                        run_id = runs['run_id'].values[0]

                        try:
                            run = mlflow.start_run(run_id=run_id)
                        except AttributeError:
                            time.sleep(0.5)
                        else:
                            break
                    else:
                        time.sleep(0.5)
                        wait_ += 1

                    if wait_ > 20:
                        raise TimeoutError('Cannot find run named {}'.format(os.environ['START_TIME']))


        if self.args.local_rank == 0:   
            mlflow.log_dict(self._config_dict, 'config.yaml')

        if self.args.local_rank == 0:
            mlflow.set_tag('progress', 'OPENNING')

        return experiment, run
        
    def _set_logger(self):
        logger = logging.getLogger('logger')
        logger.setLevel(_LEVEL[self.level])
        logging.basicConfig(format='[%(asctime)s:%(levelname)s] %(message)s',
                    level=_LEVEL[self.level])

        filename = mlflow.get_artifact_uri().split(':')[1][2:] + '/logging.txt'
        handler = logging.FileHandler(filename)
        handler.setLevel(_LEVEL[self.level])
        handler.setFormatter(logging.Formatter('[%(asctime)s : %(levelname)s] %(message)s'))
        logging.getLogger().addHandler(handler)
        
        return logger

    def artifacts_folder(self):
        return  mlflow.get_artifact_uri().split(':')[1][2:]

    def get_config_dict(self):
        return self._config_dict

if __name__=='__main__':
    cfg = Config.from_yaml('/Users/GengyuanMax/workspace/MS-CLIP/config.yaml')
    print(cfg.get_config_dict())
    # cfg.set_experiment()
    # cfg.set_seed()

        