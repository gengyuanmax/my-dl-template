import torch
from torch.utils.data import DataLoader
import TemplateDataset


dataset_dict = {
    'ms-activitynet-feat': MsActivitynetFeatDataset,
}

def get_dataloader(config, split='train'):
    dataset = dataset_dict[config.data.name].from_config(config, split)

    data_sampler = torch.utils.data.distributed.DistributedSampler(dataset) 

    dataloader = DataLoader(dataset=dataset,
                            batch_size=config.train.batch_size if split == 'train' else config.val.batch_size,
                            num_workers=config.data.num_workers,
                            shuffle=True if split == 'train' else False,
                            sampler=data_sampler if split=='train' else None,
                            drop_last=config.data.drop_last,
                            collate_fn=dataset.collate_fn if hasattr(dataset, 'collate_fn') else None)
    
    
    return dataloader, data_sampler, len(dataset)

def get_dataset(config, split='train'):
    dataset = dataset_dict[config.data.name].from_config(config, split)

    return dataset

