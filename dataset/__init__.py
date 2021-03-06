from .transform import RandomErasing
from .collate_batch import train_collate_fn
from .collate_batch import val_collate_fn
from .triplet_sampler import RandomIdentitySampler
from .data import ImageDataset
import torchvision.transforms as T
from torch.utils.data.dataloader import DataLoader
from .vehicleid import VehicleID
from .veri import VeRi



__imgreid_factory = {
    'veri': VeRi,
    'vehicleID': VehicleID,
}

def get_trm(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            normalize_transform,
            RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    else:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            normalize_transform
        ])
    return transform



def make_dataloader(cfg, num_gpus=1):
    train_trm = get_trm(cfg, is_train=True)
    val_trm = get_trm(cfg, is_train=False)

    num_workers = cfg.DATALOADER.NUM_WORKERS * num_gpus
    dataset = init_dataset(cfg.DATASETS.NAMES, root = cfg.DATASETS.DATA_PATH,
                                               verbose = True,
                                               test_size = cfg.DATASETS.TEST_SIZE)

    num_classes = dataset.num_train_pids
    train_set = ImageDataset(dataset.train, cfg, train_trm)
    if cfg.DATALOADER.SAMPLER == 'softmax':
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH * num_gpus, shuffle=True,
            num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH * num_gpus,
            sampler=RandomIdentitySampler(dataset.train,
                cfg.SOLVER.IMS_PER_BATCH * num_gpus,
                cfg.DATALOADER.NUM_INSTANCE * num_gpus),
            num_workers=num_workers, collate_fn=train_collate_fn
        )

    val_set = ImageDataset(dataset.query + dataset.gallery, cfg, val_trm)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH * num_gpus, shuffle=False,
        num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, val_loader, len(dataset.query), num_classes


def init_dataset(name, **kwargs):
    """
        Use path in cfg to init a dataset
        the dataset should be the following format
        - Each Image should be named in 
                (pid)_c(camid)_(iid).jpg
            where pid is the person id, 
                  camid is camera id,
                  iid is image id(unique to every image)
        - train set and val set should be organzed as
            cfg.DATASETS.TRAIN_PATH: all the training images
            cfg.DATASETS.QUERY_PATH: all the query images
            cfg.DATASETS.GALLERY_PATH: all the gallery images
    """
    if name not in list(__imgreid_factory.keys()):
        raise KeyError('Invalid dataset, got "{}", but expected to be one of {}'.format(name, list(__imgreid_factory.keys())))
    return __imgreid_factory[name](**kwargs)