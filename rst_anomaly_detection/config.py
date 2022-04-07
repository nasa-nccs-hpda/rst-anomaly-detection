# -*- coding: utf-8 -*-

import os
import sys
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from omegaconf import OmegaConf, MISSING


@dataclass
class Config:

    data_regex: str

    num_true_pixels: int = 1
    data_dir: str = 'output'

    tile_size: int = 128
    num_train_tiles: int = 100
    num_val_tiles: int = 25
    num_test_tiles: int = 25
    batch_size: int = 64
    pretrained: bool = False

    init_epoch: int = 0
    max_epochs: int = 10

    # model hyperparameters
    learning_rate: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 0.0005
    batch_size: int = 8

    # model training parameters
    max_epochs: int = 100
    init_epoch: int = 0
    pretrained: bool = False
        
    coco_description: str = 'maskrcnn'

    coco_info: Dict[str, str] = field(
        default_factory=lambda: {
            "description": "maskrcnn model",
            "url": "https://www.nccs.nasa.gov",
            "version": "0.1.0",
            "year": "2021",
            "contributor": "TBD"
        }
    )

    classes: List[str] = field(
        default_factory=lambda: ['CosmicRays'])

    coco_licenses: Dict[str, str] = field(
        default_factory=lambda: {
            "id": "1",
            "name": "TBD",
            "url": "https://www.nccs.nasa.gov",
        }
    )

    coco_categories: Dict[str, str] = field(
        default_factory=lambda: {
            "id": "1",
            "name": "object1",
            "supercategory": "object1",
        }
    )

    coco_category_info: Dict[str, str] = field(
        default_factory=lambda: {
            "id": "1",
            "is_crowd": False,
        }
    )
        
    model_filename: Optional[str] = '.pkl'

# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    schema = OmegaConf.structured(Config)
    conf = OmegaConf.load("../config/config_clouds/vietnam_clouds.yaml")
    try:
        conf = OmegaConf.merge(schema, conf)
    except BaseException as err:
        sys.exit(f"ERROR: {err}")

    sys.exit(conf)
