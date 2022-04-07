import os
import glob
import json
import random
import logging
import datetime
import omegaconf
import numpy as np
from tqdm.autonotebook import tqdm
from .pycococreatortools import create_image_info, create_annotation_info

def gen_data_pairs(
    filename: str, data: np.ndarray, label: np.ndarray,
    config: omegaconf.dictconfig.DictConfig,
    set: str = 'train', label_value: int = 1):
    """
    Save png images on disk
    Args:
        filename (str): list of input bands
        data (np array): array with imagery values
        label (np array): array with labels values
        config (CfgNode obj): configuration object
        set (str): dataset to prepare (e.g train, test, val)
    """    
    # set dimensions of the input image array, and get desired tile size
    x_dim, y_dim, z_dim = data.shape  # dimensions of imagery

    # set output directory
    save_dir = os.path.join(config.data_dir, set)
    os.makedirs(save_dir, exist_ok=True)
    logging.info(f'Saving file under: {save_dir}')

    # iterate over the number of tiles
    for i in range(config[f'num_{set}_tiles']):

        # Generate random integers from image
        yc = random.randint(0, y_dim - 2 * config.tile_size)
        xc = random.randint(0, x_dim - 2 * config.tile_size)
        counter_attempts = 0

        # verify data is not on nodata region
        while np.count_nonzero(
                    label[yc:(yc + config.tile_size), xc:(xc + config.tile_size)] == label_value
                ) < config.num_true_pixels:
            yc = random.randint(0, y_dim - 2 * config.tile_size)
            xc = random.randint(0, x_dim - 2 * config.tile_size)
            
            # we are going to try 1000 times, if no luck, we move on to the next
            if counter_attempts < 1000:
                counter_attempts += 1
            else:
                return

        data_tile = data[yc:(yc + config.tile_size), xc:(xc + config.tile_size), :]
        label_tile = label[yc:(yc + config.tile_size), xc:(xc + config.tile_size)]

        # save numpy files
        np.save(os.path.join(save_dir, f'{filename}_img_{i+1}.npy'), data_tile)
        np.save(os.path.join(save_dir, f'{filename}_lbl_{i+1}.npy'), label_tile)
    return


def gen_coco_dataset(
        config: omegaconf.dictconfig.DictConfig, set: str = 'train',
        data_regex: str = '*_img_*.npy', label_regex: str = '*_lbl_*.npy'
    ):
    """
    Save JSON file with COCO formatted dataset
    src: https://patrickwasp.com/create-your-own-coco-style-dataset/
    Args:
        config (CfgNode obj): configuration object
        set (str): dataset to prepare (e.g train, test, val)
        img_reg (str): image filename regex
        label_reg (str): label filename regex
    """
    
    input_dir = os.path.join(config.data_dir, set)  # directory where images reside
    json_out = os.path.join(config.data_dir, f'{config.coco_description}_{set}.json')


    # src: https://patrickwasp.com/create-your-own-coco-style-dataset/
    # Define several sections of the COCO Dataset Format

    # General Information
    #INFO = dict(cfg.DATASETS.COCO_METADATA.INFO)
    config.coco_info["date_created"] = datetime.datetime.utcnow().isoformat(' ')

    # Licenses and categories
    #LICENSES = [dict(cfg.DATASETS.COCO_METADATA.LICENSES)]
    #CATEGORIES = [dict(cfg.DATASETS.COCO_METADATA.CATEGORIES)]
    #CATEGORY_INFO = dict(cfg.DATASETS.COCO_METADATA.CATEGORY_INFO)

    # Retrieve filenames from local storage
    train_names = sorted(
        glob.glob(os.path.join(input_dir, data_regex)))
    mask_names = sorted(
        glob.glob(os.path.join(input_dir, label_regex)))

    # place holders to store dataset metadata
    images = list()
    annotations = list()
    pastId = 0

    # go through each image
    annot_counter = 0
    for curImgName, curMaskName in zip(train_names, mask_names):

        curImgFile = curImgName
        curMaskFile = curMaskName

        # taking care of the images
        # curImg = Image.open(curImgFile)
        curImg = np.load(curImgFile)
        curImgId = pastId + 1  # make sure it's properly unique
        pastId = curImgId
        curImgInfo = create_image_info(
            curImgId, os.path.basename(curImgFile), curImg.shape
        )
        images.append(curImgInfo)

        # taking care of the annotations
        curAnnotationId = str(curImgId)
        binaryMask = np.load(curMaskFile).astype(np.uint8)

        annotationInfo = create_annotation_info(
            curAnnotationId, curImgId, config.coco_category_info, binaryMask,
            curImg.shape[:-1], tolerance=2
        )

        if annotationInfo is not None:
            annotations.append(annotationInfo)
        else:
            annot_counter += 1

    print(
        f'Number of train and mask images: {len(train_names)}',
        f'{annot_counter} without annotations.'
    )

    coco_info = {
        "info": dict(config.coco_info),
        "licenses": [dict(config.coco_licenses)],
        "categories": [dict(config.coco_categories)],
        "images": images,
        "annotations": annotations,
    }

    with open(json_out, 'w') as f:
        f.write(json.dumps(coco_info))

    return