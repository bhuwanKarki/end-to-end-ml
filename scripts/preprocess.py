# importing the libraries
from shutil import copy2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

# root path for the directory
@hydra.main(config_path='../config',config_name="config")
def preprocess(cfg:DictConfig):
    root_dir=Path(hydra.utils.to_absolute_path("data"))
    img_path=Path(hydra.utils.to_absolute_path("data/raw/cats_and_dogs_filtered"))
    # creatign a directory containing the images
    print(root_dir)
    print(img_path)
    dataset=(root_dir/"dataset").mkdir(exist_ok=True)
    # creating a dataframe to store the data
    
    df=pd.DataFrame([
        {"image_name":img.name,
        "label":img.parent.name,
        "path":img,
        "raw_split":img.parents[1].name
        }
        for img in img_path.glob("**/*.jpg")
    ])
    # creating a new column for splitting the dataset
    df=df.assign(
    split=lambda x:df.raw_split.map(lambda split:(
        "train" if split=="train" else
        np.random.choice(["val","test"],p=[cfg.split.val,cfg.split.test])
    ))          
                                    )
    print(df)
    for s in ["train","test","val"]:
        for label in set(df.label):
            print(label)
            print("------------")
            (root_dir/"dataset"/s/label).mkdir(exist_ok=True,parents=True)
        
# copying the images to the corresponding folder      
    tqdm.pandas(desc="copying images to the newly created datset folder")
    df.progress_apply(lambda x:copy2(x.path,root_dir/"dataset"/x.split/x.label/x.image_name),axis=1)
   
if __name__ == "__main__":
    preprocess()