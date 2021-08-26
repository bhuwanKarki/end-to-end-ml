import wget
import os
import zipfile
from pathlib import Path
import yaml
import hydra
from omegaconf import OmegaConf, DictConfig


@hydra.main(config_path='../config',config_name="config")
def func_app(cfg:DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print(cfg.data.download.url)
    url=cfg.data.download.url
    zipfiles=cfg.data.download.zip_file
    wget.download(url,zipfiles)
    with zipfile.ZipFile(Path(hydra.utils.get_original_cwd())/Path(cfg.data.download.dir)/Path(zipfiles),"r") as zip_ref:
       zip_ref.extractall(Path(hydra.utils.get_original_cwd())/Path(cfg.data.download.dir)/"raw")
    os.remove(Path(hydra.utils.get_original_cwd())/Path(cfg.data.download.dir)/Path(zipfiles))  
    
if __name__ == '__main__':
    func_app()
  

