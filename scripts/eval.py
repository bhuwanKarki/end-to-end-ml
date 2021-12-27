import json
import pandas as pd
import tensorflow as tf
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve,roc_curve,average_precision_score,roc_auc_score
import math

@hydra.main(config_path='../config',config_name="config")
def func_app(cfg:DictConfig):
    print(OmegaConf.to_yaml(cfg))   
    test_data=tf.keras.preprocessing.image_dataset_from_directory(hydra.utils.to_absolute_path(Path(cfg.data_.test)),
                                                              shuffle=False,
                                                              batch_size=cfg.trainer.batch_size,
                                                              image_size=(cfg.image.size,cfg.image.size))
    print(hydra.utils.to_absolute_path(Path(cfg.model.path))/Path("model"))
    model=tf.keras.models.load_model(hydra.utils.to_absolute_path(Path(cfg.model.path))/Path("model"))
    prediction=(model.predict(test_data).flatten())
    df=pd.DataFrame({"image_path":test_data.file_paths,
              "prediction":prediction})
    df=df.assign(true_label=df.image_path.str.split("/").str.get(-2),
          predicted_label=lambda df:(df.prediction>0.5).astype(int).map({0:"cats",1:"dogs"}))
    print(df)
    df.to_csv(hydra.utils.to_absolute_path(Path(cfg.test.dir))/Path("predictions.csv"),index=False)
    metrics={"test_set_length":len(df),
             "accuracy":accuracy_score(df.true_label,df.predicted_label)}
    with open(hydra.utils.to_absolute_path(Path(cfg.test.dir))/Path("metrics.json"),"w") as f:
        json.dump(metrics,f)
        
    precision, recall, prc_thresholds = precision_recall_curve(df.true_label.map({"cats":0,"dogs":1}),prediction)
    fpr, tpr, roc_thresholds = roc_curve(df.true_label.map({"cats":0,"dogs":1}),prediction)

    avg_prec = average_precision_score(df.true_label.map({"cats":0,"dogs":1}),prediction)
    roc_auc = roc_auc_score(df.true_label.map({"cats":0,"dogs":1}),prediction)
    
    with open(hydra.utils.to_absolute_path(Path(cfg.test.dir))/Path("score.json"),"w") as f:
        json.dump({"avg_prec": avg_prec, "roc_auc": roc_auc}, f, indent=4)
        
    nth_point = math.ceil(len(prc_thresholds) / 1000)
    prc_points = list(zip(precision, recall, prc_thresholds))[::nth_point]
   
    with open(hydra.utils.to_absolute_path(Path(cfg.test.dir))/Path("prc.json"),"w") as f:
        json.dump(
            {
                "prc": [
                 {"precision": p, "recall": r, "threshold": float(t)}
                for p, r, t in prc_points
            ]
        },
        f,
        indent=4,
    )   
    with open(hydra.utils.to_absolute_path(Path(cfg.test.dir))/Path("roc.json"),"w") as f:
        json.dump(
         {
            "roc": [
                {"fpr": fp, "tpr": tp, "threshold": float(t)}
                for fp, tp, t in zip(fpr, tpr, roc_thresholds)
            ]
            },
        f,
        indent=4,
    )
    
if __name__ == '__main__':
    func_app()