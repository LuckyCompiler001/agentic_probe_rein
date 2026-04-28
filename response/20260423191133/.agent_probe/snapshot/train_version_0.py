import json
import os.path as osp
from datetime import datetime

import numpy as np
import torch
torch.set_float32_matmul_precision('high')
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (EarlyStopping, ModelCheckpoint,
                                         RichProgressBar)

from datamodules.celebadatamodule import CelebADataModule
from hparams import Parameters
from lightningmodules.classification import Classification
from utils.callbacks import MetricsCallback
from utils.constant import ATTRIBUTES
from utils.utils_functions import create_dir


def get_trainer_device_args(gpu: int):
    if torch.cuda.is_available():
        return {"accelerator": "gpu", "devices": gpu if gpu > 0 else 1}
    return {"accelerator": "cpu", "devices": 1}


def main():
    config = Parameters.parse()
    trainer_device_args = get_trainer_device_args(config.hparams.gpu)

    dataset_module = CelebADataModule(config.data_param)

    if config.hparams.train:

        model = Classification(config.train_param, ATTRIBUTES)

        callbacks = [EarlyStopping(**config.callback_param.early_stopping_params),
                     MetricsCallback(config.train_param.n_classes),
                     ModelCheckpoint(**config.callback_param.model_checkpoint_params),
                     RichProgressBar()
        ]

        trainer_logger = False
        if config.hparams.use_wandb:
            from pytorch_lightning.loggers import WandbLogger
            from utils.callbacks import WandbImageCallback

            wdb_config = {}
            for k, v in vars(config).items():
                for key, value in vars(v).items():
                    wdb_config[f"{k}-{key}"] = value

            trainer_logger = WandbLogger(
                config=wdb_config,
                project=config.hparams.wandb_project,
                entity=config.hparams.wandb_entity,
                allow_val_change=True,
                save_dir=config.hparams.save_dir,
            )
            callbacks.insert(2, WandbImageCallback(config.callback_param.nb_image))

        trainer = Trainer(logger=trainer_logger,
                          **trainer_device_args,
                          callbacks=callbacks,
                          log_every_n_steps=1,
                          enable_checkpointing=True,
                          fast_dev_run=config.hparams.fast_dev_run,
                          max_epochs=config.hparams.max_epochs,
                          limit_train_batches=config.hparams.limit_train_batches,
                          val_check_interval=config.hparams.val_check_interval,
                          )

        trainer.fit(model, dataset_module)

    if config.hparams.predict:
        output_dict = {"filenames":[], "logits":[], "converted_preds":[], "preds_with_conf":[]}
        model = Classification.load_from_checkpoint(
            config.inference_param.ckpt_path,
            config=config.inference_param,
            attr_dict=dataset_module.attr_dict,
            map_location="cpu",
        )
        trainer = Trainer(logger=False,
                          enable_checkpointing=False,
                          **trainer_device_args)
        predictions = trainer.predict(model, dataloaders=dataset_module.predict_dataloader(),
                                      return_predictions=True)

        output_root = config.inference_param.output_root
        create_dir(output_root)
        name_output = f"output_dict-{datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}.json"
        output_full_path = osp.join(output_root, name_output)

        for pred_batch in predictions:
            img_names, preds, converted_preds, converted_logits = pred_batch[0], pred_batch[1], pred_batch[2], pred_batch[3]
            for i, img_name in enumerate(img_names):
                output_dict['filenames'].append(img_name)
                output_dict['logits'].append(converted_logits.tolist()[i])
                output_dict['converted_preds'].append(converted_preds[i])
                preds_with_conf = {ATTRIBUTES[idx]:round(converted_logits.tolist()[i][idx], 3) for idx in np.where(preds[i]==1.0)[0]}
                output_dict['preds_with_conf'].append(preds_with_conf)
        json.dump(output_dict, open(output_full_path, 'w'))

if __name__ == "__main__":
    main()
