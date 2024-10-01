import os
import sys
import hydra
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf):
    pl.seed_everything(conf.seed, workers=True)
    output_dir = HydraConfig.get().runtime.output_dir

    logger = TensorBoardLogger(save_dir=output_dir, name="logs")

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, "checkpoints"),
            filename="{epoch}",
            monitor=f"{conf.monitor}",
            mode="min",
            save_top_k=conf.save_top_k,
            save_last=True,
        ),
        RichModelSummary(max_depth=1),
        RichProgressBar(),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        logger=logger,
        gradient_clip_val=conf.gradient_clip_val,
        gradient_clip_algorithm=conf.gradient_clip_algorithm,
        max_epochs=conf.epochs,
        accelerator="gpu",
        devices=conf.gpus,
        strategy="ddp_find_unused_parameters_false",
        callbacks=callbacks,
        limit_train_batches=conf.limit_train_batches,
        limit_val_batches=conf.limit_val_batches,
        sync_batchnorm=conf.sync_bn,
    )

    model = instantiate(conf.model.target)
    os.system('cp -a %s %s' % ('conf', output_dir))
    os.system('cp -a %s %s' % ('src', output_dir))
    with open(f'{output_dir}/model.txt', 'w') as f:
        original_stdout = sys.stdout  
        sys.stdout = f  
        print(model)  
        sys.stdout = original_stdout  
    datamodule = instantiate(conf.datamodule.target)
    trainer.fit(model, datamodule, ckpt_path=conf.checkpoint)
    trainer.validate(model, datamodule.val_dataloader())


if __name__ == "__main__":
    main()
