import os
import sys
import hydra
from hydra.core.hydra_config import HydraConfig
import pytorch_lightning as pl
from hydra.utils import instantiate, to_absolute_path
from importlib import import_module


@hydra.main(version_base=None, config_path="./conf/", config_name="config")
def main(conf):
    pl.seed_everything(conf.seed)
    output_dir = HydraConfig.get().runtime.output_dir
    checkpoint = to_absolute_path("ckpts/DeMo.ckpt")  # TODO change to your checkpoint
    assert os.path.exists(checkpoint), f"Checkpoint {checkpoint} does not exist"

    trainer = pl.Trainer(
        logger=False,
        accelerator="gpu",
        devices=conf.gpus,
        max_epochs=1,
        limit_val_batches=conf.limit_val_batches,
        limit_test_batches=conf.limit_test_batches,
    )

    datamodule: pl.LightningDataModule = instantiate(conf.datamodule.target, test=conf.test)
    model = instantiate(conf.model.target)
    os.system('cp -a %s %s' % ('conf', output_dir))
    os.system('cp -a %s %s' % ('src', output_dir))
    if trainer.local_rank == 0:
        with open(os.path.join(HydraConfig.get().runtime.output_dir, "model.log"), "x") as f:
            print(model.net, file=f)
    if not conf.test:
        trainer.validate(model, datamodule, ckpt_path=checkpoint)
    else:
        trainer.test(model, datamodule, ckpt_path=checkpoint)


if __name__ == "__main__":
    main()
