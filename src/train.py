from pytorch_lightning.utilities import rank_zero_only
import torch
from dataloader.FireSpreadDataModule import FireSpreadDataModule
from pytorch_lightning.cli import LightningCLI
from baselines import SMPModel, BaseModel, ConvLSTMLightning, LogisticRegression  # noqa
from baselines import BaseModel
from evaluation.unified_eval_callback import UnifiedEvalCallback  # noqa
import wandb
import os

from dataloader.FireSpreadDataset import FireSpreadDataset
from dataloader.utils import get_means_stds_missing_values

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
torch.set_float32_matmul_precision('high')


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("trainer.default_root_dir",
                              "trainer.logger.init_args.save_dir")
        parser.link_arguments("model.class_path",
                              "trainer.logger.init_args.name")
        parser.add_argument("--do_train", type=bool,
                            help="If True: train the model.")
        parser.add_argument("--do_predict", type=bool,
                            help="If True: compute and save predictions.")
        parser.add_argument("--do_test", type=bool,
                            help="If True: compute test metrics.")
        parser.add_argument("--do_validate", type=bool,
                            default=False, help="If True: compute val metrics.")
        parser.add_argument("--ckpt_path", type=str, default=None,
                            help="Path to checkpoint to load for resuming training, for testing and predicting.")

    def before_instantiate_classes(self):
        # The number of features is only known inside the data module, but we need that info to instantiate the model.
        # Since datamodule and model are instantiated at the same time with LightningCLI, we need to set the number of features here.
        n_features = FireSpreadDataset.get_n_features(
            self.config.data.n_leading_observations,
            self.config.data.features_to_keep,
            self.config.data.remove_duplicate_features)
        self.config.model.init_args.n_channels = n_features

        # The exact positive class weight changes with the data fold in the data module, but the weight is needed to instantiate the model.
        # Non-fire pixels are marked as missing values in the active fire feature, so we simply use that to compute the positive class weight.
        train_years, _, _ = FireSpreadDataModule.split_fires(
            self.config.data.data_fold_id)
        _, _, missing_values_rates = get_means_stds_missing_values(train_years)
        fire_rate = 1 - missing_values_rates[-1]
        pos_class_weight = float(1 / fire_rate)

        self.config.model.init_args.pos_class_weight = pos_class_weight

    def before_fit(self):
        self.wandb_setup()

    def before_test(self):
        self.wandb_setup()

    def before_validate(self):
        self.wandb_setup()

    @rank_zero_only
    def wandb_setup(self):
        """
        Save the config used by LightningCLI to disk, then save that file to wandb.
        Using wandb.config adds some strange formating that means we'd have to do some 
        processing to be able to use it again as CLI input.

        Also define min and max metrics in wandb, because otherwise it just reports the 
        last known values, which is not what we want.
        """
        if wandb.run is None:
            return

        config_file_name = os.path.join(wandb.run.dir, "cli_config.yaml")

        cfg_string = self.parser.dump(self.config, skip_none=False)
        with open(config_file_name, "w") as f:
            f.write(cfg_string)
        wandb.save(config_file_name, policy="now", base_path=wandb.run.dir)
        wandb.define_metric("train_loss_epoch", summary="min")
        wandb.define_metric("val_loss", summary="min")
        wandb.define_metric("train_f1_epoch", summary="max")
        wandb.define_metric("val_f1", summary="max")
        wandb.define_metric("eval/*", step_metric="epoch")


def main():
    # setdefault so a caller can point runs at their own project/entity without
    # editing this file.
    os.environ.setdefault("WANDB_PROJECT", "WildfireSpreadBench")
    os.environ.setdefault("WANDB_ENTITY", "ram-algoverse")

    # LightningCLI automatically creates an argparse parser with required arguments and types,
    # and instantiates the model and datamodule. For this, it's important to import the model and datamodule classes above.
    cli = MyLightningCLI(BaseModel, FireSpreadDataModule, subclass_mode_model=True, save_config_kwargs={
        "overwrite": True}, parser_kwargs={"parser_mode": "yaml"}, run=False)

    if cli.config.do_train:
        cli.trainer.fit(cli.model, cli.datamodule,
                        ckpt_path=cli.config.ckpt_path)

    # If we have trained a model, use the best checkpoint for testing and predicting.
    # Without this, the model's state at the end of the training would be used, which is not necessarily the best.
    ckpt = cli.config.ckpt_path
    if cli.config.do_train:
        ckpt = "best"

    if cli.config.do_validate:
        cli.trainer.validate(cli.model, cli.datamodule, ckpt_path=ckpt)

    if cli.config.do_test:
        cli.trainer.test(cli.model, cli.datamodule, ckpt_path=ckpt)

        # Unified eval pass — identical metric computation to generative models
        from evaluation.unified_eval import evaluate_lightning_model

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = cli.config.model.class_path.split(".")[-1]

        evaluate_lightning_model(
            model=cli.model.to(device),
            datamodule=cli.datamodule,
            device=device,
            model_name=model_name,
            wandb_log=True,
        )
        from baselines.visualize_predictions import run_visualization
        run_visualization(
            model=cli.model,
            datamodule=cli.datamodule,
            output_dir=cli.config.trainer.default_root_dir,
            model_name=model_name,
            n_samples=8,
            wandb_log=True,
        )

    if cli.config.do_predict:

        # Produce predictions, save them in a single file, including ground truth fire targets and input fire masks.
        prediction_output = cli.trainer.predict(
            cli.model, cli.datamodule, ckpt_path=ckpt)
        x_af = torch.cat(
            list(map(lambda tup: tup[0][:, -1, :, :].squeeze(), prediction_output)), axis=0)
        y = torch.cat(list(map(lambda tup: tup[1], prediction_output)), axis=0)
        y_hat = torch.cat(
            list(map(lambda tup: tup[2], prediction_output)), axis=0)
        fire_masks_combined = torch.cat(
            [x_af.unsqueeze(0), y_hat.unsqueeze(0), y.unsqueeze(0)], axis=0)

        run_id = wandb.run.id if wandb.run is not None else "local"
        predictions_file_name = os.path.join(
            cli.config.trainer.default_root_dir, f"predictions_{run_id}.pt")
        torch.save(fire_masks_combined, predictions_file_name)


if __name__ == "__main__":
    main()
