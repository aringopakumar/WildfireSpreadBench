from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only

from evaluation.unified_eval import evaluate_lightning_module


class UnifiedEvalCallback(Callback):
    """Run unified benchmark eval on the test set every N epochs during training."""

    def __init__(
        self,
        eval_every_n_epochs: int = 10,
        max_batches: int | None = None,
        threshold: float = 0.5,
        wandb_log: bool = True,
        verbose: bool = True,
    ):
        self.eval_every_n_epochs = eval_every_n_epochs
        self.max_batches = max_batches
        self.threshold = threshold
        self.wandb_log = wandb_log
        self.verbose = verbose

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        epoch = trainer.current_epoch
        if (epoch + 1) % self.eval_every_n_epochs != 0:
            return

        datamodule = trainer.datamodule
        if datamodule is None:
            return

        datamodule.setup("test")
        test_loader = datamodule.test_dataloader()
        model_name = pl_module.__class__.__name__

        if self.verbose:
            print(
                f"\n[UnifiedEval] Running benchmark eval at epoch {epoch + 1} "
                f"(every {self.eval_every_n_epochs} epochs)..."
            )

        evaluate_lightning_module(
            pl_module=pl_module,
            eval_loader=test_loader,
            device=pl_module.device,
            model_name=model_name,
            epoch=epoch,
            threshold=self.threshold,
            wandb_log=self.wandb_log,
            verbose=self.verbose,
            max_batches=self.max_batches,
        )
