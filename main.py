from lightning.pytorch.cli import LightningCLI
import lightning
from importlib.metadata import version


def main():
    lightning.seed_everything(666)  # for reproducibility
    print(f"lightning_toolbox version {version('lightning-toolbox')} is installed.")
    LightningCLI(
        lightning.LightningModule,
        lightning.LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        auto_configure_optimizers=False,
    )   

    


if __name__ == "__main__":
    main()
