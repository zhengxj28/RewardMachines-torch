import wandb

wandb.init(
    # set the wandb project where this run will be logged
    project="QRM test",

    # track hyperparameters and run metadata
    config={
        "env": "OfficeWorld",
        "map": 0,
    }
)

for i in range(100):
    wandb.log({"square": i**2})

wandb.finish()