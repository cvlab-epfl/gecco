import argparse
import os
import json
from datetime import datetime
import git

import gecco_jax


def execute(config_path):
    path, _py_file_name = os.path.split(config_path)
    path = os.path.abspath(path)
    config = gecco_jax.load_config(config_path)

    # fail fast
    required = (
        "make_train_loader",
        "make_val_loader",
        "make_model",
        "train",
    )
    for attr in required:
        assert hasattr(config, attr), attr

    train_dataloader = config.make_train_loader()
    val_dataloader = config.make_val_loader()

    try:
        commit_hash = git.Repo(os.getcwd()).git.rev_parse("HEAD")
    except git.exc.GitError:
        commit_hash = "unknown"

    date = datetime.utcnow().isoformat()
    with open(os.path.join(path, "metadata.json"), "w") as metadata_file:
        json.dump(
            {
                "commit_hash": commit_hash,
                "date": date,
            },
            metadata_file,
        )

    config.train(
        model=config.make_model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        save_path=path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    args = parser.parse_args()

    execute(args.config_path)


if __name__ == "__main__":
    main()
