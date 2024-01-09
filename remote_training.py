import kaggle
from pathlib import Path, PurePosixPath
import json
try:
    from __kaggle_login import kaggle_users
except ImportError:
    raise ImportError("Please create a __kaggle_login.py file with a kaggle_users" +
                      "dict containing your Kaggle credentials.\n" +
                      "See https://github.com/balthazarneveu/mva_pepites for more details.")
import argparse
import sys
import subprocess
from shared import ROOT_DIR, OUTPUT_FOLDER_NAME
from train import get_parser as get_train_parser
from typing import Optional
from configuration import KAGGLE_DATASET_LIST, NB_ID, GIT_USER, GIT_REPO


def get_git_branch_name():
    try:
        branch_name = subprocess.check_output(["git", "branch", "--show-current"]).strip().decode()
        return branch_name
    except subprocess.CalledProcessError:
        return "Error: Could not determine the Git branch name."


def prepare_notebook(
    output_nb_path: Path,
    exp: int,
    branch: str,
    git_user: str = None,
    git_repo: str = None,
    template_nb_path: Path = Path(__file__).parent/"remote_training_template.ipynb",
    wandb_flag: bool = False,
    output_dir: Path = OUTPUT_FOLDER_NAME,
    dataset_files: Optional[list] = None,
):
    assert git_user is not None, "Please provide a git username for the repo"
    assert git_repo is not None, "Please provide a git repo name for the repo"
    expressions = [
        ("exp", f"{exp}"),
        ("branch", f"\'{branch}\'"),
        ("git_user", f"\'{git_user}\'"),
        ("git_repo", f"\'{git_repo}\'"),
        ("wandb_flag", "True" if wandb_flag else "False"),
        ("output_dir", "None" if output_dir is None else f"\'{output_dir}\'"),
        ("dataset_files", "None" if dataset_files is None else f"{dataset_files}")
    ]
    with open(template_nb_path) as f:
        template_nb = f.readlines()
        for line_idx, li in enumerate(template_nb):
            for expr, expr_replace in expressions:
                if f"!!!{expr}!!!" in li:
                    template_nb[line_idx] = template_nb[line_idx].replace(f"!!!{expr}!!!", expr_replace)
        template_nb = "".join(template_nb)
    with open(output_nb_path, "w") as w:
        w.write(template_nb)


def main(argv):
    parser = argparse.ArgumentParser(description="Train a model on Kaggle using a script\n" +
                                     "Help: https://github.com/balthazarneveu/mva_pepites")
    parser.add_argument("-n", "--nb_id", type=str, help="Notebook name in kaggle", default=NB_ID)
    parser.add_argument("-u", "--user", type=str, help="Kaggle user", choices=list(kaggle_users.keys()))
    parser.add_argument("--branch", type=str, help="Git branch name", default=get_git_branch_name())
    parser.add_argument("-p", "--push", action="store_true", help="Push")
    parser.add_argument("-d", "--download", action="store_true", help="Download results")
    get_train_parser(parser)
    args = parser.parse_args(argv)
    nb_id = args.nb_id
    exp_str = "_".join(f"{exp:04d}" for exp in args.exp)
    kaggle_user = kaggle_users[args.user]
    uname_kaggle = kaggle_user["username"]
    kaggle.api._load_config(kaggle_user)
    if args.download:
        tmp_dir = ROOT_DIR/f"__tmp_{exp_str}"
        tmp_dir.mkdir(exist_ok=True, parents=True)
        kaggle.api.kernels_output_cli(f"{kaggle_user['username']}/{nb_id}", path=str(tmp_dir))
        subprocess.run(["tar", "-xzf", tmp_dir/"output.tgz", OUTPUT_FOLDER_NAME])
        # @FIXME: windows probably does not have tar command
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return
    kernel_root = ROOT_DIR/f"__nb_{uname_kaggle}"
    kernel_root.mkdir(exist_ok=True, parents=True)

    kernel_path = kernel_root/exp_str
    kernel_path.mkdir(exist_ok=True, parents=True)
    branch = args.branch
    config = {
        "id": str(PurePosixPath(f"{kaggle_user['username']}")/nb_id),
        "title": nb_id.lower(),
        "code_file": f"{nb_id}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true" if not args.cpu else "false",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": KAGGLE_DATASET_LIST,
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": []
    }
    prepare_notebook((kernel_path/nb_id).with_suffix(".ipynb"), args.exp, branch,
                     git_user=GIT_USER, git_repo=GIT_REPO, wandb_flag=not args.no_wandb)
    assert (kernel_path/nb_id).with_suffix(".ipynb").exists()
    with open(kernel_path/"kernel-metadata.json", "w") as f:
        json.dump(config, f, indent=4)

    if args.push:
        kaggle.api.kernels_push_cli(str(kernel_path))


if __name__ == '__main__':
    main(sys.argv[1:])
