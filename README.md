# :computer::fire::gift: Kaggle compute power from your terminal
:free: Access to free kaggle compute power from your command line.
- Push notebooks through Kaggle API.
- Version everything under github.
- Save training metrics (+option to log to weights and biases)

![](/illustrations/overview.png)


Here's what your runs could look like on [Weights and biases](https://wandb.ai/)...

![](/illustrations/wandb.png)

:sos: How to use? 
- `pip install kaggle`
- Create a kaggle account, get a kaggle API token.
- Copy paste this repo as a template and start customizing, check that you can train locally...
- Fill the `__kaggle_login.py` :warning: do not push it to git
- Push your code to github
- Use command line to push your notebook to Kaggle.

:bulb: What to customize?

- Customize [configuration.py](/configuration.py)
```python
NB_ID = "training-notebook" # This will be the name which appears on Kaggle.
GIT_USER = "balthazarneveu" # Your git user name
GIT_REPO = "mva_pepites" # Your current git repo
KAGGLE_DATASET_LIST = [] # Keep free unless you need to acess kaggle datasets. You'll need to modify the remote_training_template.ipynb.
```
> Note: *you can add kaggle datasets (if you need to put 4Gb of data, it's possible to host it with [Kaggle datasets](https://www.kaggle.com/datasets)). Fill the `KAGGLE_DATASET_LIST`. You'll also have to customize the [remote_training_template.ipynb](/remote_training_template.ipynb) to unzip and acess datasets.*


> You can run several experiments in a row using `-e 1 2 3`. If initialization is long (decompress datasets, preprocess etc...), it may be worth running several experiments in a row.



-------

## :chart_with_downwards_trend: Training

:id: Keep track of experiments by an integer id. 

Each experiment is defined by:
- :scroll: Dataloader configuration (data, augmentations)
- :gear: Model (architecture, sizes)
- :chart: Optimizer configuration (hyperparameters)

:test_tube: Code to define [new experiments](/experiments.py)

### Remote training
- Retrive your kaggle token from the [website](https://www.kaggle.com/docs/api).
- *Several accounts mean simply more GPU power*. As of 2024, Kaggle allows 30 hours per week, limited to 12hours of execution per notebook.
- :unlock: Create a [__kaggle_login.py](__kaggle_login.py) file locally.
```python
kaggle_users = {
    "user1": {
        "username": "user1_kaggle_name",
        "key": "user1_kaggle_key"
    },
    "user2": {
        "username": "user2_kaggle_name",
        "key": "user2_kaggle_key"
    },
}
```
:warning: Do not push these secret tokens to github and leak it publicly :facepalm:

Run `python remote_training.py -u user1 -e X -nowb`
This will create a dedicated folder for training a specific experiment with a dedicated notebook.

- use **`-p`** (`--push`) will upload/push the notebook and run it.
- use **`-d`** (`--download`) to download the training results and save it to disk. This is not automatic


#### :green_circle: First time setup
> - `python remote_training.py -u user1 -e 0 --cpu --push -nowb`
> - use **`--cpu`** to setup at the begining (avoid using GPU when you set up :warning: )
> - Go to kaggle and check your notifications to access your notebook.
> - Edit notebook manually
> - allow internet requires your permission (internet is required to clone the git)
>   - :phone: a verified kaggle account is required
> - :key: Allow [Kaggle secrets](https://www.kaggle.com/discussions/product-feedback/114053) to access wandb:
>   - `wandb_api_key`: weights and biases API key.
> - You'll need to manually edit the notebook under kaggle web page to allow secrets.
> - Quick save your notebook.
> - Now run the remote training script again, this should execute. 

:heart: Don't be scared, the provided experiments will go very fast (less than 2 minutes to run on kaggle).

### Local training
`python train.py -e 0 1`


-----
:warning: This is fully experimental, there are probably much better ways to wrap an existing training script.

:mag: Want to contribute, new features, spotted a bug under your OS? file an [issue here](https://github.com/balthazarneveu/mva_pepites/issues)

:key: It is possible to work with private github repositories but it will require your github token to be inserted into kaggle secrets.

:star: Give a star to this repo if you're planning using it.

-----