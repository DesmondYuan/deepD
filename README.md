# DeepD
This is the main script of DeepD v0.1

Developed and maintained by Xu lab at https://github.com/DesmondYuan/deepD


For quick start, try this with 

```
python scripts/main.py -config=configs/example_GENT_NC.json
```

If you want to discuss the usage or to report a bug, please use the 'Issues' function here on GitHub.

If you find DeepD useful for your research, please consider citing the corresponding publication.

## Installation using pip 
The following command will install pertbio from a particular branch using the '@' notation:

```
git clone https://github.com/DesmondYuan/deepD
cd deepD
pip install -r requirements.txt
```

Note that only python=3.7 is currectly supported. Anaconda or pipenv is recommended to create a python environment. 

## Quick start
The experiment type configuration file is specified by `--experiment_config_path` or `-config` and a random seed can also be assigned by using argument `-seed`

```
python scripts/main.py -config=configs/example_GENT_NC.json -seed=1234
```

## Repo structure
### ./DeepD/ - the core DeepD package
* The module DeepD.data is designed for normalization and data clipping as we described in Methods section.
* The module DeepD.model designs all the DeepD models we discussed about in the paper.
* The module DeepD.train contains core optimization methods used for training.
* The module DeepD.util contains a few helper functions for evaluation and training monitoring.

### ./scripts/ folder
This folder contains the main scripts that is ready to use.
* The `main.py` is the main script of DeepD v0.1
* The `preprocess_data.py` is the main script for data preprocessing.
For quick start, preprocessed data files can be found in the ./data/ directory and can be directly used for training.

### ./data/ folder
This folder contains preprocessed dataset for testing runs.

* `Dataset1_GENT_L1000_U133Plus2` is the dataset from this paper. We used this dataset for training normal vs cancer classification.

    _Shin G, Kang TW, Yang S, Baek SJ, Jeong YS, Kim SY. GENT: gene expression database of normal and tumor tissues. Cancer Inform 2011; 10:149-157.Shin G, Kang TW, Yang S, Baek SJ, Jeong YS, Kim SY. GENT: gene expression database of normal and tumor tissues. Cancer Inform 2011; 10:149-157._

    Update 2020: The GENT2 is release at [http://gent2.appex.kr/gent2/](http://gent2.appex.kr/gent2/) and the MySQL dataset can be accessed at [http://www.appex.kr/web_download/GENT2/GENT2_dump.sql.gz](http://www.appex.kr/web_download/GENT2/GENT2_dump.sql.gz)

* `Dataset2_GDC_L1000` is the dataset from the NCI Genome Data Commons available at [https://gdc.cancer.gov](https://gdc.cancer.gov).

* `L1000_reference.csv` is the L1000 landmark genes we used for dataset preprocessing and model training. It is defined by the [NIH LINCS project](http://www.lincsproject.org/LINCS/).

### ./configs/ folder

The folder contains the configuration files in json format. 

Example configs are provided for results reproductivity and a `debug.json` is also included for testing compilation.

An example json looks like this

```
{
    "expr_name": "Example_GENT",
    "train_dataset": "data/Dataset1_GENT_L1000_U133Plus2.experiment.csv",
    "test_dataset": "data/Dataset1_GENT_L1000_U133Plus2.withheld.csv",
    "annotation_col": "nc_label",
    "validation_ratio": 0.3,
    "n_genes": 978,
    "unsupervised_layers": [
        [978, 1000],
        [1000, 500],
        [500, 200],
        [200, 100],
        [100, 30]
    ],
   "supervised_layers": [
        [30, 30],
        [30, 30],
        [30, 30],
        [30, 2]
    ],
    "pretrain_tp2vec": true,
    "plot_pretrain_results": true,
    "train_disconnected_classifier": true,
    "train_connected_classifier": true,
    "max_iteration": 100000,
    "max_iteration_pretrain": 3000,
    "n_iter_patience": 1000,
    "n_iter_patience_pretrain": 100,
    "n_iter_buffer": 5,
    "activation": "tf.nn.relu",
    "learning_rate": 1e-3,
    "l1": 1e-4,
    "l2": 1e-2,
    "optimizer": "tf.compat.v1.train.AdamOptimizer({}, beta1=0.9, beta2=0.9)",
    "verbose": 4,
    "listen_freq": 10,
    "pretrain_batch_size": 1024,
    "batch_size": 1024
}
```

Each configuration needs the following information

```
"expr_name": (str) Label used as folder name under results
"train_dataset": (str) Location of the dataset that would be further split into training set and validation set with
                 the "validation_ratio.
"test_dataset": (str) Location of the withheld/test dataset.
"annotation_col": (str) On which column of the input data frame would a supervised model be trained to classify.
"validation_ratio": (float) The training/validation ratio for data partition used for "train_dataset".
"n_genes": (int) Number of genes from the input data.
"unsupervised_layers": (list) A list of layer sizes used for encoders and decoders.
"supervised_layers": (list) A list of layer sizes used for supervised classifier DeepDCancer.
"pretrain_tp2vec": (bool) Whether to perform unsupervised pretraining.
"plot_pretrain_results": (bool) Whether to plot the results after pretraining.
"train_disconnected_classifier": (bool) Whether to perform the disconnected supervised classification (DeepDCancer).
"train_connected_classifier": (bool) Whether to perform the connected supervised classification (DeepDcCancer).
"max_iteration": (int) Maximum number of iterations used for training.
"max_iteration_pretrain": (int) Maximum number of iterations used for pretraining.
n_iter_buffer (int): The moving window for eval losses during training.
n_iter_patience (int): How many iterations without buffered loss on validation dataset decreases would result in
                       an earlystop in training.
"n_iter_patience_pretrain":How many iterations without buffered loss on validation dataset decreases would result in
                       an earlystop in pretraining (for each layer).
learning_rate (float): The learning rate for Adam Optimizer. Note that we used the default beta1 and beta2 for Adam.
l1 (float): l1 regularization strength.
l2 (float): l2 regularization strength.
"activation": (tensorflow) Activation function for dense layers.
"optimizer": (tensorflow) Which optimizer would be used for training.
"verbose": (int) export verbose
"listen_freq": (int) Printing training loss for each # of iterations.
"pretrain_batch_size": Batch size for each iteration in pretraining.
"batch_size": Batch size for each iteration in training.
```