# Extending a Parser to Distant Domains

This repository contains the code used to generate the results described in [Extending a Parser to Distant Domains Using a Few Dozen Partially Annotated Examples](TODO) from ACL 2018, forked from the [Minimal Span Parser](https://github.com/mitchellstern/minimal-span-parser) repository.


## Requirements and Setup

* Python 3.5 or higher.
* [DyNet](https://github.com/clab/dynet). We recommend installing DyNet from source with MKL support for significantly faster run time.
* [EVALB](http://nlp.cs.nyu.edu/evalb/). Before starting, run `make` inside the `EVALB/` directory to compile an `evalb` executable. This will be called from Python for evaluation.
* Pre-trained models. Before starting, run `unzip models/model_dev=94.48.zip` and `unzip zipped/no_elmo_model_dev=92.34.zip` in the `models/` directory to extract the pre-trained models.

## Experiments

### Training

A new model can be trained using the command `python3 src/main.py train ...` with the following arguments:

Argument | Description | Default
--- | --- | ---
`--expt-name` | The name of the experiment | N/A
`--train-trees-path` | Path to training trees | N/A
`--train-elmo-embeddings-path` | Path to training trees | N/A
`--dev-trees-path` | Path to development trees | N/A
`--dev-elmo-embeddings-path` | Path to development trees | N/A
`--numpy-seed` | NumPy random seed | Random
`--word-embedding-dim` | Dimension of the learned word embeddings | 100
`--lstm-layers` | Number of bidirectional LSTM layers | 2
`--lstm-dim` | Hidden dimension of each LSTM within each layer | 250
`--dropout` | Dropout rate for LSTMs | 0.4
`--model-path-base` | Path base to use for saving models | N/A
`--evalb-dir` |  Path to EVALB directory | `EVALB/`
`--batch-size` | Number of examples per training update | 10
`--epochs` | Number of training epochs | No limit
`--checks-per-epoch` | Number of development evaluations per epoch | 4


Any of the DyNet command line options can also be specified.

For each development evaluation, the F-score on the development set is computed and compared to the previous best. If the current model is better, the previous model will be deleted and the current model will be saved. The new filename will be derived from the provided model path base and the development F-score.

As an example, to train the parser using the default hyperparameters, you can use the command:

```
python3 src/main.py train-on-wsj --model-path-base models/model --train-trees-path data/train.trees --train-elmo-embeddings-path data/train.hdf5 --dev-trees-path data/dev.trees --dev-elmo-embeddings-path data/dev.hdf5
```


A compressed pre-trained model with these settings is provided in `models/model.zip`. See the section above for extraction instructions.

To train the model without using ELMo word embeddings, set `--train-elmo-embeddings-path` and `--dev-elmo-embeddings-path` to `None`. A model with this setting is provided in `models/no_elmo_model.zip`.

## Evaluation

A saved model can be evaluated on a test corpus using the command `python3 src/main.py test ...` with the following arguments:

Argument | Description | Default
--- | --- | ---
`--model-path-base` | Path base of saved model | N/A
`--evalb-dir` |  Path to EVALB directory | `EVALB/`
`--trees-path` | Path to test trees | N/A
`--elmo-embeddings-path` | Path to ELMo embeddings | N/A

As above, any of the DyNet command line options can also be specified.

As an example, after extracting the pre-trained model, you can evaluate it on the test set using the following command:

```
python3 src/main.py test --model-path-base models/model_dev=94.48 --trees-path data/test.trees --elmo-embeddings-path data/test.hdf5
```

The pre-trained model obtains F-scores of 94.48 on the development set and 94.28 on the test set. The pre-trained model without ELMo obtain an F-score of 92.34 on the development set.

TODO specify how exactly the elmo hdf5 files work and where to get them from etc!

## Parsing New Sentences
The `parse` method of the parser can be used to parse new sentences. In particular, `parser.parse(sentence, elmo_embeddings)` will return a the predicted tree.

See the `run_test` function in `src/main.py` for an example of how a parser can be loaded from disk and used to parse sentences.

## Citation

If you use this software for research, please cite our paper as follows:

```
TODO
```
