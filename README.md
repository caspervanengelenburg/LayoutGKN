# LayoutGKN: Graph Similarity Learning of Floor Plans

TODO: Mention the paper here + links. Include figure.

## Overview

TODO: Write a short (2-4 line) summary of the work, and it's takeaways.

## Structure

The repository has the following self-explainable structure:

```text
├── data/               
│   ├── rplan
│   ├── msd 
├── models/  # pretrained models
├── src/LayoutGKN  # source code
│   ├── constants
│   ├── data
│   ├── graph
│   ├── loss
│   ├── metrics
│   ├── model
│   ├── plot
│   ├── train
│   ├── utils
├── scripts/
│   ├── rplan_to_graphs
│   ├── nx_to_pyg
│   ├── gh_similarity
|   ├── msd_to_graphs       # tba
│   ├── generate_triplets
│   ├── train               # tba
|   ├── evaluate            # tba
│   ├── visualize           # tba
│   cfg.yaml  # general configurations
├── pyproject.toml      
├── README.md           
└── LICENSE
```

## Usage
TODO: Include usage.

For installation.
```bash
git clone https://github.com/caspervanengelenburg/LayoutGKN.git
cd LayoutGKN
pip install -e .
```

## Data

Download the RPLAN images here, and add them to the folder `data/rplan/images`.
Download the MSD dataframe here, and add it to the folder `data/msd` as `DF_msd`.

Run the following to extract the corresponding attributed access graphs from RPLAN:

```bash
cd scripts
python run rplan_to_graphs
```

The graphs will be split into a training, validation, and test part (ratios: 0.7/0.2/0.1, respectively).

Same for MSD:

```bash
cd scripts
python run msd_to_graphs
```

To get the triplets for training, run:

```bash
cd scripts
python run get_triplets
```

## Training and evaluation

For running a single training run + (optionally) log to W&B:
```bash
cd scripts
python run train.py --lr=1e-4 --num_layers=4  # etc. see cfg.yaml
```

See `pyproject.toml` for dependencies

## Citation

<pre><code>
@misc{vanengelenburg2025layoutgkn,
      title={LayoutGKN: Graph Similarity Learning of Floor Plans},
      author={van Engelenburg, Casper and van Gemert, Jan and Khademi, Seyran},
      year={2025}
}
</code></pre>
