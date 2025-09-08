# LayoutGKN: Graph Similarity Learning of Floor Plans

The code accompanies the paper [LayoutGKN: Graph Similarity Learning of Floor Plans](https://arxiv.org/abs/2509.03737), to appear in [BMVC 2025](https://bmvc2025.bmva.org/).
<br />
By 
[Casper van Engelenburg](https://caspervanengelenburg.github.io/) (👋), 
[Jan van Gemert](https://jvgemert.github.io/), 
[Seyran Khademi](https://www.tudelft.nl/en/ewi/over-de-faculteit/afdelingen/intelligent-systems/pattern-recognition-bioinformatics/computer-vision-lab/people/seyran-khademi).

![method_fig](assets/teaser.jpg)

**Abstract** -
Floor plans depict building layouts and are often represented as graphs to capture the underlying spatial relationships. 
Comparison of these graphs is critical for applications like search, clustering, and data visualization. 
The most successful methods to compare graphs i.e., graph matching networks, rely on costly intermediate cross-graph node-level interactions, therefore being slow in inference time. 
We introduce **LayoutGKN**, a more efficient approach that postpones the cross-graph node-level interactions to the end of the joint embedding architecture. 
We do so by using a differentiable graph kernel as a distance function on the final learned node-level embeddings. 
We show that LayoutGKN computes similarity comparably or better than graph matching networks while significantly increasing the speed.

## Updates

I tried to 

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
@inproceedings{van_engelenburg_layoutgkn_2025,
      title={LayoutGKN: Graph Similarity Learning of Floor Plans},
      author={van Engelenburg, Casper and van Gemert, Jan and Khademi, Seyran},
      booktitle={BMVC},
      year={2025}
}
</code></pre>
