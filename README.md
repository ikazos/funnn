# funnn

**fun** with **n**eural **n**etworks

# build

You need [nltk](https://www.nltk.org/install.html), [pytorch](https://pytorch.org/get-started/locally/) and [transformers](https://huggingface.co/transformers/installation.html). You also need a [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

Here's how I set up my environment. I use conda.

```
conda create --name funnn python=3.6
conda activate funnn
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install --user -U nltk
pip install transformers
```

# `fillmaskfillscreen.py`

Hit `q` to quit. Hit any key to advance the program.