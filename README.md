
# Style Pooling
Repo for the EMNLP 2021 paper "Style Pooling: An Empirical Study of Automatic Text Style Obfuscation"

Link for pretrained models and data:[https://zenodo.org/record/4768489#](https://zenodo.org/record/4768489#)

When you download the models-data compressed folder, extract it. Place the content of the data folder, in the data folder in the code provided (uploaded as supplementary material), and place the models in corresponding model folder. This code is built upon the code  [https://github.com/cindyxinyiwang/deep-latent-sequence-model](https://github.com/cindyxinyiwang/deep-latent-sequence-model)

The instructions provided here can be used for all the tasks/datasets in the paper.  We make the examples with 3 domains. 

# Dependencies

You can find a list of dependencies in dependencies.txt

# Datasets

We have provided the preprocessing code used + the preprocessed data (after running the code) in the above anonymized link. If you want to preprocess the data yourself, in the data folder run:

```
python preprocess.py
```

then, in the root directory run:

```
bash data/blogs_3dom_cleaned/process.sh
```

You can skip this step and just use our preprocessed data. 

# Language Models for Priors

Once you have all the data extracted you can train the LMs for the priors. The LM training scripts are in the scripts forlder. You should run them from the root directory of the code. Here is how you call them:

```
bash scripts/train_lm_blogs.sh blogs_3dom_cleaned 0 2

bash scripts/train_lm_blogs.sh blogs_3dom_cleaned 1 2

bash scripts/train_lm_blogs.sh blogs_3dom_cleaned 2 2

bash scripts/train_lm_blogs.sh blogs_3dom_cleaned 3 2
```

Which trains 4 language models, one for each of the three domains, and a last one for the "one-lm" setup. We have also provided our LMs in the link on top of the page. you can download and place them in the pretrained_lm directory instead of training your own. 

# Attribute classifiers

You can train the attribute classifiers similar to the Language Models:

```
bash scripts/train_classifier_blogs_3dom_cleaned.sh
```

Alternatively, you can use the classifiers we have provided. 

# Training Our proposed VAE models

In the scripts folder for each dataset, you can find corresponding training scripts. You can run them from the root directory of the code. You can modify all the flags so that you can train models with or without de-boosting, with or without length control and so on. We have provided the scripts used to train our models. The logs and checkpoints will be saved in a "outputs_X" folder, where X is the name of the dataset. The file name includes the hypeparameters and the date of training. Here is how you train a sample model, without deboosting:

```
bash scripts/blogs_3dom_cleaned/train_blogs_3dom_boost.sh
```
with deboosting:
```
bash scripts/blogs_3dom_cleaned/train_blogs_3dom_boost.sh
```
one-lm baseline:
```
bash scripts/blogs_3dom_cleaned/train_blogs_3dom_onelm.sh
```
union:
```
bash scripts/blogs_3dom_cleaned/train_blogs_3dom_union.sh
```
You can find our models in the provided link. You just download them and extract them to the corresponding folders. The datasets/scripts with "industry" in their names are for the occupation classification experiment, relating to the fairness setup. 

# Evaluations and Reported Metrics

To get the  metrics for a given checkpoint at a given step, you can use the ipython notebooks provided in the "results_ipynb" folder. You should change the checkpoint name to match the one you need, and the notebook will run a series of bash scripts and print the results. The results are printed in such a way that you can easily paste them in google sheets and form a table. 

# The GPT-2 Model for evaluation

We use the following instantiation of the GPT-2 model for our evolutions, and use  their scripts to get scores. Please clone this repo next to where you extract our code: [https://github.com/priya-dwivedi/Deep-Learning/tree/master/GPT2-HarryPotter-Training](https://github.com/priya-dwivedi/Deep-Learning/tree/master/GPT2-HarryPotter-Training)



# Citation
```
@inproceedings{mireshghallah-style-pooling,
  title={Style Pooling: An Empirical Study of Automatic Text Style Obfuscation},
  author={Mireshghallah, Fatemehsadat and Berg-Kirkpatrick, Taylor},
  booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2021},
  month={November}
}
```
