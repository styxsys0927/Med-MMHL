# Med-MMHL
This is the repository of the dataset corresponding to the article [Med-MMHL: A Multi-Modal Dataset for Detecting Human- and
LLM-Generated Misinformation in the Medical Domain](https://arxiv.org/pdf/2306.08871.pdf). The data can be found at [here](https://drive.google.com/drive/folders/1aB3c5CuPZ8hzcbUZFg6uE4MRlx-2jdk_?usp=sharing).

## Dataset Description ##
The data are already split into train/dev/test sets. 

Below tables summarize the task and its source path, where the statistics is in Tab 2 of our [paper](https://arxiv.org/pdf/2306.08871.pdf).

| Task                              | Benchmarked Results | Data Location                                |
| --------------------------------- | ------------------- | -------------------------------------------- |
| Fake news<br>detection            | Tab 3 in [paper](https://arxiv.org/pdf/2306.08871.pdf)      | [fakenews_article](https://drive.google.com/drive/folders/1UVnU57NOUbtxAX-tOzSIa6fPZiDtpWoc?usp=drive_link)                             |
| LLM-generated fake sent detection | Tab 3 in [paper](https://arxiv.org/pdf/2306.08871.pdf)      | [sentence](https://drive.google.com/drive/folders/15WI-FKK5B-SviSN2aEu5RQWg0gOtAsRX?usp=drive_link)                                     |
| Multimodal fake news detection    | Tab 3 in [paper](https://arxiv.org/pdf/2306.08871.pdf)      | [image_article](https://drive.google.com/drive/folders/1iuF9LaGG9Yz5wGsR7hBg4tPaR6WgZv64?usp=drive_link) |
| Fake tweet detection              | Tab 4 in [paper](https://arxiv.org/pdf/2306.08871.pdf)      | [fakenews_tweet](https://drive.google.com/drive/folders/1qoncX_CD4slkKU2Ylk12A8HiRy5R5b5A?usp=drive_link)                               |
| Multimodal tweet detection        | Tab 4 in [paper](https://arxiv.org/pdf/2306.08871.pdf)      | [image_tweet](https://drive.google.com/drive/folders/12__mlVW4gVoxh_mESB_g4EJofpjaZ2qh?usp=drive_link)   |


For multimodal tasks, the paths to the images are stored in the column ***image***. The path looks like */images/2023-05-09_fakenews/LeadStories/551_32.png* for news. You do not need to modify the path of the ***images*** folder in the root directory of your project.

The content and images of tweets can be crawled with the code ***collect_by_tweetid_tweepy_clean.py*** or other legal twitter extraction tool given tweet IDs.

## Enviroment Configure ##

> conda create -f clip_env.yaml
>
> conda activate clip_env

## Running Baselines ##

Most of our baselines are drawn from [Hugging Face](https://huggingface.co/), so you need to provide the name of the models to make the code run. The Hugging Face models included in our baseline experiments are listed below.

| Model Name          | Hugging Face Name                            |
| ------------------- | -------------------------------------------- |
| BERT                | [bert-base-cased](https://huggingface.co/bert-base-cased) |
| BioBERT             | [pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb](https://huggingface.co/pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb) |
| Funnel Transformer  | [funnel-transformer/medium-base](https://huggingface.co/funnel-transformer/medium-base) |
| FN-BERT             | [ungjus/Fake_News_BERT_Classifier](https://huggingface.co/ungjus/Fake_News_BERT_Classifier) |
| SentenceBERT        | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| DistilBERT          | [sentence-transformers/msmarco-distilbert-base-tas-b](https://huggingface.co/sentence-transformers/msmarco-distilbert-base-tas-b) |
| CLIP                | [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) |
| VisualBERT          | [uclanlp/visualbert-vqa-coco-pre](https://huggingface.co/uclanlp/visualbert-vqa-coco-pre) |

Below are some examples of training and testing the Hugging Face models. Please refer to the code to explore more editable arguments. 

To train a fine-tuned version of bioBERT, the command looks like this:

```shell
python fake_news_detection_main.py \
    -bert-type pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb \
    -device 0 \
    -batch-size 4 \
    -benchmark-path path/to/your/data \
    -dataset-type fakenews_article
```

To test an existing model, the command is:

```shell
python fake_news_detection_main.py \
    -bert-type pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb \
    -device 0 \
    -batch-size 4 \
    -benchmark-path path/to/your/data \
    -dataset-type fakenews_article \
    -snapshot path/to/your/model \
    -test
```

Similarly, to train and test a multimodal model, the commands are:

```shell
python fake_news_detection_multimodal_main.py \
    -clip-type uclanlp/visualbert-vqa-coco-pre \
    -device 0 \
    -batch-size 4 \
    -benchmark-path path/to/your/data \
    -dataset-type image_article
```

and 

```shell
python fake_news_detection_multimodal_main.py \
    -clip-type uclanlp/visualbert-vqa-coco-pre \
    -device 0 \
    -batch-size 4 \
    -benchmark-path path/to/your/data \
    -dataset-type image_article \
    -snapshot path/to/your/model \
    -test
```
