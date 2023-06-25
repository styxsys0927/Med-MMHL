# Med-MMHL
This is the repository of the dataset corresponding to the article "Med-MMHL: A Multi-Modal Dataset for Detecting Human- and
LLM-Generated Misinformation in the Medical Domain." The data can be found at https://drive.google.com/drive/folders/1aB3c5CuPZ8hzcbUZFg6uE4MRlx-2jdk_?usp=sharing.

## Dataset Description ##
The data are already split into train/dev/test sets. 

The data in "fakenews_article" are for the task "fake news detection."
The data in "fakenews_tweet" are for the task "fake tweets detection."
The data in "sentence" are for the task "LLM-generated fake sentence detection."
The data in "image_article" are for the task "multimodal fake news detection."
The data in "image_tweet" are for the task "multimodal tweets detection."

For multimodal tasks, the paths to the images are stored in the column "image." The path looks like "/images/2023-05-09_fakenews/LeadStories/551_32.png" for news. You do not need to modify the path of the "images" folder in the root directory of your project.

The content and images of tweets can be crawled with the code collect_by_tweetid_tweepy_clean.py.

## Running Baselines ##

Most of our baselines are drawn from Hugging Face[https://huggingface.co/], so you need to provide the name of the models to make the code run.

For example, to train a fine-tuned version of bioBERT, the command looks like this:

Markup :  `code(python fake_news_detection_main.py -bert-type pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb -device 0 -batch-size 4 -benchmark-path path/to/your/data -dataset-type fakenews_article)`

To test an existing model, the command is:

Markup :  `code(python fake_news_detection_main.py -bert-type pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb -device 0 -batch-size 4 -benchmark-path path/to/your/data -dataset-type fakenews_article -snapshot path/to/your/model -test)`

Similarly, to train and test a multimodal model, the commands are:

Markup :  `code(python fake_news_detection_multimodal_main.py -clip-type uclanlp/visualbert-vqa-coco-pre -device 0 -batch-size 4 -benchmark-path path/to/your/data -dataset-type image_article)`

and 

Markup :  `code(python fake_news_detection_multimodal_main.py -clip-type uclanlp/visualbert-vqa-coco-pre -device 0 -batch-size 4 -benchmark-path path/to/your/data -dataset-type image_article -snapshot path/to/your/model -test)`
