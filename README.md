# Med-MMHL
This is the repository of the dataset corresponding to the article "Med-MMHL: A Multi-Modal Dataset for Detecting Human- and
LLM-Generated Misinformation in the Medical Domain." The data can be found at https://drive.google.com/drive/folders/1aB3c5CuPZ8hzcbUZFg6uE4MRlx-2jdk_?usp=sharing.

The data are already split into train/dev/test sets. 

The data in "fakenews_article" are for the task "fake news detection."
The data in "fakenews_tweet" are for the task "fake tweets detection."
The data in "sentence" are for the task "LLM-generated fake sentence detection."
The data in "image_article" are for the task "multimodal fake news detection."
The data in "image_tweet" are for the task "multimodal tweets detection."

For multimodal tasks, the paths to the images are stored in the column "image." The path looks like "/images/2023-05-09_fakenews/LeadStories/551_32.png" for news. You do not need to modify the path of the "images" folder in the root directory of your project.

The content and images of tweets can be crawled with the code.
