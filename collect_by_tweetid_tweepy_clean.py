import re
import pandas as pd
import os
import tweepy

# following keys are generated through the twitter developer
consumer_key = 'XXXXX'
consumer_secret = 'XXXXX'
access_token_key = 'XXXXX'
access_token_secret = 'XXXXX'

Bearer_Token = 'XXXXXX'

auth = tweepy.OAuth2BearerHandler(Bearer_Token)
api = tweepy.API(auth)

def read_text_img(tweet_id): # you can add more key word you like or other tweet dehydrate tool/package
    try:
        tweet = api.get_status(tweet_id, tweet_mode='extended')
    except:
        return None, None

    text = tweet.full_text
    # print(text)

    media_urls = []
    if 'media' in tweet.entities:
        for media in tweet.entities['media']:
            if media['type'] == 'photo':
                media_urls.append(media['media_url'])

    return text, media_urls

def recover_tweet_file(csv_path, id_keyword):
    df_s = pd.read_csv(csv_path)
    df_s.insert(df_s.shape[1], 'content', '')
    df_s.insert(df_s.shape[1], 'img_link', '')

    df_copy = df_s.copy()

    row, col = df_s.shape
    for i in range(row):
        if i % 1000 == 0:
            print(i)
        cur_tweet_id = df_s.iloc[i][id_keyword]
        cur_tweet_id = re.findall(r"\d+\.?\d*", cur_tweet_id)[0]
        tweet_content, tweet_imglink = read_text_img(cur_tweet_id)
        if tweet_content != None:
            mid_d = dict(df_copy.iloc[i])
            mid_d['content'] = tweet_content
            mid_d['img_link'] = tweet_imglink
            df_copy.iloc[i] = pd.Series(mid_d)
    df_copy.to_csv('./collected_tweet.csv')
    print('finished')


root_path = 'a path to your dataset path'

tweet_file = os.path.join(root_path, 'tweet_id_csv_file_name')

tweet_id_keyword = 'tweet_id'

recover_tweet_file(tweet_file, tweet_id_keyword)





