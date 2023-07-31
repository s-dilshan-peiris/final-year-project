import json
import sys

from flask import jsonify
from pytube import extract
import googleapiclient.discovery
import emoji
import re
import pandas as pd
from urllib.parse import urlparse

from flaskProject.backend.src.models.classification_model import predicting_from_linear_svc
from flaskProject.backend.src.models.neural_net import predicting_from_rnn
from flaskProject.backend.src.models.separate_models_rnns import seperate_model_prediction_and_result_concatanation
from flaskProject.backend.src.models.unsupervised_learning import predicting_from_kmeans, \
    comparison_with_rnn_predictions
from flaskProject.backend.src.utils.utils import extract_only_sinhala, realtime_clean_emoji_unicodes, \
    unique_cleaned_emoji_unicodes, emojizing, to_csv

# from src.models.classification_model import predicting_from_linear_svc
# from src.models.neural_net import predicting_from_rnn
# from src.models.separate_models_rnns import seperate_model_prediction_and_result_concatanation
# from src.models.unsupervised_learning import predicting_from_kmeans, comparison_with_rnn_predictions
# from src.utils.utils import extract_only_sinhala, realtime_clean_emoji_unicodes, unique_cleaned_emoji_unicodes, \
#     emojizing, to_csv

sinhala_letters = "['අ', 'ආ', 'ඇ', 'ඈ', 'ඉ', 'ඊ', 'උ', 'ඌ', 'ඍ', 'ඎ', 'ඏ', 'ඐ', 'එ', 'ඒ', 'ඓ','ඔ', 'ඕ', 'ඖ', 'ක', " \
                  "'ඛ', 'ග', 'ඝ', 'ඞ', 'ඟ', 'ච', 'ඡ', 'ජ', 'ඣ', 'ඤ', 'ඦ', 'ට','ඨ', 'ඩ', 'ඪ', 'ණ', 'ඬ', 'ත', 'ථ', 'ද', " \
                  "'ධ', 'න', 'ඳ', 'ප', 'ඵ', 'බ', 'භ', 'ම', 'ඹ','ය', 'ර', 'ල', 'ව', 'ශ', 'ෂ', 'ස', 'හ', 'ළ', 'ෆ', 'ං', " \
                  "'ඃ', '්', 'ා', 'ැ', 'ෑ', 'ි', 'ී','ු', 'ූ', 'ෘ', 'ෙ', 'ේ', 'ෛ', 'ො', 'ෝ', 'ෞ', 'ෟ', 'ෲ', 'ෳ', '෴'] "


def is_youtube_url(url):
    parsed_url = urlparse(url)
    return parsed_url.netloc == 'www.youtube.com' or parsed_url.netloc == 'youtube.com'


def magic(youtubeUrl):
    video_link = str(youtubeUrl)
    # is_yt_url = is_youtube_url(video_link)
    # if not is_yt_url:
    #     concat_comments = []
    #     return concat_comments
    # else:
    video_id = extract.video_id(video_link)
    print("video Link:.." + video_link, "video Id:.." + video_id)

    comments_list = []
    next_page_token = None
    while 1:
        api_service_name = "youtube"
        api_version = "v3"
        DEVELOPER_KEY = "AIzaSyA5NvLoVebwn8GdEIQ2ljAiW4OxX5_-z1g"
        results_per_page = 0
        no_of_comments = 0
        youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=DEVELOPER_KEY)
        request = youtube.commentThreads().list(
            part="id,replies,snippet",
            maxResults=100,
            videoId=video_id,
            pageToken=next_page_token
        )
        api_response = request.execute()
        next_page_token = api_response.get('nextPageToken')
        for item in api_response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            # print(comment)
            comments_list.append(comment)
        if next_page_token is None:
            break
    comment_list_len = len(comments_list)
    print('comment_list_len', comment_list_len)

    filtered_comments = []
    for comment in comments_list:
        emoji_count = emoji.emoji_count(comment)
        english = re.findall("[a-zA-Z]", comment)
        sinhala_check = re.findall(sinhala_letters, comment)
        if emoji_count and len(comment) > 0 and sinhala_check and not english:
            filtered_comments.append(comment)

    pure_comments = []
    emojis_in_comment = []
    cleaned_emoji_unicodes = []
    for filtered_comment in filtered_comments:
        pure_text = extract_only_sinhala(filtered_comment)
        pure_comments.append(pure_text)
        pure_emojis = str(filtered_comment).replace(pure_text, "")
        emojis_in_comment.append(pure_emojis)

    for pure_emojis in emojis_in_comment:
        demojize_unicode = emoji.demojize(pure_emojis).split(" ")
        cleaned_emoji_unicodes.append(realtime_clean_emoji_unicodes(demojize_unicode))
    unique_emoji_unicodes = unique_cleaned_emoji_unicodes(cleaned_emoji_unicodes)
    unique_emojis_in_comment = emojizing(unique_emoji_unicodes)

    concated_comments = to_csv(video_id, video_link, filtered_comments, pure_comments, emojis_in_comment,
                               cleaned_emoji_unicodes, unique_emoji_unicodes, unique_emojis_in_comment)

    print("data", type(concated_comments))
    if len(concated_comments) > 0:
        # print("comments in model platform", concated_comments)
        return concated_comments
    else:
        concat_comments = []
        return concat_comments

    # print("after saved filename",saved_filename)
    # data = pd.read_csv(str(saved_filename), encoding='utf-8')
    # data = pd.read_csv("src/real_time_data/2UqjfKdaVGA.csv")
    # data = pd.DataFrame(data)

    # print("data", concated_comments)
    # comments = data["concat_comment"]
    # if len(comments) > 0:
    #     print("comments in model platform", comments)
    #     return comments
    # else:
    #     concat_comments = []
    #     return concat_comments


def magic_for_text(comment):
    filtered_comments = []
    comments_list = []
    comments_list.append(comment)
    for comment in comments_list:
        emoji_count = emoji.emoji_count(comment)
        english = re.findall("[a-zA-Z]", comment)
        sinhala_check = re.findall(sinhala_letters, comment)
        if emoji_count and len(comment) > 0 and sinhala_check and not english:
            filtered_comments.append(comment)

    pure_comments = []
    emojis_in_comment = []
    cleaned_emoji_unicodes = []
    for filtered_comment in filtered_comments:
        pure_text = extract_only_sinhala(filtered_comment)
        pure_comments.append(pure_text)
        pure_emojis = str(filtered_comment).replace(pure_text, "")
        emojis_in_comment.append(pure_emojis)

    for pure_emojis in emojis_in_comment:
        demojize_unicode = emoji.demojize(pure_emojis).split(" ")
        cleaned_emoji_unicodes.append(realtime_clean_emoji_unicodes(demojize_unicode))
    unique_emoji_unicodes = unique_cleaned_emoji_unicodes(cleaned_emoji_unicodes)
    unique_emojis_in_comment = emojizing(unique_emoji_unicodes)

    video_id = ""
    video_link = ""
    concated_comments = to_csv(video_id, video_link, filtered_comments, pure_comments, emojis_in_comment,
                               cleaned_emoji_unicodes, unique_emoji_unicodes, unique_emojis_in_comment)

    if len(concated_comments) > 0:
        # print("comments in model platform", concated_comments)
        return concated_comments
    else:
        concat_comments = []
        return concat_comments

    # print("after saved filename")
    # data = pd.read_csv(saved_filename, encoding='utf-8')
    # data = pd.DataFrame(data)
    #
    # # print("data", data)
    # comments = data["concat_comment"]
    # if len(comments) > 0:
    #     print("comments in model platform", comments)
    #     return comments
    # else:
    #     concat_comments = []
    #     return concat_comments


def classify_from_linear_svc(concat_comments):
    if len(concat_comments) > 0:
        occurence_results = predicting_from_linear_svc(concat_comments)
        return occurence_results
    else:
        return "Video possess no valid comments for analyzing"


def classify_from_neural_net(concat_comments):
    # print("concat_comments", concat_comments)

    # predicting_from_rnn(concat_comments)
    if len(concat_comments) > 0:
        occurence_results = predicting_from_rnn(concat_comments)
        return occurence_results
    else:
        return "Video possess no valid comments for analyzing"

    # print("occurence_results",occurence_results)


def classifying_from_kmeans(concat_comments):
    # predicting_from_kmeans(concat_comments)
    if len(concat_comments) > 0:
        occurence_results = predicting_from_kmeans(concat_comments)
        return occurence_results
    else:
        return "Video possess no valid comments for analyzing"


def classifying_from_all_models(concat_comments):
    if len(concat_comments) > 0:
        separate_lstm_occurence_results = seperate_model_prediction_and_result_concatanation(concat_comments)
        rnn_occurence_results = predicting_from_rnn(concat_comments)
        linear_svc_occurence_results = predicting_from_linear_svc(concat_comments)
        kmeans_occurence_results = predicting_from_kmeans(concat_comments)
        return [separate_lstm_occurence_results, rnn_occurence_results, linear_svc_occurence_results,
                kmeans_occurence_results]
    else:
        return "Video possess no valid comments for analyzing"


def classify_from_separate_model_LSTM(concat_comments):
    if len(concat_comments) > 0:
        occurence_results = seperate_model_prediction_and_result_concatanation(concat_comments)
        print("returned model platform")
        return occurence_results
    else:
        return "Video possess no valid comments for analyzing"


def classifying_text_from_separate_model(concat_comments):
    if len(concat_comments) > 0:
        occurence_results = seperate_model_prediction_and_result_concatanation(concat_comments)
        return occurence_results
    else:
        return "Video possess no valid comments for analyzing"


def rnn_kmeans_predictions_comparisson():
    comparison_with_rnn_predictions()


def test():
    # path = "../real_time_data/2UqjfKdaVGA.csv"
    path = "../real_time_data/" + "2UqjfKdaVGA.csv"
    data = pd.read_csv(path)
    # data = pd.read_csv(path)
    print(data)


if __name__ == "__main__":
    test()
