import pandas as pd
import numpy as np
import re
import sys
import emoji
import string

# sys.path.append('../../../backend')
# import real_time_data as real_time_data

sinhala_letters = "['අ', 'ආ', 'ඇ', 'ඈ', 'ඉ', 'ඊ', 'උ', 'ඌ', 'ඍ', 'ඎ', 'ඏ', 'ඐ', 'එ', 'ඒ', 'ඓ','ඔ', 'ඕ', 'ඖ', 'ක', " \
                  "'ඛ', 'ග', 'ඝ', 'ඞ', 'ඟ', 'ච', 'ඡ', 'ජ', 'ඣ', 'ඤ', 'ඦ', 'ට','ඨ', 'ඩ', 'ඪ', 'ණ', 'ඬ', 'ත', 'ථ', 'ද', " \
                  "'ධ', 'න', 'ඳ', 'ප', 'ඵ', 'බ', 'භ', 'ම', 'ඹ','ය', 'ර', 'ල', 'ව', 'ශ', 'ෂ', 'ස', 'හ', 'ළ', 'ෆ', 'ං', " \
                  "'ඃ', '්', 'ා', 'ැ', 'ෑ', 'ි', 'ී','ු', 'ූ', 'ෘ', 'ෙ', 'ේ', 'ෛ', 'ො', 'ෝ', 'ෞ', 'ෟ', 'ෲ', 'ෳ', '෴'] "

alphabet = string.ascii_lowercase + "_" + " "
numbers = "1234567890"


def longest_comment(comment_list):
    text_len_list = []
    for text in comment_list:
        text_len_list.append(len(text))
    longest = max(text_len_list)
    return longest


def to_csv(v_id, v_link, filtered_comments, pure_comments, emojis_in_comment, cleaned_emoji_unicodes,
           unique_emoji_unicodes, unique_emojis_in_comment):
    comments = []
    data_file_path = "src/real_time_data/" + str(v_id) + ".csv"
    # data_file_path = "real_time_data/"+str(v_id)+".csv"
    print(data_file_path)
    print("to csv")
    for i in range(len(filtered_comments)):
        comment = str(pure_comments[i]) + str(unique_emoji_unicodes[i])
        comments.append(comment)
    print(len(filtered_comments), len(pure_comments), len(emojis_in_comment), len(cleaned_emoji_unicodes),
          len(unique_emoji_unicodes)
          , len(unique_emojis_in_comment), len(comments))

    new_df = pd.DataFrame({"v_link": v_link,"comments": filtered_comments, "pure_comments": pure_comments,
                           "emojis_in_comment": emojis_in_comment,
                           "pure_unicodes": cleaned_emoji_unicodes,
                           "unique_emojis_in_comment": unique_emojis_in_comment,
                           "unique_emoji_unicodes": unique_emoji_unicodes, "concat_comment": comments})
    new_df.to_csv(data_file_path, encoding='utf-8', index=False)
    # print("to csv newdf", new_df)
    # return data_file_path
    # print(comments)
    return comments


def extract_only_sinhala(comment):
    comment = str(comment)
    only_text = ""
    for i in comment:
        # print(i)
        if i in sinhala_letters:
            only_text += i
            only_text = only_text.translate(str.maketrans('', '', string.punctuation))
    # print(only_text)
    return only_text


def emojizing(unique_emoji_unicodes):
    unique_emojis_in_comment = []
    for unique_emoji_unicode in unique_emoji_unicodes:
        unique_emoji_unicode = str(unique_emoji_unicode).replace(" ", ": :")
        unique_emojis_in_comment.append(emoji.emojize(str(":" + unique_emoji_unicode + ":")))
    return unique_emojis_in_comment


def realtime_clean_emoji_unicodes(demojize_unicode):
    demojize_unicode = str(demojize_unicode).replace(':', " ")
    demojize_unicode = str(demojize_unicode).replace('.', "")
    # print(demojize_unicode)
    cleaned_emoji_unicodes = []
    only_text = ""
    for i in range(len(demojize_unicode)):
        if demojize_unicode[i] in sinhala_letters and demojize_unicode[i] not in numbers:
            pass
        else:
            only_text += demojize_unicode[i]
    cleaned_emoji_unicodes.append(only_text)
    # print(cleaned_emoji_unicodes)
    return cleaned_emoji_unicodes


def clean_emoji_unicodes(emoji_unicodes):
    # print(unicodes for unicodes in emoji_unicodes)
    alphabet = string.ascii_lowercase + "_" + " "
    clean_unicodes = ""
    for letter in emoji_unicodes:
        if letter in alphabet:
            clean_unicodes += letter
    return clean_unicodes


def unique_cleaned_emoji_unicodes(cleaned_emoji_unicodes):
    each_unicode_list = []
    unique_cleaned_unicode_list = []
    # global each_unicode_list
    for unicode in range(len(cleaned_emoji_unicodes)):
        # each_unicode =
        each_unicode_list.append(unique(cleaned_emoji_unicodes[unicode][0].split("  ")))
    for each_unicode in each_unicode_list:
        unique_cleaned_unicode_list.append(clean_emoji_unicodes(each_unicode))
    # print(unique_cleaned_unicode_list)
    return unique_cleaned_unicode_list
    # for each_unicode in each_unicode_list: each_unicode if each_unicode in


def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


# Writing data into a csv file as a list into a row
def write_to_csv(data, file_name):
    print("write_to_csv called")
    df = pd.DataFrame.from_dict(data)
    df.to_csv(file_name, header=True, encoding='utf-8')


# converting lists to more dimensional lists
def convert_list_to_2d_list(custom_list, num_of_items_in_each_list):
    n = num_of_items_in_each_list
    return [custom_list[i:i + n] for i in range(0, len(custom_list))]


# function to get unique values
def unique(list):
    x = np.array(list)
    uniques = str(np.unique(x))
    return uniques


# clear a list
def clean(list):
    for item in list:
        list.remove(item)


# remove unwanted unicode haracters in string
def in_sinhala_unicode(line):
    # 0D80 - 0DFF
    # line = 'හිට් එකක් තමා ඉතිං 😍\u200d\U0001f979'

    unicodes = ["\u200d", '&#39;', '\U0001f979', '\u200c']

    flag = 0
    for i in line:
        for j in unicodes:
            if i == j:
                line = (line.rstrip(i))
                flag = 1

    if flag == 0:
        # print("String contains the list element")
        # print(line)
        return line
    else:
        # print("String does not contains the list element")
        # print(line)
        return line


def test():
    path = "../real_time_data/"+"2UqjfKdaVGA.csv"
    data = pd.read_csv(path)
    print(data)


if __name__ == "__main__":
    test()
