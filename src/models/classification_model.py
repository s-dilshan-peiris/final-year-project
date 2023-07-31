import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import seaborn as sns
import tensorflow as tf
import pickle


def classification():
    data = pd.read_csv("../datasets/rnn_training_features.csv", encoding='utf-8', index_col=False)
    data["polarity_emotions_ID"] = data["polarity_emotions"].factorize()[0]
    transposed_data = data.head(1).T
    data_shape = data.shape
    pd.set_option("display.max_colwidth", None)
    # vectorizing
    cv = CountVectorizer(ngram_range=(1,2),max_features=None,lowercase=False)
    # pickle.dump(cv, open("Cvectorizer_max_.pickle", 'wb'))
    features = cv.fit_transform(data["conated_comments"]).toarray()
    labels = data["polarity_emotions_ID"]
    # print("features",features)
    #
    # print("Each of the %d complaints is represented by %d features (TF-IDF score of unigrams and bigrams)" % features.shape)

    # print(data.head(), data.shape)

    features_id_df = data[['polarity_emotions', 'polarity_emotions_ID']].drop_duplicates()

    # Dictionaries for future use
    features_to_id = dict(features_id_df.values)
    id_to_features = dict(features_id_df[['polarity_emotions_ID', 'polarity_emotions']].values)



    plt.figure(figsize=(15, 15))
    colors = ['grey', 'grey', 'grey', 'grey', 'grey', 'grey', 'grey', 'grey', 'grey',
              'grey', 'darkblue', 'darkblue', 'darkblue']
    data.groupby('polarity_emotions').conated_comments.count().sort_values().plot.barh(
        ylim=0, color=colors, title='NUMBER OF COMMENTS IN EACH EMOTION CATEGORY')
    plt.xlabel('Number of ocurrences', fontsize=10)

    plt.show()
    # plt.savefig("reports/Number of comments for each category")




    # Finding the three most correlated terms with each of the product categories
    # N = 20
    # for context_polarity_emoji_emotion, context_polarity_emoji_emotion_ID in sorted(features_to_id.items()):
    #     features_chi2 = chi2(features, labels == context_polarity_emoji_emotion_ID)
    #     print("features_chi2", features_chi2)
    #     indices = np.argsort(features_chi2[0])
    #     print("indices", indices)
    #     feature_names = np.array(tfidv.get_feature_names())[indices]
    #     print("feature_names", feature_names)
    #     unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    #     bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    #     # words = [v for v in feature_names if len(v.split(' ')) > 2]
    #     print("n==> %s:" % context_polarity_emoji_emotion)
    #     print("  * Most Correlated Unigrams are: %s" % (', '.join(unigrams[-N:])))
    #     print("  * Most Correlated Bigrams are: %s" % (', '.join(bigrams[-N:])))
    #     # print("  * Most Correlated words are: %s" % (', '.join(words[-N:])))

    # X = data["concated_comment"]
    # y = data["emotion"]
    #
    # X_train, X_test, y_train, y_test = train_test_split(features, labels,
    #                                                     test_size=0.25,
    #                                                     random_state=0)
    #
    # models = [
    #     RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0),
    #     LinearSVC(),
    #     MultinomialNB(),
    #     LogisticRegression(random_state=0),
    # ]
    #
    # # 5 Cross-validation
    # CV = 5
    # cv_df = pd.DataFrame(index=range(CV * len(models)))
    # entries = []
    # for model in models:
    #     model_name = model.__class__.__name__
    #     accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
    #     for fold_idx, accuracy in enumerate(accuracies):
    #         entries.append((model_name, fold_idx, accuracy))
    # cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
    # print(cv_df)
    #
    # plt.figure(figsize=(8, 5))
    # sns.boxplot(x='model_name', y='accuracy',
    #             data=cv_df,
    #             color='lightblue',
    #             showmeans=True)
    # plt.title("MEAN ACCURACY (cv = 5)n", size=14);
    # plt.show()


   # trainiing final model
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    # print("y_test",y_test.columns)
    model = LinearSVC(max_iter=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("y_pred", y_pred)
    pickle.dump(model, open("../resources/linear_svc_classification_full_train_model_2.pickle", 'wb'))

    # Classification report

    target_names = [' negative anger', ' negative joy', ' negative love', ' negative sad',
                    ' negative sarcasm', ' neutral anger', ' neutral joy', ' neutral love',
                    ' neutral sad', ' neutral sarcasm', ' positive anger', ' positive joy',
                    ' positive love', ' positive sad', ' positive sarcasm']
    print('CLASSIFICATIION METRICS')
    summary = metrics.classification_report(y_test, y_pred, target_names=target_names, zero_division = 0)
    print(summary)


    # confusion_matrix
    conf_mat = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(13, 13))
    sns.heatmap(conf_mat, annot=True, cmap="Blues", fmt='d',
                xticklabels=features_id_df.polarity_emotions.values,
                yticklabels=features_id_df.polarity_emotions.values)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title("CONFUSION MATRIX - LinearSVCn", size=16);
    plt.show()
    # plt.savefig("reports/CONFUSION MATRIX - LinearSVC")



    # with open("reports/info.txt", 'a', encoding="utf-8") as log:
    #     log.write("Dataset used (transposed) \n")
    #     log.write(str(transposed_data))
    #     log.write("\n\nDataset dimensions \n")
    #     log.write("     \nnumber of comments :.." +str(data_shape[0]))
    #     log.write("     \nnumber of features :.." +str(data_shape[1]))
    #     log.write("\n\nLinearSVC classifier Summary\n")
    #     log.write(summary)
    #     log.write("\nConfusion matrix\n")
    #     log.write(str(conf_mat))
    #     log.write("\n")

def full_train_model():
    data = pd.read_csv("../datasets/rnn_training_features.csv", encoding='utf-8',index_col=False)
    data = data.sample(frac=1).reset_index(drop=True)
    # data["polarity_emotions_ID"] = data["polarity_emotions"].factorize()[0]
    # transposed_data = data.head(1).T
    # data_shape = data.shape
    X = data["conated_comments"]
    # y = data["polarity_emotions_ID"]
    y = data["polarity_emotions"]

    print(y)
    tokenizer_10 = tf.keras.preprocessing.text.Tokenizer(num_words=7000, split=' ')
    tokenizer_10.fit_on_texts(X.values)

    X_seq = tokenizer_10.texts_to_sequences(X.values)
    # pickle.dump(tokenizer_10, open("ovr_models/tokenizer_10.pickle", 'wb'))

    sent_max_len = 2581
    X_seq = tf.keras.utils.pad_sequences(X_seq, sent_max_len, padding="pre")
    print(X_seq, len(X_seq))

    X_train, X_test, y_train, y_test = train_test_split(X_seq, y, test_size=0.2, random_state=42)
    # print("features, labels",X_train, y_train)
    print("y_test", y_test)

    model = LinearSVC(random_state=42)
    model.fit(X_train, y_train)

    pickle.dump(model, open("../resources/linear_svc_classification_full_train_model_2.pickle", 'wb'))
    y_pred = model.predict(X_test)
    print("y_pred", y_pred)
    target_names = [' negative anger', ' negative joy', ' negative love', ' negative sad',
                    ' negative sarcasm', ' neutral anger', ' neutral joy', ' neutral love',
                    ' neutral sad', ' neutral sarcasm', ' positive anger', ' positive joy',
                    ' positive love', ' positive sad', ' positive sarcasm']
    summary = metrics.classification_report(y_test, y_pred, target_names=target_names,
                                            zero_division=0)
    print(summary)




# predicting unseen
def predicting_from_linear_svc(comment):
    # data = pd.read_csv("rnn_training_features.csv", encoding='utf-8', index_col=False)
    # data = pd.read_csv("../datasets/rnn_training_features.csv", encoding='utf-8', index_col=False)
    # data["polarity_emotions_ID"] = data["polarity_emotions"].factorize()[0]
    # transposed_data = data.head(1).T
    # data_shape = data.shape
    # pd.set_option("display.max_colwidth", None)

    print("model for predicting unseen")
    # print(comment)
    # comment_df = pd.DataFrame(comment)
    # print("comment_df",comment_df)
    # concat_comments = comment_df["concat_comment"]
    # labels = data["polarity_emotions_ID"]


    # tfidv = TfidfVectorizer(ngram_range=(1, 2),max_features=100)
    # vectorizer = pickle.load(open("Cvectorizer_max_.pickle", 'rb'))
    vectorizer = pickle.load(open("src/resources/Cvectorizer_max_.pickle", 'rb'))
    comments = vectorizer.fit_transform(comment).toarray()
    # print("vectorizer_comments",comments)

    # pad_max_len = longest_comment(concat_comments)
    # print("pad_max_len", pad_max_len)
    pad = tf.keras.utils.pad_sequences(comments, maxlen=2581, padding='pre')
    # print("padded sequences = {}" .format(pad))

    # model = pickle.load(open("linear_svc_classification_full_train_model_2.pickle", 'rb'))
    model = pickle.load(open("src/resources/linear_svc_classification_full_train_model_2.pickle", 'rb'))
    results = list(model.predict(pad))
    print(results)

    # indices = np.argmax(results)
    # categories = np.array(labels)[indices]
    # print(indices)
    # category_names = []
    # categories_values = {"positive love": 14, "neutral sad": 7, "neutral love": 6, "negative sad": 13,
    #                      "positive joy": 11,
    #                      "neutral joy": 1, "neutral sarcasm": 9, "negative sarcasm": 8, "positive sad": 12,
    #                      "positive sarcasm": 2, "negative anger": 0,
    #                      "neutral anger": 4, "negative joy": 5, "negative love": 10, "positive anger": 3}
    #
    # # category_names = [categories_values[label] for label in results]
    # #
    # for categoryID in results:
    #     category_name = list(categories_values.values())[list(categories_values.keys()).index(int(categoryID))]
    #     category_names.append(category_name)

    # results_df = pd.DataFrame({"comments":comment,"results":results, "category_names":category_names})
    results_df = pd.DataFrame({"comments":comment,"results":results})

    # print(results_df)

    # results_df.to_csv("reports/LinearSVC_classifier_results.csv", encoding='utf-8', index_label=True)
    results_df.to_csv("src/reports/LinearSVC_classifier_results.csv", encoding='utf-8', index_label=True)

    occurrences = pd.Series(results,name="").value_counts(ascending=True)
    occurrences_dict = occurrences.to_dict()
    occurrences_dict["total"] = sum(occurrences.values)

    for key in occurrences_dict:
        occurrences_dict[key] = (occurrences_dict[key] / occurrences_dict["total"]) * 100
    del occurrences_dict["total"]
    # print("occurrences_dict", occurrences_dict)
    chart_data = [{'type': key.replace(' ', '_'), 'value': value} for key, value in occurrences_dict.items()]
    print("occurrences_dict", chart_data)
    return chart_data

if __name__ == "__main__":
    # classification()
    full_train_model()