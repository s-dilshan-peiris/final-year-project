import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import seaborn as sns
import tensorflow as tf
import pickle
from sklearn.preprocessing import scale, MaxAbsScaler


def k_means_clustering():
    data = pd.read_csv('rnn_training_features.csv', encoding='utf-8')
    X = data["conated_comments"]
    print(type(X[0]))
    vectorizer = TfidfVectorizer(lowercase=False, ngram_range=(1, 2))
    X = vectorizer.fit_transform(X).toarray()
    X = scale(X)

    inertias = []

    for i in range(8, 20):
        clustering_algo = KMeans(n_clusters=i, init='k-means++', max_iter=100, n_init=1)
        clustering_algo.fit(X)
        inertias.append(clustering_algo.inertia_)

    plt.plot(range(8, 20), inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()


def clustering():
    # clustering_algo = KMeans(n_clusters=num_of_clusters, random_state=0, n_init=15)
    # clustering_algo.fit(data)
    #
    # plt.scatter(X_padded_vect, y_padded_vect, c=clustering_algo.labels_)
    # plt.show()
    data = pd.read_csv('../datasets/rnn_training_features.csv', encoding='utf-8')

    X = data["conated_comments"]
    vectorizer = TfidfVectorizer(lowercase=False, ngram_range=(1, 2), analyzer='word')
    X = vectorizer.fit_transform(X)
    print("vectors", X)

    maxAb_scaler = MaxAbsScaler()
    X = maxAb_scaler.fit_transform(X)

    print("scales", X)
    true_k = 15
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=5)
    model.fit(X)

    inertia = model.inertia_
    print("inertia ", inertia)
    # pickle.dump(vectorizer,open("TfidfVectorizer.pickle",'wb'))
    # pickle.dump(min_max_scaler,open("min_max_scaler.pickle", 'wb'))
    # pickle.dump(model,open("clusterring_model.pickle", 'wb'))

    data['cluster'] = model.labels_
    print(data.head())
    data.to_csv("clustered_data.csv", encoding='utf-8')

    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()

    for i in range(true_k):
        print("Cluster %d:" % i),
        for ind in order_centroids[i, :10]:
            print('% s' % terms[ind])


def comparing_supervised_unsupervised():
    data = pd.read_csv('clustered_data.csv', encoding='utf-8')
    data = pd.DataFrame(data)

    data.hist()
    plt.show()

    target = data["polarity_emotions_ID"]
    labels = data["cluster"]

    plt.figure(figsize=(13, 13))
    colors = np.array(
        ["Red", "Green", "Blue", "cyan", "Magenta", "Yellow", "Black", "Brown", "Lime", "Violet", "Gray", "Orange",
         "Purple", "Gold", "Pink", "Teal"])
    plt.subplot(1, 2, 1)
    data.groupby('polarity_emotions').conated_comments.count().sort_values().plot.barh(
        ylim=0, color=colors, title='NUMBER OF COMMENTS IN EACH EMOTION CATEGORY')
    plt.xlabel('Number of occurrences', fontsize=10)

    plt.subplot(1, 2, 2)
    data.groupby('cluster').conated_comments.count().sort_values().plot.barh(
        ylim=0, color=colors, title='NUMBER OF COMMENTS IN EACH cluster')
    plt.xlabel('Number of occurrences', fontsize=10)
    plt.show()


def predicting_from_kmeans(real_comments):
    data = pd.DataFrame(real_comments)

    # vectorizer = pickle.load(open("TfidfVectorizer.pickle", 'rb'))
    vectorizer = pickle.load(open("src/resources/TfidfVectorizer.pickle", 'rb'))
    # scaler = pickle.load((open("min_max_scaler.pickle", 'rb')))
    scaler = pickle.load((open("src/resources/min_max_scaler.pickle", 'rb')))

    vectors = vectorizer.transform(real_comments)
    # scales = scaler.transform(vectors)

    # model = pickle.load(open("clusterring_model.pickle",'rb'))
    model = pickle.load(open("src/resources/clusterring_model.pickle", 'rb'))

    data['cluster'] = model.predict(vectors)
    results = data['cluster']
    print(results)
    # data['cluster'] = results

    # results_df = pd.DataFrame({"comments": real_comments, "clusters": results })
    # data.to_csv("reports/kmeans_prediction_results.csv")
    data.to_csv("src/reports/kmeans_prediction_results.csv")

    # groups = data.groupby('cluster').concat_comment.count()
    # print(("groups", groups))

    # groups_dict = groups.to_dict()
    # print("groups_dict", groups_dict)

    # groups_dict["total"] = sum(groups.values)
    # print("groups_total_dict", groups_dict)

    categories_values = {"positive love": 14, "neutral sad": 7, "neutral love": 6, "negative sad": 13,
                         "positive joy": 11,
                         "neutral joy": 1, "neutral sarcasm": 9, "negative sarcasm": 8, "positive sad": 12,
                         "positive sarcasm": 2, "negative anger": 0,
                         "neutral anger": 4, "negative joy": 5, "negative love": 10, "positive anger": 3}

    results_df = pd.DataFrame({"comments": real_comments, "results": results})

    # print(results_df)

    # results_df.to_csv("reports/LinearSVC_classifier_results.csv", encoding='utf-8', index_label=True)
    results_df.to_csv("src/reports/LinearSVC_classifier_results.csv", encoding='utf-8', index_label=True)

    occurrences = pd.Series(results, name="").value_counts(ascending=True)
    occurrences_dict = occurrences.to_dict()
    occurrences_dict["total"] = sum(occurrences.values)

    for key in occurrences_dict:
        occurrences_dict[key] = (occurrences_dict[key] / occurrences_dict["total"]) * 100
    del occurrences_dict["total"]
    print("occurrences_dict", occurrences_dict)
    chart_data = [{'type': key, 'value': value} for key, value in occurrences_dict.items()]
    print("occurrences_dict", chart_data)
    return chart_data

    # for key in groups_dict:
    #     groups_dict[key] = (groups_dict[key] / groups_dict["total"]) * 100
    # del groups_dict["total"]
    # # print("occurrences_dict", groups_dict)
    #
    # # keys_mapped_dict = {categories_values[value]: val for value, val in groups_dict.items() if value in categories_values.values()}
    # # print("keys_mapped_dict", keys_mapped_dict)
    #
    # new_dict = {}
    #
    # for keyy in groups_dict.keys():
    #     for key, value in categories_values.items():
    #         if value == keyy:
    #             new_dict[key] = groups_dict[value]
    #
    # # print("new_dict",new_dict)
    #
    # chart_data = [{'type': key.replace(' ', '_'), 'value': value} for key, value in new_dict.items()]
    # print("chart_data", chart_data)
    # return chart_data

    # results_df.to_csv("reports/kmeans_prediction_results.csv")


def comparison_with_rnn_predictions():
    lstm_data = pd.read_csv("reports/LSTM RNN_results.csv", encoding='utf-8')
    kmeans_data = pd.read_csv("reports/kmeans_prediction_results.csv", encoding='utf-8')

    lstm_data = pd.DataFrame(lstm_data)
    kmeans_data = pd.DataFrame(kmeans_data)

    plt.figure(figsize=(13, 13))
    colors = np.array(
        ["Red", "Green", "Blue", "cyan", "Magenta", "Yellow", "Black", "Brown", "Lime", "Violet", "Gray", "Orange",
         "Purple", "Gold", "Pink", "Teal"])
    plt.subplot(1, 2, 1)
    lstm_data.groupby('category_names').comments.count().sort_values().plot.barh(
        ylim=0, color=colors, title='NUMBER OF COMMENTS IN EACH EMOTION CATEGORY')
    plt.xlabel('Number of occurrences', fontsize=10)

    plt.subplot(1, 2, 2)
    kmeans_data.groupby('cluster').concat_comment.count().sort_values().plot.barh(
        ylim=0, color=colors, title='NUMBER OF COMMENTS IN EACH cluster')
    plt.xlabel('Number of occurrences', fontsize=10)
    plt.show()


if __name__ == "__main__":
    clustering()
