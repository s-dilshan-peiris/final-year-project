import pandas as pd
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
import seaborn as sns
import tensorflow as tf
import pickle
import numpy as np


def rnn_model():
    data = pd.read_csv("rnn_training_features.csv", encoding='utf-8')

    X = data["conated_comments"]
    y = data["polarity_emotions_ID"]

    print("x shape:", X.shape)
    print("y shape:", y.shape)

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=9000, split=' ')
    tokenizer.fit_on_texts(data["conated_comments"].values)
    X_seq = tokenizer.texts_to_sequences(data["conated_comments"].values)
    # X_seq = pad_sequences(X)

    sent_max_len = 150
    X_seq = tf.keras.utils.pad_sequences(X_seq, sent_max_len, padding="pre")
    print(X_seq, len(X_seq))

    lb = LabelEncoder()
    data["polarity_emotions_ID"] = lb.fit_transform(data["polarity_emotions_ID"])

    # unique_data = data["polarity_emotions"].unique()
    # print(unique_data)
    # emotion_category = data[data['polarity_emotions_ID'] == i]
    # y = data["polarity_emotions_ID"]

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(9000, 64, input_length=sent_max_len))
    model.add(tf.keras.layers.SpatialDropout1D(0.2))
    model.add(tf.keras.layers.LSTM(176, recurrent_dropout=0.2))  ##, dropout=0.2,
    model.add(tf.keras.layers.Dense(15, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    summary = model.summary()
    print(summary)

    # Splitting the data into training and testing
    y = pd.get_dummies(data["polarity_emotions_ID"])
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y, test_size=0.2, random_state=42)

    print("X_test", X_test, X_test[1], type(X_test))
    batch_size = 32
    model.fit(X_train, y_train, epochs=10, batch_size=batch_size, verbose='auto')

    eval = model.evaluate(X_test, y_test)
    print("eval", eval)

    # pickle.dump(model,open("rnn_model.pickle", 'wb'))

    # logging to a txt file
    # with open("reports/info.txt", 'a', encoding="utf-8") as log:
    #     log.write("\n\n\nLSTM RNN Summary\n")
    #     log.write(summary)
    #     log.write("\nModel evaluation\n")
    #     log.write(eval)
    #     log.write("\n")


def full_train_LSTM_RNN():
    data = pd.read_csv("../datasets/rnn_training_features.csv", encoding='utf-8')
    data = data.sample(frac=1).reset_index(drop=True)

    X = data["conated_comments"]
    # y = data["polarity_emotions_ID"]
    y = data["polarity_emotions"]

    print(y)
    count_per_class = y.value_counts().to_numpy()
    unique_labels = np.unique(y)
    # print("y.value_counts", unique_labels)

    # class_weights = class_weight.compute_class_weight('balanced', unique_labels, y.values)
    # print(class_weights)
    # tokenizer = Tokenizer(num_words=9000, split=' ')
    # tokenizer.fit_on_texts(data["conated_comments"].values)
    # X_seq = tokenizer.texts_to_sequences(X.values)

    # cv = CountVectorizer(lowercase=False)
    tokenizer_10 = tf.keras.preprocessing.text.Tokenizer(num_words=7000, split=' ')
    tokenizer_10.fit_on_texts(X.values)

    X_seq = tokenizer_10.texts_to_sequences(X.values)
    pickle.dump(tokenizer_10, open("../resources/tokenizer_10.pickle", 'wb'))

    # X_seq = cv.fit_transform(X).toarray()
    # pickle.dump(cv, open("Cvectorizer_rnn.pickle", 'wb'))

    sent_max_len = 150
    X_seq = tf.keras.utils.pad_sequences(X_seq, sent_max_len, padding="pre")
    print(X_seq, len(X_seq))

    # lb = LabelEncoder()
    # y = lb.fit_transform(y)

    y = pd.get_dummies(y)
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y, test_size=0.2, random_state=42, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, shuffle=True)

    print("y_test", y_test)
    print("y_test", y_test.columns)
    unique_labels = np.unique(y_train)
    print("y.value_counts", unique_labels)
    # # create a logistic regression model_10
    # model_10 = LogisticRegression(random_state=42)
    #
    # # define the class weights to be tested
    # class_weights = [{0: 1, 1: w} for w in np.arange(0.1, 5, 0.1)]
    #
    # # define the parameter grid to be searched
    # param_grid = {'class_weight': class_weights}
    #
    # # define the scorer to be used for grid search
    # scorer = make_scorer(f1_score)
    #
    # # perform grid search with cross-validation
    # grid_search = GridSearchCV(estimator=model_10, param_grid=param_grid, cv=5, scoring=scorer)
    # grid_search.fit(X_train, y_train)
    #
    # # print the best set of class weights
    # print("Best class weights:", grid_search.best_params_)

    weights = {}
    unique, counts = np.unique(y_train, return_counts=True)
    for i in range(len(unique)):
        weights[unique[i]] = sum(counts) / counts[i]

    print(weights)
    # Compute class weights

    model_10 = tf.keras.Sequential()
    model_10.add(tf.keras.layers.Embedding(7000, 64, input_length=sent_max_len, mask_zero=True))
    model_10.add(tf.keras.layers.SpatialDropout1D(0.4))
    model_10.add(
        tf.keras.layers.LSTM(128, recurrent_dropout=0.4, name='lstm1', return_sequences=True))  ##, dropout=0.2,
    model_10.add(tf.keras.layers.LSTM(64, name='lstm2'))
    # model_10.add(LSTM(32, name='lstm2'))
    model_10.add(tf.keras.layers.Dense(15, activation='softmax', name='dense1'))

    # class_weights = {1: 1.0, 2: 2.0, 3: 1.5,4 : 3.0, 5: 1.0, 6: 2.0, 7: 1.5, 8: 3.0, 9: 1.0, 10: 2.0, 11: 1.5, 12: 3.0,
    #                  13: 1.0, 14: 2.0, 15: 1.5}

    # class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model_10.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # model_10.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], weighted_metrics=weights)
    # optimizer = Adam(lr=0.001)
    # model_10.compile(loss={'lstm1': 'categorical_crossentropy','lstm2': 'categorical_crossentropy', 'dense1': 'mse'} , loss_weights={'lstm1': 1.0, 'lstm2': 1.0, 'dense1': 0.5},
    #               optimizer=optimizer, metrics=['accuracy'], weighted_metrics=class_weights)
    summary = model_10.summary()
    print(summary)

    batch_size = 32
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')
    history = model_10.fit(X_train, y_train, validation_data=(X_val, y_val), callbacks=[early_stop], epochs=10,
                           verbose='auto', shuffle=True, batch_size=batch_size)

    eval = model_10.evaluate(X_test, y_test)
    print("full_train_LSTM_RNN eval", eval)

    y_pred = model_10.predict(X_test)
    print(y_pred)

    target_names = [' negative anger', ' negative joy', ' negative love', ' negative sad',
                    ' negative sarcasm', ' neutral anger', ' neutral joy', ' neutral love',
                    ' neutral sad', ' neutral sarcasm', ' positive anger', ' positive joy',
                    ' positive love', ' positive sad', ' positive sarcasm']

    print(classification_report(np.array(y_test).argmax(axis=1), y_pred.argmax(axis=1),
                                zero_division=1, target_names=target_names))
    conf_mat = confusion_matrix(np.array(y_test).argmax(axis=1), y_pred.argmax(axis=1))
    print(conf_mat)
    weighted_F1score = f1_score(np.array(y_test).argmax(axis=1), y_pred.argmax(axis=1),
                                average="weighted")
    print("weighted_F1score", weighted_F1score)

    # Get the accuracy and loss for each epoch
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Plot training and validation accuracy
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')

    # Plot training and validation loss
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')

    # Add title, xlabel and ylabel
    plt.title('Training and Validation Accuracy and Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy and Loss')

    # Add legend
    plt.legend()

    # Display the plot
    plt.show()

    pickle.dump(model_10, open("../resources/full_trained_rnn_model_10.pickle", 'wb'))

    # y_pred = model_10.predict(X_test)
    # # inverse_pred = y_pred.idxmax(axis=1)
    # # print('inverse_pred', inverse_pred)
    # print('y_test', y_test)
    # print('y_pred', y_pred)
    #
    # eval = model_10.evaluate(X_test, y_test)
    # print("emotion model_10 eval", eval)
    # target_names = ['anger', 'joy', 'love', 'sad', 'sarcasm']
    # print(classification_report(np.array(y_test).argmax(axis=1),
    #                             y_pred.argmax(axis=1), zero_division=1,
    #                             target_names=target_names))
    # conf_mat = confusion_matrix(np.array(y_test).argmax(axis=1),
    #                             y_pred.argmax(axis=1))
    #
    # weighted_F1score = f1_score(np.array(y_test).argmax(axis=1), y_pred.argmax(axis=1), average="weighted")
    # print("weighted_F1score", weighted_F1score)
    # print(conf_mat)
    # precision = precision_score(y_test, y_pred)
    # recall = recall_score(y_test, y_pred)
    #
    #
    # print("Precision: {:.2f}".format(precision))
    # print("Recall: {:.2f}".format(recall))


def best_params():
    data = pd.read_csv("rnn_training_features.csv", encoding='utf-8', index_col=False)
    data = data.sample(frac=1).reset_index(drop=True)

    X = data["conated_comments"]
    y = data["polarity_emotions_ID"]

    cv = CountVectorizer(lowercase=False)

    X_seq = cv.fit_transform(X).toarray()
    # pickle.dump(cv, open("Cvectorizer_rnn.pickle", 'wb'))

    sent_max_len = 150
    X_seq = tf.keras.utils.pad_sequences(X_seq, sent_max_len, padding="pre")
    print(X_seq, len(X_seq))

    lb = LabelEncoder()
    y = lb.fit_transform(y)

    y = pd.get_dummies(y)
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y, test_size=0.4, random_state=42)
    print(X_train.shape[0])

    # Define the LSTM model
    def create_model(units=50, learning_rate=0.001):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Embedding(6000, 64, input_length=sent_max_len, mask_zero=True))
        model.add(tf.keras.layers.SpatialDropout1D(0.2))
        model.add(
            tf.keras.layers.LSTM(128, recurrent_dropout=0.2, name='lstm1', return_sequences=True))  ##, dropout=0.2,
        model.add(tf.keras.layers.LSTM(32, name='lstm2'))
        model.add(tf.keras.layers.Dense(15, activation='softmax', name='dense1'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    # Define the hyperparameter grid to be searched
    param_grid = {'units': [50, 100, 150],

                  'learning_rate': [0.001, 0.01, 0.1]}

    # Create the KerasClassifier wrapper for GridSearchCV
    model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=32, verbose='auto')

    # Perform GridSearchCV with cross-validation
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters
    print(grid_search.best_params_)


def predicting_from_rnn(real_coments):
    # print("real_coments", real_coments)
    # data = pd.read_csv("src/models/neural_net.py", encoding='utf-8')
    # data = pd.read_csv("rnn_training_features.csv", encoding='utf-8')
    # y = data["polarity_emotions_ID"]

    model = pickle.load(open("src/resources/full_trained_rnn_model_10.pickle", 'rb'))
    # model = pickle.load(open("resources/full_trained_rnn_model.pickle", 'rb'))

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=7000, split=' ')
    tokenizer.fit_on_texts(real_coments)
    tokens = tokenizer.texts_to_sequences(real_coments)
    sent_max_len = 150
    padded_vect = tf.keras.utils.pad_sequences(tokens, sent_max_len, padding="pre")
    results = list(model.predict(padded_vect))

    indices = np.argmax(results, axis=1)
    # print("indices",indices)
    # categories = np.array(y)[indices]
    # category_names = []
    # print(categories)
    categories_values = {0: "negative anger", 1: "negative joy", 2: "negative love", 3: "negative sad",
                         4: "negative sarcasm",
                         5: "neutral anger", 6: "neutral joy", 7: "neutral love", 8: "neutral sad",
                         9: "neutral sarcasm", 10: "positive anger",
                         11: "positive joy", 12: "positive love", 13: "positive sad", 14: "positive sarcasm"}

    sentiment_emotion_categories = [categories_values[label] for label in indices]

    print("sentiment_emotion_categories", sentiment_emotion_categories)
    # for categoryID in categories:
    #     category_name = list(categories_values.keys())[list(categories_values.values()).index(int(categoryID))]
    #     category_names.append(category_name)

    average_results = []
    total = 0.0
    for result in results:
        for vectors in result:
            total += float(vectors)
        average_result = total / len(results)
        average_results.append(average_result)

    # results_df = pd.DataFrame({"comments": real_coments, "results": results, "average_results": average_results,
    #                            "category": categories, "category_names": category_names})
    results_df = pd.DataFrame({"comments": real_coments, "category_names": sentiment_emotion_categories})
    # print(results_df)

    results_df.to_csv("src/reports/LSTM RNN_results.csv", encoding='utf-8', index_label=True)
    # results_df.to_csv("reports/LSTM RNN_results.csv", encoding='utf-8', index_label=True)

    occurrences = pd.Series(sentiment_emotion_categories, name="").value_counts(ascending=True)
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
    full_train_LSTM_RNN()
