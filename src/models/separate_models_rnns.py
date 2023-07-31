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
from sklearn.metrics import classification_report, f1_score, confusion_matrix

from flaskProject.backend.src.utils.utils import extract_only_sinhala


def training_seperate_models():
    comment_analyzing_rnn()
    emoji_analyzing_rnn()


def comment_analyzing_rnn():
    # embedding_matrix = np.array(pickle.load(open("doc2vecSIX.pickle", 'rb')))
    data = pd.read_csv('../datasets/final_features_seperated_with_ids.csv', encoding='utf-8')
    data = pd.DataFrame(data)

    X = data["sinhala_texts"]
    # y = data["polarity_ID"]
    y = data["polarity"]
    print(y.unique())

    tokenizer_9 = tf.keras.preprocessing.text.Tokenizer(num_words=7000, split=' ')
    tokenizer_9.fit_on_texts(X.values)
    X_seq = tokenizer_9.texts_to_sequences(X.values)

    pickle.dump(tokenizer_9, open("../resources/tokenizer_9.pickle", 'wb'))

    sent_max_len = 150
    X_seq = tf.keras.utils.pad_sequences(X_seq, sent_max_len, padding="pre")
    print(X_seq, len(X_seq))

    y = pd.get_dummies(y)
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y, test_size=0.2, random_state=42,
                                                        shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42,
                                                      shuffle=True)

    print("y_test", y_test)

    model_9 = tf.keras.Sequential()
    model_9.add(
        tf.keras.layers.Embedding(7000, 64, input_length=sent_max_len, mask_zero=True))  # , weights= embedding_matrix
    model_9.add(tf.keras.layers.SpatialDropout1D(0.4))
    model_9.add(tf.keras.layers.LSTM(128, recurrent_dropout=0.4, name='lstm1', return_sequences=True))  ##, dropout=0.2,
    model_9.add(tf.keras.layers.LSTM(64, name='lstm2'))
    model_9.add(tf.keras.layers.Dense(3, activation='softmax'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model_9.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    summary = model_9.summary()
    print("polarity model_9 summary", summary)

    batch_size = 32
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, mode='min')
    history = model_9.fit(X_train, y_train, validation_data=(X_val, y_val), callbacks=[early_stop],
                          epochs=10,  batch_size=batch_size, verbose='auto', shuffle=True)

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
    plt.title('Training and Validation Accuracy and Loss comment model')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy and Loss')

    # Add legend
    plt.legend()

    # Display the plot
    plt.show()

    eval = model_9.evaluate(X_test, y_test)
    print("emotion model_9 eval", eval)
    target_names = ['negative', 'neutral', 'positive']

    y_pred = model_9.predict(X_test)

    print(classification_report(np.array(y_test).argmax(axis=1),
                                y_pred.argmax(axis=1), zero_division=1,
                                target_names=target_names))
    conf_mat = confusion_matrix(np.array(y_test).argmax(axis=1),
                                y_pred.argmax(axis=1))

    weighted_F1score = f1_score(np.array(y_test).argmax(axis=1), y_pred.argmax(axis=1),
                                average="weighted")
    print("weighted_F1score", weighted_F1score)
    print(conf_mat)

    pickle.dump(model_9, open("../resources/comment_analyzing_rnn_model_9.pickle", 'wb'))

    # eval = model_9.evaluate(X_test, y_test)
    # print("polarity model_9 eval", eval)


def emoji_analyzing_rnn():
    data = pd.read_csv('../datasets/final_features_seperated_with_ids.csv', encoding='utf-8')
    data = pd.DataFrame(data)
    X = data["unique_emojis"]
    y = data["emotion"]

    print(X.unique(), X.value_counts())
    print(y.unique(), y.value_counts())

    tokenizer_9 = tf.keras.preprocessing.text.Tokenizer(num_words=7000, split=' ')
    tokenizer_9.fit_on_texts(X.values)

    X_seq = tokenizer_9.texts_to_sequences(X.values)
    # Save the tokenizer
    pickle.dump(tokenizer_9, open("../resources/tokenizer_9.pickle", 'wb'))

    sent_max_len = 150
    X_seq = tf.keras.utils.pad_sequences(X_seq, sent_max_len, padding="pre")
    # print(X_seq, len(X_seq))
    y = pd.get_dummies(y)
    X_trainOT, X_test, y_trainOT, y_test = train_test_split(X_seq, y, test_size=0.2, random_state=42,
                                                            shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_trainOT,y_trainOT, test_size=0.2, random_state=42,
                                                      shuffle=True)
    y_test = pd.get_dummies(y_test)
    y_test = np.array(y_test)

    print('y_test get_dummies', )
    print("y_train.unique", np.unique(np.array(y_train)))
    print(" y_train.value_counts", y_train.value_counts())

    model_9 = tf.keras.Sequential()
    model_9.add(tf.keras.layers.Embedding(7000, output_dim=64, input_length=sent_max_len, mask_zero=True))
    model_9.add(tf.keras.layers.SpatialDropout1D(0.4))
    model_9.add(tf.keras.layers.LSTM(128, recurrent_dropout=0.4, name='lstm1', return_sequences=True))  ##, dropout=0.2,
    model_9.add(tf.keras.layers.LSTM(64, name='lstm2'))
    model_9.add(tf.keras.layers.Dense(5, activation='sigmoid', name='dense1'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model_9.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', 'mse'])
    summary = model_9.summary()
    print("emotion model summary", summary)

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, mode='min')

    batch_size = 32
    history = model_9.fit(X_train, y_train, validation_data=(X_val, y_val), callbacks=[early_stop],
                          epochs=6, batch_size=batch_size,
                          verbose='auto', shuffle=True)

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
    plt.title('Training and Validation Accuracy and Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy and Loss')
    plt.legend()
    plt.show()

    y_pred = model_9.predict(X_test)
    # inverse_pred = y_pred.idxmax(axis=1)
    # print('inverse_pred', inverse_pred)
    print('y_test', y_test)
    print('y_pred', y_pred)

    eval = model_9.evaluate(X_test, y_test)
    print("emotion model eval", eval)
    target_names = ['anger', 'joy', 'love', 'sad', 'sarcasm']
    print(classification_report(np.array(y_test).argmax(axis=1),
                                y_pred.argmax(axis=1), zero_division=1,
                                target_names=target_names))
    conf_mat = confusion_matrix(np.array(y_test).argmax(axis=1),
                                y_pred.argmax(axis=1))

    weighted_F1score = f1_score(np.array(y_test).argmax(axis=1), y_pred.argmax(axis=1),
                                average="weighted")
    print("weighted_F1score", weighted_F1score)
    print(conf_mat)

    def hamming_loss(true_labels, predicted_labels):
        print("predicted_labels", predicted_labels)
        n = len(true_labels)
        loss = sum(true_labels[i] != predicted_labels[i] for i in range(n))
        return loss / n

    print('hamming_loss', hamming_loss(np.array(y_test).argmax(axis=1), y_pred.argmax(axis=1)))

    pickle.dump(model_9, open("../resources/2layer_emoji_analyzing_rnn_model_9_lr_0.001_ep_6.pickle", 'wb'))


# -------------------------------------------------------------------------------------------------------------
def ovr_model():
    # Load the trained multi-class classification model
    model = pickle.load(open("../resources/2layer_emoji_analyzing_rnn_model_9_lr_0.001_ep_6.pickle","rb"))

    # Get the final layer's weights
    final_layer_weights = model.layers[-1].get_weights()
    print("final_layer_weights", final_layer_weights)

    # Get the number of classes in the original problem
    num_classes = final_layer_weights[1].shape[0]

    data = pd.read_csv('../datasets/final_features_seperated_with_ids.csv', encoding='utf-8')
    data = pd.DataFrame(data)

    X = data["unique_emojis"]
    y = data["emotion"]
    print(y.unique())

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=9000, split=' ')
    tokenizer.fit_on_texts(X.values)
    X_seq = tokenizer.texts_to_sequences(X.values)

    sent_max_len = 150
    X_seq = tf.keras.utils.pad_sequences(X_seq, sent_max_len, padding="pre")
    # print(X_seq, len(X_seq))

    y = pd.get_dummies(data["emotion"])
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y, test_size=0.2, random_state=42, shuffle=True)
    print("y_test", y_test)

    ovr_models = []
    for i in range(num_classes):
        # Create a binary classification problem for the current class
        y_binary = np.zeros_like(y_train)
        y_binary[y_train == i] = 1
        y_binary[y_train != i] = 0

        # Create a new OvR model
        ovr_model = tf.keras.Sequential()
        ovr_model.add(tf.keras.layers.Dense(64, activation='relu', input_dim=sent_max_len))
        ovr_model.add(tf.keras.layers.Dense(5, activation='sigmoid'))

        # Set the weights of the binary classification layer
        ovr_model.layers[-1].set_weights(final_layer_weights)

        # Compile the OvR model
        ovr_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mse'])

        # ovr_model.save(f'ovr_models/model_class_{i}.h5')

        # Append the OvR model to the list of models
        ovr_models.append(ovr_model)

    # Use the OvR models to predict the class labels for new data
    y_pred_ovr = np.zeros_like(y_test)

    # print("y_pred_ovr above",y_pred_ovr)
    binary_classifier_predictions = []
    classifier_predictions = []
    for i in range(num_classes):
        y_pred_i = ovr_models[i].predict(X_test)
        # print("y_pred_i",y_pred_i)
        y_pred_ovr[y_pred_i >= 0.5] = i

        # print("y_pred_ovr",y_pred_ovr)

        # classifier_predictions.append(np.array(y_pred_i))

        binary_classifier_predictions.append(np.array(y_pred_ovr))
    # print("classifier_predictions",classifier_predictions)
    print("binary_classifier_predictions", binary_classifier_predictions)

    binary_classifier_predictions_result_array = np.sum(binary_classifier_predictions, axis=0)
    print("binary_classifier_predictions_result_array", binary_classifier_predictions_result_array)

    print(np.unique(y_train))
    #
    y_pred_classes = np.argmax(binary_classifier_predictions_result_array, axis=1)
    unique_classes = np.unique(np.array(y_train).argmax(axis=1))

    #
    print("y_pred_classes", y_pred_classes)
    print("unique_classes", unique_classes)
    #
    y_test_classes = unique_classes[np.array(y_test).argmax(axis=1)]
    y_pred_classes = unique_classes[y_pred_classes]

    print("y_test_classes", y_test_classes)
    print("y_pred_classes", y_pred_classes)

    target_names = ['anger', 'joy', 'love', 'sad', 'sarcasm']
    print(classification_report(np.array(y_test).argmax(axis=1),
                                binary_classifier_predictions_result_array.argmax(axis=1), zero_division=1,
                                target_names=target_names))

    weighted_F1score = f1_score(np.array(y_test).argmax(axis=1),
                                binary_classifier_predictions_result_array.argmax(axis=1), average="weighted")
    print("weighted_F1score", weighted_F1score)

    conf_mat = confusion_matrix(np.array(y_test).argmax(axis=1),
                                binary_classifier_predictions_result_array.argmax(axis=1))
    print(conf_mat)

    # classifier_predictions_result_array = np.sum(classifier_predictions, axis=0)
    # print(classifier_predictions_result_array)

    print("y_test", np.array(y_test))

    # results_of_each_class for a list of instances
    results_of_each_class = np.array(binary_classifier_predictions_result_array)

    row_sums = np.sum(results_of_each_class, axis=0)

    print(row_sums)


def seperate_model_prediction_and_result_concatanation(concat_comments):
    emotion_model = pickle.load(open("src/resources/2layer_emoji_analyzing_rnn_model_9_lr_0.001_ep_6.pickle", 'rb'))
    comment_model = pickle.load(open("src/resources/comment_analyzing_rnn_model_9.pickle", 'rb'))
    comments = []
    emojis = []
    for i in concat_comments:
        comment = extract_only_sinhala(i)
        in_emoji = str(i).replace(comment, "")
        comments.append(comment)
        emojis.append(in_emoji)

    comment_tokenizer_9 = tf.keras.preprocessing.text.Tokenizer(num_words=7000, split=' ')
    comment_tokenizer_9.fit_on_texts(comments)
    comment_seq = comment_tokenizer_9.texts_to_sequences(comments)

    emoji_tokenizer_9 = tf.keras.preprocessing.text.Tokenizer(num_words=7000, split=' ')
    emoji_tokenizer_9.fit_on_texts(emojis)
    emotion_seq = emoji_tokenizer_9.texts_to_sequences(emojis)

    sent_max_len = 150
    comment_padded_vect = tf.keras.utils.pad_sequences(comment_seq, sent_max_len, padding="pre")
    comment_results = list(comment_model.predict(comment_padded_vect))
    emotion_padded_vect = tf.keras.utils.pad_sequences(emotion_seq, sent_max_len, padding="pre")
    emoji_results = list(emotion_model.predict(emotion_padded_vect))

    comment_indices = np.argmax(comment_results, axis=1)
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    comment_categories = [label_map[label] for label in comment_indices]
    emotion_indices = np.argmax(emoji_results, axis=1)
    # emotion_categories = np.array(y_emotion)[emotion_indices]
    label_map = {0: "anger", 1: "joy", 2: "love", 3: "sad", 4: "sarcasm"}
    emotion_categories = [label_map[label] for label in emotion_indices]
    print("emotion_indices", emotion_indices)
    emotion_comment_result = []
    for x, y in zip(comment_categories, emotion_categories):
        emotion_comment_result.append(x + ' ' + y)

    print("emotion_comment_result", emotion_comment_result)
    results_df = pd.DataFrame({"comments": concat_comments, "results": emotion_comment_result})
    results_df.to_csv("src/reports/Separate LSTM RNN_results.csv", encoding='utf-8', index_label=True)
    occurrences = pd.Series(emotion_comment_result, name="").value_counts(ascending=True)
    occurrences_dict = occurrences.to_dict()
    occurrences_dict["total"] = sum(occurrences.values)

    for key in occurrences_dict:
        occurrences_dict[key] = (occurrences_dict[key] / occurrences_dict["total"]) * 100
    del occurrences_dict["total"]
    print("occurrences_dict", occurrences_dict)
    chart_data = [{'type': key.replace(' ', '_'), 'value': value} for key, value in occurrences_dict.items()]
    print("occurrences_dict", chart_data)
    return chart_data


if __name__ == "__main__":
    # comment_analyzing_rnn()
    # training_seperate_models()
    # emoji_analyzing_rnn()
    ovr_model()
