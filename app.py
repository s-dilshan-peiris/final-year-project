import locale
import warnings

from flask import Flask, request, jsonify
from pandas.errors import SettingWithCopyWarning

from src.model_tank import model_platform
from flask_cors import CORS
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
locale.setlocale(locale.LC_ALL, '')

app = Flask(__name__)
cors = CORS(app)


@app.route('/user_model/predict', methods=['POST'])
def analyzing():
    data = request.get_json()
    youtubeUrl = data['youtubeUrl']
    print(youtubeUrl, " kkk")
    results_dict = model_platform.classify_from_separate_model_LSTM(model_platform.magic(youtubeUrl))
    print("backend", results_dict)
    return jsonify({"model_output": results_dict, "key": "user_lstm"})


@app.route('/developer_lstm_model/predict', methods=['POST'])
def developer_lstm_analyzing():
    data = request.get_json()
    youtubeUrl = data['youtubeUrl']
    print(youtubeUrl, " kkk")
    # return "ok"
    results_dict = model_platform.classify_from_neural_net(model_platform.magic(youtubeUrl))
    print("backend", results_dict)
    return jsonify({"model_output": results_dict, "key": "lstm"})


@app.route('/developer_lsvc_model/predict', methods=['POST'])
def developer_lsvc_analyzing():
    data = request.get_json()
    youtubeUrl = data['youtubeUrl']
    print(youtubeUrl, " kkk")
    # return "ok"
    results_dict = model_platform.classify_from_linear_svc(model_platform.magic(youtubeUrl))
    print("backend", results_dict)
    return jsonify({"model_output": results_dict, "key": "lsvc"})


@app.route('/developer_kmeans_model/predict', methods=['POST'])
def developer_kmeans_analyzing():
    data = request.get_json()
    youtubeUrl = data['youtubeUrl']
    print(youtubeUrl, " kkk")
    # return "ok"
    results_dict = model_platform.classifying_from_kmeans(model_platform.magic(youtubeUrl))
    print("backend", results_dict)
    return jsonify({"model_output": results_dict, "key": "kmeans"})


@app.route('/developer_separate_models/predict', methods=['POST'])
def developer_separate_models_analyzing():
    data = request.get_json()
    youtubeUrl = data['youtubeUrl']
    print(youtubeUrl, " kkk")
    # return "ok"
    results_dict_list = model_platform.classify_from_separate_model_LSTM(model_platform.magic(youtubeUrl))
    print("backend", results_dict_list)
    return jsonify({"model_output": results_dict_list, "key": "separate_models"})


@app.route('/developer_all_models/predict', methods=['POST'])
def developer_all_models_analyzing():
    data = request.get_json()
    youtubeUrl = data['youtubeUrl']
    print(youtubeUrl, " kkk")
    # return "ok"
    results_dict_list = model_platform.classifying_from_all_models(model_platform.magic(youtubeUrl))
    print("backend", results_dict_list)
    return jsonify({"model_output": results_dict_list, "key": "all_models"})


@app.route('/text_separate_models/predict', methods=['POST'])
def text_separate_models_analyzing():
    data = request.get_json()
    text = data['text']
    # print(text, " kkk")
    # return "ok"
    results_dict_list = model_platform.classifying_text_from_separate_model(model_platform.magic_for_text(text))
    # print("backend", results_dict_list)
    return jsonify({"model_output": results_dict_list, "key": "text_separate_model"})


if __name__ == '__main__':
    app.run(debug=True)
