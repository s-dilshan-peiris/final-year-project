import React, {useState} from 'react';
import {useNavigate} from "react-router-dom";
import {message} from 'antd';
import axios from "axios";
import './landing_page.css'

import "@fontsource/varela-round";
import "@fontsource/dosis";

function LandingPage() {
	const [data, setData] = useState(null)
	const [model, setModel] = useState(null)
	const [datapie, setDatapie] = useState([{}])

	const navigate = useNavigate();


	const youtubeEvent = (event) => {
		console.log(event?.target?.value, " kkk")
		setData(event?.target?.value)
	}

	const textEvent = (event) => {
		console.log(event?.target?.value, " text")
		setData(event?.target?.value)
	}


	const user_model_analyse = () => {
		console.log("analyse event : ", data)

		if (data === null) {
			return (
				message.warning(
					"Please make sure to insert a YouTube url")

			)
		}

		axios.post("http://127.0.0.1:5000/user_model/predict", {
			youtubeUrl: data,
		})
			.then(function (response) {
				console.log("user_model")
				console.log(response);
				setDatapie(response?.data?.model_output)
				console.log(response?.data?.model_output)
				if (response?.data?.model_output !== "Video possess no valid comments for analyzing") {
					navigate("/results", {state: {data: response?.data}});

				} else {
					console.log("awooo")
					return (
						message.warning(
							"Cannot Analyze! Video possess no valid comments for analyzing")

					)
				}

			})
			.catch(function (error) {

				console.log(error);
			});
	}

	const developer_lstm_model_analyse = () => {
		console.log("analyse event : ", data)
		if (data === null) {
			return (
				message.warning(
					"Please make sure to insert a YouTube url")

			)
		}
		axios.post("http://127.0.0.1:5000/developer_lstm_model/predict", {
			youtubeUrl: data,
		})
			.then(function (response) {
				console.log("lstm")
				console.log(response);
				setDatapie(response?.data?.model_output)
				console.log(response?.data?.model_output)
				if (response?.data?.model_output !== "Video possess no valid comments for analyzing") {
					navigate("/results", {state: {data: response?.data}});

				} else {
					console.log("awooo")
					return (
						message.warning(
							"Cannot Analyze! Video possess no valid comments for analyzing")

					)
				}

			})
			.catch(function (error) {

				console.log(error);
			});
	}


	const developer_lsvc_model_analyse = () => {
		console.log("analyse event : ", data)
		if (data === null) {
			return (
				message.warning(
					"Please make sure to insert a YouTube url")

			)
		}
		axios.post("http://127.0.0.1:5000/developer_lsvc_model/predict", {
			youtubeUrl: data,
		})
			.then(function (response) {
				console.log("lsvc")
				console.log(response);
				setDatapie(response?.data?.model_output)
				console.log(response?.data?.model_output)
				if (response?.data?.model_output !== "Video possess no valid comments for analyzing") {
					navigate("/results", {state: {data: response?.data}});

				} else {
					console.log("awooo")
					return (
						message.warning(
							"Cannot Analyze! Video possess no valid comments for analyzing")

					)
				}

			})
			.catch(function (error) {

				console.log(error);
			});
	}


	const developer_kmeans_model_analyse = () => {
		console.log("analyse event : ", data)
		if (data === null) {
			return (
				message.warning(
					"Please make sure to insert a YouTube url")

			)
		}
		axios.post("http://127.0.0.1:5000/developer_kmeans_model/predict", {
			youtubeUrl: data,
		})
			.then(function (response) {
				console.log("kmeans")
				console.log(response);
				setDatapie(response?.data?.model_output)
				console.log(response?.data?.model_output)
				if (response?.data?.model_output !== "Video possess no valid comments for analyzing") {
					navigate("/results", {state: {data: response?.data}});

				} else {
					console.log("awooo")
					return (
						message.warning(
							"Cannot Analyze! Video possess no valid comments for analyzing")

					)
				}

			})
			.catch(function (error) {

				console.log(error);
			});
	}


	const developer_developer_all_models_analyse = () => {
		console.log("analyse event : ", data)
		if (data === null) {
			return (
				message.warning(
					"Please make sure to insert a YouTube url")

			)
		}
		axios.post("http://127.0.0.1:5000/developer_all_models/predict", {
			youtubeUrl: data,
		})
			.then(function (response) {
				console.log("all_models")
				console.log(response);
				setDatapie(response?.data?.model_output)
				console.log(response?.data?.model_output)
				if (response?.data?.model_output !== "Video possess no valid comments for analyzing") {
					navigate("/results", {state: {data: response?.data}});

				} else {
					console.log("awooo")
					return (
						message.warning(
							"Cannot Analyze! Video possess no valid comments for analyzing")

					)
				}

			})
			.catch(function (error) {

				console.log(error);
			});
	}

	const developer_developer_separate_models_analyse = () => {
		console.log("analyse event : ", data)
		if (data === null) {
			return (
				message.warning(
					"Please make sure to insert a YouTube url")

			)
		}
		axios.post("http://127.0.0.1:5000/developer_separate_models/predict", {
			youtubeUrl: data,
		})
			.then(function (response) {
				console.log("separate_models")
				console.log(response);
				setDatapie(response?.data?.model_output)
				console.log(response?.data?.model_output)
				if (response?.data?.model_output !== "Video possess no valid comments for analyzing") {
					navigate("/results", {state: {data: response?.data}});

				} else {
					console.log("awooo")
					return (
						message.warning(
							"Cannot Analyze! Video possess no valid comments for analyzing")

					)
				}

			})
			.catch(function (error) {

				console.log(error);
			});
	}



	const text_separate_models_analyse = () => {
		console.log("analyse event : ", data)
		if (data === null) {
			return (
				message.warning(
					"Please make sure to insert a YouTube url")

			)
		}
		axios.post("http://127.0.0.1:5000/text_separate_models/predict", {
			text: data,
		})
			.then(function (response) {
				console.log("text_separate_models")
				console.log(response);
				setDatapie(response?.data?.model_output)
				console.log(response?.data?.model_output)
				if (response?.data?.model_output !== "Video possess no valid comments for analyzing") {
					navigate("/results", {state: {data: response?.data}});

				} else {
					console.log("awooo")
					return (
						message.warning(
							"Cannot Analyze! Video possess no valid comments for analyzing")

					)
				}

			})
			.catch(function (error) {

				console.log(error);
			});
	}


	return (
		<div className="main">

			<div className="header">
				{/*<img className="logo" src={logo} width="100" height="100"/>*/}
				<h1 className="logo-text" REFL-TEXT="EmoZenze">EmoZenze</h1>

				{/*<h1 className="wish">hi</h1>*/}
				<div className="user-lstm-card">
					<h2 className="model-name"></h2>
					<form className="youtube-link-form">
						<input type="text" className="user-youtube_link" name="youtube_link"
							   placeholder="Enter YouTube Link" onChange={(event) => {
							youtubeEvent(event)
						}} required/><br/><br/>

						<input type="button" className="user-analyze" value="Analyze"
							   onClick={user_model_analyse}/>

					</form>
				</div>
			</div>

			<div className="distraction">
				<div className="user-text-card">
					<h2 className="try">Try out manually !!!</h2><br/>
					<form className="text-form">
						<input type="text" className="text-input" name="text-input"
							   placeholder="Enter Sinhala texts with emojis" onChange={(event) => {
							textEvent(event)
						}} required/><br/><br/>

						<input type="button" className="user-analyze" value="Analyze"
							   onClick={text_separate_models_analyse}/>

					</form>
				</div>

			</div>

			<div className="developer-purpose">
				<h1 className="mode">Nerds Only Zone </h1>

				<div className="holders">
					<div className="holder1">
						<div className="card-holder-one">
							<div className="separate-lstm-card">
								<h2 className="model-name">Separate LSTM Models</h2>
								<form>
									<input type="text" className="youtube_link" name="youtube_link"
										   placeholder="Enter YouTube Link" onChange={(event) => {
										youtubeEvent(event)
									}} required/><br/><br/>
									<input type="button" className="analyze1" value="Analyze"
										   onClick={developer_developer_separate_models_analyse}/>
								</form>
							</div>
						</div>


					</div>

					<div className="holder2">


						<div className="card-holder-two">

							<div className="lstm-card">
								<h2 className="model-name">LSTM</h2>
								<form className="form">
									<input type="text" className="youtube_link" name="youtube_link"
										   placeholder="Enter YouTube Link" onChange={(event) => {
										youtubeEvent(event)
									}} required/><br/><br/>
									<input type="button" className="analyze" value="Analyze"
										   onClick={developer_lstm_model_analyse}/>
								</form>
							</div>
							<br/>
							<div className="kmeans-card">
								<h2 className="model-name"> KMeans</h2>
								<form className="form">
									<input type="text" className="youtube_link" name="youtube_link"
										   placeholder="Enter YouTube Link" onChange={(event) => {
										youtubeEvent(event)
									}} required/><br/><br/>
									<input type="button" className="analyze" value="Analyze"
										   onClick={developer_kmeans_model_analyse}/>
								</form>
							</div>
						</div>

						<div className="card-holder-three">
							<div className="lsvc-card">
								<h2 className="model-name">LSVC</h2>
								<form className="form">
									<input type="text" className="youtube_link" name="youtube_link"
										   placeholder="Enter YouTube Link" onChange={(event) => {
										youtubeEvent(event)
									}} required/><br/><br/>
									<input type="button" className="analyze" value="Analyze"
										   onClick={developer_lsvc_model_analyse}/>
								</form>
							</div>
							<br/>
							<div className="all-model-card">
								<h2 className="model-name">All Models</h2>
								<form className="form">
									<input type="text" className="youtube_link" name="youtube_link"
										   placeholder="Enter YouTube Link" onChange={(event) => {
										youtubeEvent(event)
									}} required/><br/><br/>
									<input type="button" className="analyze" value="Analyze"
										   onClick={developer_developer_all_models_analyse}/>
								</form>
							</div>
						</div>
					</div>
				</div>
			</div>


		</div>

	)
}


export default LandingPage
