import React from 'react';
import {useLocation} from "react-router-dom";
import {Pie} from "@ant-design/plots";
import './results_page.css'


const ResultsPage = () => {
	const location = useLocation();
	// const [data, setData] = useState(null)
	var data = null;
	var data1 = null;
	var data2 = null;
	var data3 = null;
	var data4 = null;

	var config = null;
	var config1 = null;
	var config2 = null;
	var config3 = null;
	var config4 = null;

	const customColors = [
		"#4A4E69",
		"#713b71",
		"#1D1E2C",
		"#373c70",
		"#144636",
		"#585e2f",
		"#202ebb",
		"#5b083e",
		"#da0948",
		"#238f05",
		"#4e0375",
		"#3d3b3b",
		"#05445b",
		"#9d3f09",
		"#d39416"
	]

	if (location?.state?.data?.key === "all_models") {
		data1 = (location?.state?.data?.model_output[0])
		data2 = (location?.state?.data?.model_output[1])
		data3 = (location?.state?.data?.model_output[2])
		data4 = (location?.state?.data?.model_output[3])


		config1 = {
			appendPadding: 10,
			data: data1,
			angleField: "value",
			colorField: "type",
			color: customColors,
			radius: 0.8,

			label: {
				type: "inner",
				offset: "4%",

				content: "{percentage}",
				formatter: "{name} {value}",
			},
			legend: {
				style: {
					fontSize: 24,
					scrollable: true,

				},
				layout: 'vertical',
				position: 'right',
				height: 'auto',
			},
			interactions: [
				{
					type: "pie-legend-active",
				},
				{
					type: "element-active",
				},
			],
		};

		config2 = {
			appendPadding: 10,
			data: data2,
			angleField: "value",
			colorField: "type",
			color: customColors,
			radius: 0.8,

			label: {
				type: "inner",
				offset: "4%",


				content: "{percentage}",
				formatter: "{name} {value}",
			},
			legend: {
				style: {
					fontSize: 24,
				},
				layout: 'vertical',
				position: 'right',
				height: '400px',
			},
			interactions: [
				{
					type: "pie-legend-active",
				},
				{
					type: "element-active",
				},
			],
		};

		config3 = {
			appendPadding: 10,
			data: data3,
			angleField: "value",
			colorField: "type",
			color: customColors,
			radius: 0.8,

			label: {
				style: {
					height: 'auto',
					scrollable: true,

				},
				type: "inner",
				offset: "4%",

				content: "{percentage}",
				formatter: "{name} {value}",
			},
			legend: {
				style: {
					fontSize: 24,
				},
				layout: 'vertical',
				position: 'right',
				height: 'auto',
			},
			interactions: [
				{
					type: "pie-legend-active",
				},
				{
					type: "element-active",
				},
			],
		};

		config4 = {
			appendPadding: 10,
			data: data4,
			angleField: "value",
			colorField: "type",
			color: customColors,
			radius: 0.8,

			label: {
				style: {
					height: 'auto',
					scrollable: true,

				},
				type: "inner",
				offset: "4%",

				content: "{percentage}",
				formatter: "{name} {value}",
			},
			legend: {
				style: {
					fontSize: 24,
				},
				layout: 'vertical',
				position: 'right',
				height: 'auto',
			},
			interactions: [
				{
					type: "pie-legend-active",
				},
				{
					type: "element-active",
				},
			],
		};

	} else {
		const customColors = [
			"#4A4E69",
			"#713b71",
			"#1D1E2C",
			"#373c70",
			"#006734",
			"#585e2f",
			"#202ebb",
			"#5b083e",
			"#da0948",
			"#238f05",
			"#4e0375",
			"#3d3b3b",
			"#05445b",
			"#9d3f09",
			"#d39416"
		]
		data = (location?.state?.data?.model_output)
		config = {
			appendPadding: 10,
			data: data,
			angleField: "value",
			colorField: "type",
			color: customColors,
			radius: 0.8,

			label: {
				type: "inner",
				offset: "4%",

				content: "{percentage}",
				formatter: "{name} {value}",
			},
			legend: {
				style: {
					fontSize: 24,
					scrollable: true,

				},
				layout: 'vertical',
				position: 'right',
				height: 'auto',
			},
			interactions: [
				{
					type: "pie-legend-active",
				},
				{
					type: "element-active",
				},
			],
		};
	}


	return (
		<div className="result-main">
			<h1 className="title">Predictions Visualization</h1>

			<div className="chart">
				{location?.state?.data?.key === "all_models" ?
					<>
						<div className="chart1">
							<p className="model-name">Separate Models Predictions</p>
							<Pie className="pie-style" {...config1} />
						</div>

						<div className="chart2">
							<p className="model-name">LSTM Predictions</p>
							<Pie className="pie-style" {...config2} />
						</div>

						<div className="chart3">
							<p className="model-name">LSVC Predictions</p>
							<Pie className="pie-style" {...config3} />
						</div>

						<div className="chart4">
							<p className="model-name">KMeans Predictions</p>
							<Pie className="pie-style" {...config4} />
						</div>

					</>
					:
					<>
						<div className="chart-default">
							<p className="model-name">Predictions</p>
							<Pie className="pie-style" {...config} />
						</div>

					</>
				}

			</div>


		</div>
	)
}
export default ResultsPage
