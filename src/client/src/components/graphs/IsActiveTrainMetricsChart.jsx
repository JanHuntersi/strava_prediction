import React from "react";
import { Line } from "react-chartjs-2";
import {
	Chart as ChartJS,
	CategoryScale,
	LinearScale,
	PointElement,
	LineElement,
	Title,
	Tooltip,
	Legend,
} from "chart.js";

// Register Chart.js components
ChartJS.register(
	CategoryScale,
	LinearScale,
	PointElement,
	LineElement,
	Title,
	Tooltip,
	Legend
);

export default function IsActiveTrainMetricsChart({ metrics }) {
	// Extract the data from the metrics array
	const labels = metrics.map((_, index) => index + 1); // X-axis labels (index)
	const accuracy = metrics.map((metric) => metric.accuracy);
	const loss = metrics.map((metric) => metric.loss);
	const valAccuracy = metrics.map((metric) => metric.val_accuracy);
	const valLoss = metrics.map((metric) => metric.val_loss);

	const data = {
		labels,
		datasets: [
			{
				label: "Training Accuracy",
				data: accuracy,
				borderColor: "rgba(75, 192, 192, 1)",
				backgroundColor: "rgba(75, 192, 192, 0.2)",
				fill: false,
			},
			{
				label: "Training Loss",
				data: loss,
				borderColor: "rgba(255, 99, 132, 1)",
				backgroundColor: "rgba(255, 99, 132, 0.2)",
				fill: false,
			},
			{
				label: "Validation Accuracy",
				data: valAccuracy,
				borderColor: "rgba(54, 162, 235, 1)",
				backgroundColor: "rgba(54, 162, 235, 0.2)",
				fill: false,
			},
			{
				label: "Validation Loss",
				data: valLoss,
				borderColor: "rgba(255, 206, 86, 1)",
				backgroundColor: "rgba(255, 206, 86, 0.2)",
				fill: false,
			},
		],
	};

	const options = {
		responsive: true,
		plugins: {
			legend: {
				position: "top",
			},
			title: {
				display: true,
				text: "Training and Validation Metrics Over Time",
			},
		},
	};

	return (
		<div style={{ height: `350px`, width: `600px` }}>
			<Line data={data} height={"350px"} width={"600px"} options={options} />
		</div>
	);
}
