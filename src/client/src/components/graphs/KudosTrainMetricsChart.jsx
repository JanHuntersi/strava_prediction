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

export default function KudosTrainMetricsChart({ metrics }) {
	// Extract the data from the metrics array
	const labels = metrics.map((_, index) => index + 1); // X-axis labels (index)
	const meanAbsoluteError = metrics.map(
		(metric) => metric.training_mean_absolute_error
	);
	const meanSquaredError = metrics.map(
		(metric) => metric.training_mean_squared_error
	);
	const r2Score = metrics.map((metric) => metric.training_r2_score);
	const rootMeanSquaredError = metrics.map(
		(metric) => metric.training_root_mean_squared_error
	);
	const trainingScore = metrics.map((metric) => metric.training_score);

	const data = {
		labels,
		datasets: [
			{
				label: "Mean Absolute Error",
				data: meanAbsoluteError,
				borderColor: "rgba(255, 99, 132, 1)",
				backgroundColor: "rgba(255, 99, 132, 0.2)",
				fill: false,
			},
			{
				label: "Mean Squared Error",
				data: meanSquaredError,
				borderColor: "rgba(54, 162, 235, 1)",
				backgroundColor: "rgba(54, 162, 235, 0.2)",
				fill: false,
			},
			{
				label: "R2 Score",
				data: r2Score,
				borderColor: "rgba(75, 192, 192, 1)",
				backgroundColor: "rgba(75, 192, 192, 0.2)",
				fill: false,
			},
			{
				label: "Root Mean Squared Error",
				data: rootMeanSquaredError,
				borderColor: "rgba(153, 102, 255, 1)",
				backgroundColor: "rgba(153, 102, 255, 0.2)",
				fill: false,
			},
			{
				label: "Training Score",
				data: trainingScore,
				borderColor: "rgba(255, 159, 64, 1)",
				backgroundColor: "rgba(255, 159, 64, 0.2)",
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
				text: "Training Metrics Over Time",
			},
		},
	};

	return (
		<div style={{ height: `350px`, width: `500px` }}>
			<Line data={data} height={"350px"} width={"500px"} options={options} />
		</div>
	);
}
