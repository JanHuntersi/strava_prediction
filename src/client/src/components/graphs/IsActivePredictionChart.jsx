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

export default function IsActivePredictionChart({
	metrics,
	height = 400,
	width = 600,
}) {
	// Extract the data from the metrics array
	const labels = metrics.map((_, index) => index + 1); // X-axis labels (index)
	const accuracy = metrics.map((metric) => metric.accuracy);
	const balancedAccuracy = metrics.map((metric) => metric.balanced_accuracy);
	const cohenKappa = metrics.map((metric) => metric.cohen_kappa);
	const mcc = metrics.map((metric) => metric.mcc);

	const data = {
		labels,
		datasets: [
			{
				label: "Accuracy",
				data: accuracy,
				borderColor: "rgba(75, 192, 192, 1)",
				backgroundColor: "rgba(75, 192, 192, 0.2)",
				fill: false,
			},
			{
				label: "Balanced Accuracy",
				data: balancedAccuracy,
				borderColor: "rgba(255, 99, 132, 1)",
				backgroundColor: "rgba(255, 99, 132, 0.2)",
				fill: false,
			},
			{
				label: "Cohen Kappa",
				data: cohenKappa,
				borderColor: "rgba(54, 162, 235, 1)",
				backgroundColor: "rgba(54, 162, 235, 0.2)",
				fill: false,
			},
			{
				label: "MCC",
				data: mcc,
				borderColor: "rgba(153, 102, 255, 1)",
				backgroundColor: "rgba(153, 102, 255, 0.2)",
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
				text: "Is Active Prediction Metrics Over Time",
			},
		},
	};

	return (
		<div style={{ height: `${height}px`, width: `${width}px` }}>
			<Line data={data} options={options} height={height} width={width} />
		</div>
	);
}
