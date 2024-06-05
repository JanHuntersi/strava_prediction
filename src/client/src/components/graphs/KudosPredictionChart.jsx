import React from 'react';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';

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

export default function KudosPredictionChart({ metrics, height = 400, width = 600 }) {
    // Extract the data from the metrics array
    const labels = metrics.map((_, index) => index + 1); // X-axis labels (index)
    const evs = metrics.map(metric => metric.evs);
    const mae = metrics.map(metric => metric.mae);
    const mse = metrics.map(metric => metric.mse);

    const data = {
        labels,
        datasets: [
            {
                label: 'EVS',
                data: evs,
                borderColor: 'rgba(75, 192, 192, 1)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                fill: false,
            },
            {
                label: 'MAE',
                data: mae,
                borderColor: 'rgba(255, 99, 132, 1)',
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                fill: false,
            },
            {
                label: 'MSE',
                data: mse,
                borderColor: 'rgba(54, 162, 235, 1)',
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                fill: false,
            },
        ],
    };

    const options = {
        responsive: true,
        plugins: {
            legend: {
                position: 'top',
            },
            title: {
                display: true,
                text: 'Kudos Prediction Metrics Over Time',
            },
        },
    };

    return (
        <div style={{ height: `${height}px`, width: `${width}px` }}>
            <Line data={data} options={options} height={height} width={width} />
        </div>
    );
};


