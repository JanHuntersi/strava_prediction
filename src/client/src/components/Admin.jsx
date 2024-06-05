import HeaderPageName from "./HeaderPageName";
import IframeComponent from "./IframeComponent";
import KudosTrainMetricsChart from "./graphs/KudosTrainMetricsChart";
import IsActiveTrainMetricsChart from "./graphs/IsActiveTrainMetricsChart";
import { useQuery } from "@tanstack/react-query";
import KudosPredictionChart from "./graphs/KudosPredictionChart";
import IsActivePredictionChart from "./graphs/IsActivePredictionChart";

export default function Admin() {
	const { data, isLoading, error } = useQuery({
		queryKey: ["metrics"],
		queryFn: async () => {
			const response = await fetch(
				"https://p01--iis-api--q6nfcmmd42sk.code.run/metrics"
			);
			console.log(response);
			if (!response.ok) {
				throw new Error("Network response was not ok");
			}
			return response.json();
		},
		retry: 4,
	});

	return (
		<>
			<div className="pageContainer">
				<HeaderPageName pageName="Admin" />
				<div
					style={{
						paddingLeft: "1em",
						paddingTop: "1em",
						boxSizing: "border-box",
					}}
					className="pageContainer"
				>
					<div
						style={{
							fontSize: 22,
							font: "Poppins",
							fontWeight: 600,
						}}
					>
						Model training metrics
					</div>
					<div>
						{data && (
							<div style={{ display: "flex", flexDirection: "row" }}>
								<KudosTrainMetricsChart metrics={data.kudos_train_metrics} />
								<IsActiveTrainMetricsChart
									metrics={data.is_active_train_metrics}
								/>
							</div>
						)}
					</div>
					<div
						style={{
							fontSize: 22,
							font: "Poppins",
							fontWeight: 600,
							marginTop: "20px",
						}}
					>
						Data Evaluation
					</div>
					<div>
						{data && (
							<div style={{ display: "flex", flexDirection: "row" }}>
								<KudosPredictionChart metrics={data.kudos_prediction} />
								<IsActivePredictionChart metrics={data.is_active_prediction} />
							</div>
						)}
					</div>
				</div>
			</div>
		</>
	);
}
