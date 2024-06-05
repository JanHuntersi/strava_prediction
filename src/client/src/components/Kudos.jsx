import { useQuery } from "@tanstack/react-query";

export default function Kudos() {
	const { data, isLoading, error } = useQuery({
		queryKey: ["kudos_predictions"],
		queryFn: async () => {
			const response = await fetch(`http://127.0.0.1:5000/kudos/4`);
			console.log(response);
			if (!response.ok) {
				throw new Error("Network response was not ok");
			}
			return response.json();
		},
	});

	if (isLoading) return "Loading...";

	if (error) return "An error has occurred: " + error.message;

	console.log(data);

	return (
		<>
			<div className="kudosRow">
				{data
					.map((pred, index) => (
						<div key={index}>
							<div
								style={{
									fontSize: "22px",
									marginBottom: "5px",
									backgroundColor: "#fc4c02",
									color: "white",
									padding: "5px",
									borderRadius: "10px",
								}}
							>
								<div style={{ fontSize: "28px", fontWeight: 700 }}>
									{pred.activity_name}
								</div>
								<div style={{ fontSize: "20px" }}>
									Type: {pred.activity_type}
								</div>
								<div style={{ fontSize: "20px" }}>{pred.date}</div>
								<div style={{ fontSize: "20px" }}>
									Prediction: {pred.kudos_prediction}
								</div>
								<div style={{ fontSize: "20px" }}>
									Actual Value: {pred.actual_value}
								</div>
							</div>
						</div>
					))
					.reverse()}
			</div>
		</>
	);
}
