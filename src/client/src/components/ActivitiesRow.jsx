import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import isActiveIcon from "../../assets/is_active.png";
import notActiveIcon from "../../assets/not_active.png";

export default function ActivitiesRow() {
	const [time, setTime] = useState([]);

	const API_URL = "https://p01--iis-api--q6nfcmmd42sk.code.run/";

	const { data, isLoading, error } = useQuery({
		queryKey: ["activities"],
		queryFn: async () => {
			const response = await fetch(`${API_URL}/activities`);
			console.log(response);
			if (!response.ok) {
				throw new Error("Network response was not ok");
			}

			const givenTime = new Date();
			for (let i = 1; i < 9; i++) {
				givenTime.setMinutes(Math.round(givenTime.getMinutes() / 60) * 60);
				const roundedTime = givenTime.toLocaleTimeString("en-US", {
					hour: "2-digit",
					minute: "2-digit",
					hour12: false,
				});
				givenTime.setHours(givenTime.getHours() + 1);
				setTime((time) => [...time, roundedTime]);
			}

			return response.json();
		},
	});

	const isActive = (activity) => {
		if (activity == 1) {
			return <img className="activityIcon" src={isActiveIcon} alt="Active" />;
		}
		return (
			<img className="activityIcon" src={notActiveIcon} alt="Not Active" />
		);
	};

	if (isLoading) return "Loading...";

	if (error) return "An error has occurred: " + error.message;

	console.log(data);

	return (
		<>
			<div className="activitiesContainer" style={{ backgroundColor: "white" }}>
				{data.predictions.map((pred, index) => (
					<div key={index} className="activityRow">
						<div style={{ fontSize: "22px", marginBottom: "5px" }}>
							{time[index]}
						</div>
						<div>{isActive(pred)}</div>
					</div>
				))}
			</div>
		</>
	);
}
