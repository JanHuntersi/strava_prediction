import React from "react";
import HeaderPageName from "./HeaderPageName";
import ActivitiesRow from "./ActivitiesRow";
import Kudos from "./Kudos";

const Home = () => {
	return (
		<div className="pageContainer">
			<HeaderPageName pageName="Home" />
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
					Jure's activity prediction
				</div>
				<div className="paper-effect" style={{ marginBlockStart: "20px" }}>
					<ActivitiesRow />
				</div>
				<div
					style={{
						marginBlockStart: "20px",
						fontSize: 22,
						font: "Poppins",
						fontWeight: 600,
					}}
				>
					Last 5 kudos predictions
				</div>
				<div className="paper-effect" style={{ marginBlockStart: "20px" }}>
					<Kudos />
				</div>
			</div>
		</div>
	);
};

export default Home;
