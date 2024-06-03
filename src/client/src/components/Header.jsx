import React from "react";
import adminIcon from "../../assets/admin.png";
import { useNavigate } from "react-router-dom";

const Header = () => {
	const navigate = useNavigate();
	return (
		<div
			style={{
				backgroundColor: "var(--primary-color)",
				display: "flex",
				justifyContent: "center",
				alignItems: "center",
				textAlign: "center",
				height: "8%",
				color: "white",
				width: "100%",
				fontSize: "1.5rem",
				fontFamily: "var(--font-family)",
				position: "relative", // Relative positioning for the parent
			}}
		>
			<div
				onClick={() => {
					navigate("/");
				}}
				style={{
					position: "absolute",
					left: "50%",
					transform: "translateX(-50%)",
					cursor: "pointer",
				}}
			>
				Strava Predictions
			</div>
			<div
				onClick={() => {
					navigate("/admin");
				}}
				style={{
					position: "absolute",
					right: "40px",
					display: "flex",
					cursor: "pointer",
				}}
			>
				<img
					src={adminIcon}
					alt="admin"
					style={{
						width: "30px",
						height: "30px",
						color: "white",
						marginRight: "5px",
					}}
				/>
				Admin
			</div>
		</div>
	);
};

export default Header;
