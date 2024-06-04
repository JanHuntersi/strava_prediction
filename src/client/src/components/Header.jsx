import React from "react";
import adminIcon from "../../assets/admin.png";
import dataIcon from "../../assets/data.png";
import homepageIcon from "../../assets/homepage.png";
import activityIcon from "../../assets/activity.png";
import reportIcon from "../../assets/report.png";
import { useNavigate } from "react-router-dom";

const Header = () => {
	const navigate = useNavigate();
	return (
		<div
			style={{
				backgroundColor: "var(--primary-color)",
				display: "flex",
				flexDirection: "column",
				textAlign: "center",
				width: "20px",
				height: "100vh",
				color: "white",
				width: "275px",
				fontSize: "1.3rem",
				fontFamily: "var(--font-family)",
				position: "relative", // Relative positioning for the parent
				paddingTop: "25px",
			}}
		>
			<div
				onClick={() => {
					navigate("/");
				}}
				className="logoContainer"
			>
				<img src={activityIcon} alt="activity" className="logo" />
				ActivityChecker
			</div>
			<div
				style={{
					width: "100%",
					height: "1px",
					backgroundColor: "white",
					marginBlockStart: "1em",
				}}
			></div>
			<div
				style={{
					display: "flex",
					alignItems: "center",
					justifyContent: "flex-start",
				}}
			>
				<div
					style={{
						display: "flex",
						flexDirection: "column",
						alignItems: "flex-start",
						marginBlockStart: "2em",
						marginInlineStart: "1em",
						gap: "25px",
						fontSize: "1rem",
					}}
				>
					<div
						onClick={() => {
							navigate("/");
						}}
						className="navItems"
					>
						<img src={homepageIcon} alt="data" className="iconStyle" />
						Home
					</div>

					<div
						onClick={() => {
							navigate("/reports");
						}}
						className="navItems"
					>
						<img src={reportIcon} alt="reports" className="iconStyle" />
						Reports
					</div>

					<div
						onClick={() => {
							navigate("/data");
						}}
						className="navItems"
					>
						<img src={dataIcon} alt="data" className="iconStyle" />
						Data
					</div>

					<div
						onClick={() => {
							navigate("/admin");
						}}
						className="navItems"
					>
						<img src={adminIcon} alt="admin" className="iconStyle" />
						Admin panel
					</div>
				</div>
			</div>
		</div>
	);
};

export default Header;
