import HeaderPageName from "./HeaderPageName";
import JureIcon from "../../assets/jure.jpg";

export default function Home() {
	return (
		<div className="pageContainer">
			<HeaderPageName pageName="What is ActivityChecker?" />
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
						fontSize: 28,
						font: "Poppins",
						fontWeight: 600,
					}}
				>
					ActivityChecker is a tool that predicts the next activity of a user
					based on the user's previous activities.
				</div>
				<div
					style={{
						marginBlockStart: "20px",
						fontSize: 20,
						fontWeight: 400,
						font: "Poppins",
					}}
				>
					ActivityChecker uses a machine learning model to predict the next
					activity of a user based on the user's previous activities. The model
					uses the user's previous activities to predict the next activity. The
					model takes into account the user's previous activities, the time of
					day, and the day of the week to make the prediction. The model is
					trained on a dataset of user activities and their corresponding times.
					The model is then used to predict the next activity of a user based on
					the user's previous activities.
				</div>
				<div
					style={{
						marginBlockStart: "20px",
						fontSize: 28,
						font: "Poppins",
						fontWeight: 600,
					}}
				>
					How does ActivityChecker work?
				</div>
				<div
					style={{
						marginBlockStart: "20px",
						fontSize: 20,
						fontWeight: 400,
						font: "Poppins",
					}}
				>
					ActivityChecker works by using a machine learning model to predict the
					next activity of a user based on the user's previous activities. The
					model takes into account the user's previous activities, the time of
					day, and the day of the week to make the prediction. The model is
					trained on a dataset of user activities and their corresponding times.
					The model is then used to predict the next activity of a user based on
					the user's previous activities.
					<br /> <br />
					ActivityChecker can be used to predict the number of kudos a user will
					receive for an activity. The model uses the user's previous activities
					to predict the number of kudos the user will receive for an activity.
					The model takes into account the user's previous activities, the time,
					past kudos, number of achievements, and the day of the week to make
					the prediction. The model is trained on a dataset of user activities
					and their corresponding kudos. The model is then used to predict the
					number of kudos a user will receive for an activity based on the
					user's previous activities.
				</div>

				<div
					style={{
						marginBlockStart: "20px",
						fontSize: 28,
						font: "Poppins",
						fontWeight: 600,
					}}
				>
					Who is Jure and why do we collect his data?
				</div>
				<div
					style={{
						display: "flex",
						alignItems: "flex-start",
						marginBlockStart: "20px",
					}}
				>
					<img
						src={JureIcon}
						alt="Jure"
						style={{ height: "180px", marginRight: "20px" }}
					/>
					<div style={{ fontSize: 20, fontWeight: 400, font: "Poppins" }}>
						Jure is a user and a friend :) <br /> who has agreed to share his
						activity data with us. We use Jure's activity data to train our
						machine learning model. Jure's activity data is used to train the
						model to predict the next activity of a user based on the user's
						previous activities. Jure's activity data is also used to train the
						model to predict the number of kudos a user will receive for an
						activity based on the user's previous activities.
					</div>
				</div>
			</div>
		</div>
	);
}
