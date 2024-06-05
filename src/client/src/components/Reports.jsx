import React from "react";
import IframeComponent from "./IframeComponent";
import HeaderPageName from "./HeaderPageName";

export default function Reports() {
	return (
		<div className="pageContainer">
			<HeaderPageName pageName="Reports" />
			<div className="pageContainer">
				<IframeComponent web_url="https://stravapredict.netlify.app/" />
				<IframeComponent web_url="https://evidently-is-active.netlify.app/" />
			</div>
		</div>
	);
}
