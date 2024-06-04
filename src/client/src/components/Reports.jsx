import React from "react";
import IframeComponent from "./IframeComponent";
import HeaderPageName from "./HeaderPageName";

export default function Reports() {
	return (
		<div className="pageContainer">
			<HeaderPageName pageName="Reports" />
			<div className="pageContainer">
				<IframeComponent />
			</div>
		</div>
	);
}
