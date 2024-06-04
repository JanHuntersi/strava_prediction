export default function IframeComponent() {
	return (
		<div>
			<iframe
				title="iframe"
				src="https://stravapredict.netlify.app/"
				width="100%"
				height="600px"
				allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
				allowFullScreen
			></iframe>
		</div>
	);
}
