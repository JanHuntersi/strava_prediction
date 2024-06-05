export default function IframeComponent({ web_url }) {
	return (
		<div>
			<iframe
				title="iframe"
				src={web_url}
				width="100%"
				height="600px"
				allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
				allowFullScreen
			></iframe>
		</div>
	);
}
