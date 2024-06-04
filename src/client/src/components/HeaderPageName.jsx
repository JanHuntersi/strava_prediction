export default function HeaderPageName({ pageName }) {
	return (
		<div style={{ display: "flex", flexDirection: "column", fontWeight: 600 }}>
			<div style={{ fontSize: 28, font: "Poppins" }}>{pageName}</div>
			<div
				style={{
					marginTop: "1em",
					height: "1px",
					backgroundColor: "var(--primary-color)",
				}}
			></div>
		</div>
	);
}
