import { useQuery } from "@tanstack/react-query";

export default function Kudos() {
	const API_URL = "https://p01--iis-api--q6nfcmmd42sk.code.run/";

	const { data, isLoading, error } = useQuery({
		queryKey: ["kudos_predictions"],
		queryFn: async () => {
			const response = await fetch(`${API_URL}/kudos/5`);
			console.log(response);
			if (!response.ok) {
				throw new Error("Network response was not ok");
			}
			return response.json();
		},
	});

	if (isLoading) return "Loading...";

	if (error) return "An error has occurred: " + error.message;

	console.log(data);

	return (
		<>
			<div>Kudos</div>
		</>
	);
}
