import { useState } from "react";
import "./App.css";
import Header from "./components/Header";
import { Routes, Route } from "react-router-dom";
import Home from "./components/Home";
import Admin from "./components/Admin";
import Reports from "./components/Reports";

function App() {
	return (
		<>
			<div className="app">
				<div style={{ display: "flex", flexDirection: "row" }}>
					<Header />
					<div
						style={{
							backgroundColor: "white",
							display: "flex",
							width: "100%",
							height: "100vh",
							padding: "2em",
						}}
					>
						<Routes>
							<Route path="/" element={<Home />} />
							<Route path="/admin" element={<Admin />} />
							<Route path="/reports" element={<Reports />} />
						</Routes>
					</div>
				</div>
			</div>
		</>
	);
}

export default App;
