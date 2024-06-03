import { useState } from "react";
import "./App.css";
import Header from "./components/Header";
import { Routes, Route } from "react-router-dom";
import Home from "./components/Home";
import Admin from "./components/Admin";

function App() {
	return (
		<>
			<div className="app">
				<Header />
				<div
					style={{
						display: "flex",
						width: "100%",
						height: "100%",
					}}
				>
					<Routes>
						<Route path="/" element={<Home />} />
						<Route path="/admin" element={<Admin />} />
					</Routes>
				</div>
			</div>
		</>
	);
}

export default App;
