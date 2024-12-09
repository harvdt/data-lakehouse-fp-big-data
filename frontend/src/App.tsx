import { BrowserRouter, Route, Routes } from "react-router-dom";

import Layout from "./Layout";
import SalesAnalysisPerformance from "./pages/SalesAnalysisPerformance";
import PricingStrategyAnalysis from "./pages/PricingStrategyAnalysis";
import CustomerSatisfactionMetrics from "./pages/CustomerSatisfactionMetrics";

function App() {
	return (
		<BrowserRouter>
			<Layout>
				<Routes>
					<Route path="/" element={<SalesAnalysisPerformance />} />
					<Route
						path="/pricing-strategy"
						element={<PricingStrategyAnalysis />}
					/>
					<Route
						path="/customer-satisfaction"
						element={<CustomerSatisfactionMetrics />}
					/>
				</Routes>
			</Layout>
		</BrowserRouter>
	);
}

export default App;
