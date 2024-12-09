import React, { Fragment } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Loader2 } from "lucide-react";
import useFetch from "@/hooks/useFetch";
import { Categories, PricingStrategy, SalesPrediction } from "@/types/api";
import { useEffect, useRef, FormEvent } from "react";

function SalesAnalysisPerformancePage() {
	const [selectedPricingCategory, setSelectedPricingCategory] = React.useState<
		string | null
	>(null);
	const formRef = useRef<HTMLFormElement>(null);
	const baseAPI = "http://localhost:8000";

	const {
		data: categories,
		error: categoriesError,
		loading: categoriesLoading,
	} = useFetch<Categories>(`${baseAPI}/categories`);

	const { data: pricingAnalysis, loading: pricingAnalysisLoading } =
		useFetch<PricingStrategy>(
			`${baseAPI}/analytics/revenue?category=${selectedPricingCategory}`
		);

	const {
		data: salesPrediction,
		loading: salesPredictionLoading,
		executeRequest,
	} = useFetch<SalesPrediction>(`${baseAPI}/predict/sales`, "POST");

	useEffect(() => {
		if (salesPrediction) {
			formRef.current?.reset();
		}
	}, [salesPrediction]);

	const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
		e.preventDefault();
		const formData = new FormData(e.currentTarget);

		await executeRequest({
			price: parseFloat(formData.get("product-price") as string),
			discount_percentage: parseInt(
				formData.get("discount-percentage") as string
			),
			rating: parseInt(formData.get("rating") as string),
			num_reviews: parseInt(formData.get("total-reviews") as string),
			category: formData.get("categories") as string,
		});
	};

	if (categoriesLoading) {
		return (
			<div className="flex mx-auto h-screen items-center justify-center">
				<Loader2 className="h-8 w-8 animate-spin text-blue-500" />
			</div>
		);
	}

	if (categoriesError) {
		return (
			<div className="flex mx-auto h-screen items-center justify-center text-red-500">
				Error loading categories
			</div>
		);
	}

	const defaultPricingAnalysis = [
		{
			Category: "-",
			TotalRevenue: "-",
			TotalSales: "-",
			AveragePrice: "-",
			ProductCount: "-",
			AvgDiscountPercentage: "-",
		},
	];

	const defaultSalesPrediction = [
		{
			predicted_sales: "-",
			confidence_score: "-",
		},
	];

	const displayPricingAnalysis = pricingAnalysisLoading
		? defaultPricingAnalysis
		: pricingAnalysis || defaultPricingAnalysis;

	const displaySalesPrediction = salesPredictionLoading
		? defaultSalesPrediction
		: salesPrediction || defaultSalesPrediction;

	return (
		<main className="min-h-screen">
			<div className="max-w-7xl space-y-8">
				<p className="text-lg font-semibold">Sales Analysis Performance</p>

				<Card className="overflow-hidden shadow-lg">
					<CardHeader>
						<CardTitle className="text-xl text-gray-800">
							Pricing Analysis
						</CardTitle>
					</CardHeader>
					<CardContent>
						<select
							onChange={(e) => setSelectedPricingCategory(e.target.value)}
							className="w-full p-2 border rounded-lg bg-white shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
							value={selectedPricingCategory || ""}
						>
							<option value="">Select a category</option>
							{categories?.categories?.map((category, index) => (
								<option key={index} value={category}>
									{category}
								</option>
							))}
						</select>

						{pricingAnalysisLoading ? (
							<div className="flex my-10 items-center justify-center">
								<Loader2 className="h-8 w-8 animate-spin text-blue-500" />
							</div>
						) : (
							<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mt-6">
								{displayPricingAnalysis.map((metric, index) => (
									<Fragment key={index}>
										<Card className="bg-gradient-to-br from-blue-50 to-blue-100 shadow-sm">
											<CardContent className="p-4">
												<p className="text-lg font-semibold text-blue-900">
													Average Discount
												</p>
												<p className="text-2xl font-bold text-blue-700 mt-2">
													{metric.AvgDiscountPercentage}
												</p>
											</CardContent>
										</Card>

										<Card className="bg-gradient-to-br from-green-50 to-green-100 shadow-sm">
											<CardContent className="p-4">
												<p className="text-lg font-semibold text-green-900">
													Total Revenue
												</p>
												<p className="text-2xl font-bold text-green-700 mt-2">
													${metric.TotalRevenue}
												</p>
											</CardContent>
										</Card>

										<Card className="bg-gradient-to-br from-purple-50 to-purple-100 shadow-sm">
											<CardContent className="p-4">
												<p className="text-lg font-semibold text-purple-900">
													Total Sales
												</p>
												<p className="text-2xl font-bold text-purple-700 mt-2">
													{metric.TotalSales}
												</p>
											</CardContent>
										</Card>

										<Card className="bg-gradient-to-br from-yellow-50 to-yellow-100 shadow-sm">
											<CardContent className="p-4">
												<p className="text-lg font-semibold text-yellow-900">
													Average Price
												</p>
												<p className="text-2xl font-bold text-yellow-700 mt-2">
													${metric.AveragePrice}
												</p>
											</CardContent>
										</Card>

										<Card className="bg-gradient-to-br from-indigo-50 to-indigo-100 shadow-sm">
											<CardContent className="p-4">
												<p className="text-lg font-semibold text-indigo-900">
													Product Count
												</p>
												<p className="text-2xl font-bold text-indigo-700 mt-2">
													{metric.ProductCount}
												</p>
											</CardContent>
										</Card>
									</Fragment>
								))}
							</div>
						)}
					</CardContent>
				</Card>

				<Card className="overflow-hidden shadow-lg">
					<CardHeader>
						<CardTitle className="text-xl text-gray-800">
							Sales Prediction
						</CardTitle>
					</CardHeader>
					<CardContent>
						<div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
							<form ref={formRef} onSubmit={handleSubmit} className="space-y-4">
								<div className="space-y-4">
									<div>
										<Label className="text-gray-800">Product Price</Label>
										<Input
											id="product-price"
											name="product-price"
											type="number"
											step="0.01"
											placeholder="Enter product price"
											className="w-full px-3 py-2 rounded-md text-slate-900 bg-white"
											required
										/>
									</div>

									<div>
										<Label className="text-gray-800">Discount Percentage</Label>
										<Input
											id="discount-percentage"
											name="discount-percentage"
											type="number"
											min={0}
											max={100}
											placeholder="Enter discount (0-100)"
											className="w-full px-3 py-2 rounded-md text-slate-900 bg-white"
											required
										/>
									</div>

									<div>
										<Label className="text-gray-800">Rating</Label>
										<Input
											id="rating"
											name="rating"
											type="number"
											min={1}
											max={5}
											placeholder="Enter rating (1-5)"
											className="w-full px-3 py-2 rounded-md text-slate-900 bg-white"
											required
										/>
									</div>

									<div>
										<Label className="text-gray-800">Total Reviews</Label>
										<Input
											id="total-reviews"
											name="total-reviews"
											type="number"
											placeholder="Enter total reviews"
											className="w-full px-3 py-2 rounded-md text-slate-900 bg-white"
											required
										/>
									</div>

									<div>
										<Label className="text-gray-800">Category</Label>
										<select
											name="categories"
											className="w-full p-2 border rounded-lg bg-white shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
											required
										>
											<option value="">Select a category</option>
											{categories?.categories?.map((category, index) => (
												<option key={index} value={category}>
													{category}
												</option>
											))}
										</select>
									</div>
								</div>

								<Button
									type="submit"
									className="w-full bg-blue-600 hover:bg-blue-700 text-white"
								>
									Predict Sales
								</Button>
							</form>

							{displaySalesPrediction && (
								<div className="rounded-lg space-y-2 text-gray-800">
									<p className="font-semibold text-xl">Prediction Results</p>

									<Card className="bg-gradient-to-br from-green-50 to-green-100 shadow-sm">
										<CardContent className="p-4 text-green-900">
											<p className="text-lg font-semibold ">Predicted Sales</p>
											<p className="text-2xl font-bold text-green-700 mt-2">
												$
												{Array.isArray(displaySalesPrediction)
													? displaySalesPrediction[0].predicted_sales
													: displaySalesPrediction.predicted_sales}
											</p>
										</CardContent>
									</Card>

									<Card className="bg-gradient-to-br from-blue-50 to-blue-100 shadow-sm">
										<CardContent className="p-4">
											<p className="text-lg font-semibold  text-blue-900">
												Confidence Score
											</p>
											<p className="text-2xl font-bold text-blue-700 mt-2">
												{Array.isArray(displaySalesPrediction)
													? displaySalesPrediction[0].confidence_score
													: displaySalesPrediction.confidence_score}
												%
											</p>
										</CardContent>
									</Card>
								</div>
							)}
						</div>
					</CardContent>
				</Card>
			</div>
		</main>
	);
}

export default SalesAnalysisPerformancePage;
