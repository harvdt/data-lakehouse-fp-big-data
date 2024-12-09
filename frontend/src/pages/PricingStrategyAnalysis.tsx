import { Fragment, useState } from "react";
import { Loader2 } from "lucide-react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import useFetch from "@/hooks/useFetch";
import { Categories, DiscountAnalysis, PriceRatingRelation } from "@/types/api";

const PricingStrategyAnalysisPage = () => {
	const [selectedDiscountCategory, setSelectedDiscountCategory] = useState<
		string | null
	>(null);
	const [selectedPriceRatingCategory, setSelectedPriceRatingCategory] =
		useState<string | null>(null);

	const baseAPI = "http://localhost:8000";

	const {
		data: categories,
		error: categoriesError,
		loading: categoriesLoading,
	} = useFetch<Categories>(`${baseAPI}/categories`);

	const { data: discountAnalysis, loading: discountLoading } =
		useFetch<DiscountAnalysis>(
			`${baseAPI}/analytics/discount?category=${selectedDiscountCategory}`
		);

	const { data: priceRatingRelation, loading: priceRatingLoading } =
		useFetch<PriceRatingRelation>(
			`${baseAPI}/analytics/price-rating?category=${selectedPriceRatingCategory}`
		);

	const defaultDiscountMetrics = [
		{
			AvgDiscountPercentage: "-",
			TotalDiscountedRevenue: "-",
			PotentialRevenue: "-",
			AvgSales: "-",
			DiscountROI: "-",
			RevenueLossFromDiscount: "-",
		},
	];

	const defaultPriceRatingMetrics = [
		{
			PriceRatingRelationship: "Select a category to see price-rating analysis",
			PriceRatingCorrelation: "-",
			AvgPrice: "-",
			AvgRating: "-",
			AvgReviews: "-",
		},
	];

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

	const displayDiscountMetrics = discountLoading
		? defaultDiscountMetrics
		: discountAnalysis || defaultDiscountMetrics;

	const displayPriceRatingMetrics = priceRatingLoading
		? defaultPriceRatingMetrics
		: priceRatingRelation || defaultPriceRatingMetrics;

	return (
		<div className="min-h-screen">
			<div className="max-w-7xl space-y-8">
				<p className="text-lg font-semibold">Pricing Strategy Analysis</p>

				<div className="grid grid-cols-1 md:grid-cols-2 gap-4">
					<Card className="overflow-hidden shadow-lg">
						<CardHeader className="bg-white">
							<CardTitle className="text-xl text-gray-800">
								Discount Analysis
							</CardTitle>
						</CardHeader>
						<CardContent className="bg-white">
							<select
								onChange={(e) => setSelectedDiscountCategory(e.target.value)}
								className="w-full p-2 border rounded-lg bg-white shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
								value={selectedDiscountCategory || ""}
							>
								<option value="">Select a category</option>
								{categories?.categories?.map((category, index) => (
									<option key={index} value={category}>
										{category}
									</option>
								))}
							</select>

							{discountLoading ? (
								<div className="flex my-10 items-center justify-center">
									<Loader2 className="h-8 w-8 animate-spin text-blue-500" />
								</div>
							) : (
								<div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
									{displayDiscountMetrics.map((metric, index) => (
										<Fragment key={index}>
											<Card className="bg-gradient-to-br from-blue-50 to-blue-100 shadow-sm">
												<CardContent className="p-4">
													<p className="text-lg font-semibold text-blue-900">
														Average Discount
													</p>
													<p className="text-2xl font-bold text-blue-700 mt-2">
														{metric.AvgDiscountPercentage}%
													</p>
												</CardContent>
											</Card>

											<Card className="bg-gradient-to-br from-green-50 to-green-100 shadow-sm">
												<CardContent className="p-4">
													<p className="text-lg font-semibold text-green-900">
														Total Revenue
													</p>
													<p className="text-2xl font-bold text-green-700 mt-2">
														${metric.TotalDiscountedRevenue}
													</p>
												</CardContent>
											</Card>

											<Card className="bg-gradient-to-br from-purple-50 to-purple-100 shadow-sm">
												<CardContent className="p-4">
													<p className="text-lg font-semibold text-purple-900">
														Potential Revenue
													</p>
													<p className="text-2xl font-bold text-purple-700 mt-2">
														${metric.PotentialRevenue}
													</p>
												</CardContent>
											</Card>

											<Card className="bg-gradient-to-br from-yellow-50 to-yellow-100 shadow-sm">
												<CardContent className="p-4">
													<p className="text-lg font-semibold text-yellow-900">
														Average Sales
													</p>
													<p className="text-2xl font-bold text-yellow-700 mt-2">
														{metric.AvgSales}
													</p>
												</CardContent>
											</Card>

											<Card className="bg-gradient-to-br from-indigo-50 to-indigo-100 shadow-sm">
												<CardContent className="p-4">
													<p className="text-lg font-semibold text-indigo-900">
														ROI
													</p>
													<p className="text-2xl font-bold text-indigo-700 mt-2">
														{metric.DiscountROI}%
													</p>
												</CardContent>
											</Card>

											<Card className="bg-gradient-to-br from-red-50 to-red-100 shadow-sm">
												<CardContent className="p-4">
													<p className="text-lg font-semibold text-red-900">
														Revenue Loss
													</p>
													<p className="text-2xl font-bold text-red-700 mt-2">
														${metric.RevenueLossFromDiscount}
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
						<CardHeader className="bg-white">
							<CardTitle className="text-xl text-gray-800">
								Price-Rating Analysis
							</CardTitle>
						</CardHeader>
						<CardContent className="bg-white">
							<select
								onChange={(e) => setSelectedPriceRatingCategory(e.target.value)}
								className="w-full p-2 border rounded-lg bg-white shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
								value={selectedPriceRatingCategory || ""}
							>
								<option value="">Select a category</option>
								{categories?.categories?.map((category, index) => (
									<option key={index} value={category}>
										{category}
									</option>
								))}
							</select>

							{priceRatingLoading ? (
								<div className="flex my-10 items-center justify-center">
									<Loader2 className="h-8 w-8 animate-spin text-blue-500" />
								</div>
							) : (
								<div>
									{displayPriceRatingMetrics.map((metric, index) => (
										<Fragment key={index}>
											<Card className="mt-6 bg-gradient-to-br from-gray-50 to-gray-100">
												<CardContent className="p-4">
													<p className="text-lg font-semibold text-gray-900 mb-2">
														Price-Rating Relationship
													</p>
													<p className="text-gray-700">
														{metric.PriceRatingRelationship}
													</p>
												</CardContent>
											</Card>

											<div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
												<Card className="bg-gradient-to-br from-pink-50 to-pink-100 shadow-sm">
													<CardContent className="p-4">
														<p className="text-lg font-semibold text-pink-900">
															Correlation
														</p>
														<p className="text-2xl font-bold text-pink-700 mt-2">
															{metric.PriceRatingCorrelation}
														</p>
													</CardContent>
												</Card>

												<Card className="bg-gradient-to-br from-orange-50 to-orange-100 shadow-sm">
													<CardContent className="p-4">
														<p className="text-lg font-semibold text-orange-900">
															Average Price
														</p>
														<p className="text-2xl font-bold text-orange-700 mt-2">
															${metric.AvgPrice}
														</p>
													</CardContent>
												</Card>

												<Card className="bg-gradient-to-br from-teal-50 to-teal-100 shadow-sm">
													<CardContent className="p-4">
														<p className="text-lg font-semibold text-teal-900">
															Average Rating
														</p>
														<p className="text-2xl font-bold text-teal-700 mt-2">
															{metric.AvgRating}
														</p>
													</CardContent>
												</Card>

												<Card className="bg-gradient-to-br from-cyan-50 to-cyan-100 shadow-sm">
													<CardContent className="p-4">
														<p className="text-lg font-semibold text-cyan-900">
															Average Reviews
														</p>
														<p className="text-2xl font-bold text-cyan-700 mt-2">
															{metric.AvgReviews}
														</p>
													</CardContent>
												</Card>
											</div>
										</Fragment>
									))}
								</div>
							)}
						</CardContent>
					</Card>
				</div>
			</div>
		</div>
	);
};

export default PricingStrategyAnalysisPage;
