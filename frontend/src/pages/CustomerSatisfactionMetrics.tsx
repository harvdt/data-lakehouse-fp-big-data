import useFetch from "@/hooks/useFetch";
import { Categories, CustomerSatisfaction } from "@/types/api";
import { Fragment, useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Loader2 } from "lucide-react";

function CustomerSatisfactionMetricsPage() {
	const [selectedCategory, setSelectedCategory] = useState<string | null>(null);

	const baseAPI = "http://localhost:8000";

	const {
		data: categories,
		error: categoriesError,
		loading: categoriesLoading,
	} = useFetch<Categories>(`${baseAPI}/categories`);

	const { data: customerSatisfaction, loading: customerSatisfactionLoading } =
		useFetch<CustomerSatisfaction>(
			`${baseAPI}/analytics/customer-satisfaction?category=${selectedCategory}`
		);

	const defaultCustomerSatisfaction = [
		{
			TotalReviews: "-",
			TotalSales: "-",
			AvgRating: "-",
			HighRatingCount: "-",
			ReviewToSalesRatio: "-",
			CustomerEngagementLevel: "-",
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

	const displayCustomerSatisfaction = customerSatisfactionLoading
		? defaultCustomerSatisfaction
		: customerSatisfaction || defaultCustomerSatisfaction;

	return (
		<main className="min-h-screen">
			<div className="max-w-7xl space-y-8">
				<p className="text-lg font-semibold">Customer Satisfaction Metrics</p>

				<Card className="overflow-hidden shadow-lg">
					<CardHeader className="bg-white">
						<CardTitle className="text-xl text-gray-800">
							Customer Satisfaction Metrics
						</CardTitle>
					</CardHeader>

					<CardContent>
						<select
							onChange={(e) => setSelectedCategory(e.target.value)}
							className="w-full p-2 border rounded-lg bg-white shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
							value={selectedCategory || ""}
						>
							<option value="">Select a category</option>
							{categories?.categories?.map((category, index) => (
								<option key={index} value={category}>
									{category}
								</option>
							))}
						</select>

						{customerSatisfactionLoading ? (
							<div className="flex my-10 items-center justify-center">
								<Loader2 className="h-8 w-8 animate-spin text-blue-500" />
							</div>
						) : (
							<div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
								{displayCustomerSatisfaction.map((metric, index) => (
									<Fragment key={index}>
										<Card className="bg-gradient-to-br from-blue-50 to-blue-100 shadow-sm">
											<CardContent className="p-4">
												<p className="text-lg font-semibold text-blue-900">
													Total Reviews
												</p>
												<p className="text-2xl font-bold text-blue-700 mt-2">
													{metric.TotalReviews}
												</p>
											</CardContent>
										</Card>

										<Card className="bg-gradient-to-br from-green-50 to-green-100 shadow-sm">
											<CardContent className="p-4">
												<p className="text-lg font-semibold text-green-900">
													Total Sales
												</p>
												<p className="text-2xl font-bold text-green-700 mt-2">
													{metric.TotalSales}
												</p>
											</CardContent>
										</Card>

										<Card className="bg-gradient-to-br from-purple-50 to-purple-100 shadow-sm">
											<CardContent className="p-4">
												<p className="text-lg font-semibold text-purple-900">
													Average Rating
												</p>
												<p className="text-2xl font-bold text-purple-700 mt-2">
													{metric.AvgRating}
												</p>
											</CardContent>
										</Card>

										<Card className="bg-gradient-to-br from-yellow-50 to-yellow-100 shadow-sm">
											<CardContent className="p-4">
												<p className="text-lg font-semibold text-yellow-900">
													High Rating Count
												</p>
												<p className="text-2xl font-bold text-yellow-700 mt-2">
													{metric.HighRatingCount}
												</p>
											</CardContent>
										</Card>

										<Card className="bg-gradient-to-br from-indigo-50 to-indigo-100 shadow-sm">
											<CardContent className="p-4">
												<p className="text-lg font-semibold text-indigo-900">
													Review to Sales Ratio
												</p>
												<p className="text-2xl font-bold text-indigo-700 mt-2">
													{metric.ReviewToSalesRatio}
												</p>
											</CardContent>
										</Card>

										<Card className="bg-gradient-to-br from-red-50 to-red-100 shadow-sm">
											<CardContent className="p-4">
												<p className="text-lg font-semibold text-red-900">
													Revenue Loss
												</p>
												<p className="text-2xl font-bold text-red-700 mt-2">
													{metric.CustomerEngagementLevel}
												</p>
											</CardContent>
										</Card>
									</Fragment>
								))}
							</div>
						)}
					</CardContent>
				</Card>
			</div>
		</main>
	);
}

export default CustomerSatisfactionMetricsPage;
