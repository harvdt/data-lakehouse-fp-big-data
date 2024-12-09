export interface Categories {
	categories: string[];
}

export interface SalesPrediction {
	predicted_sales: number;
	confidence_score: number;
}

export type PricingStrategy = PricingAnalysis[];

export interface PricingAnalysis {
	Category: string;
	TotalRevenue: number;
	TotalSales: number;
	AveragePrice: number;
	ProductCount: number;
	AvgDiscountPercentage: number;
}

export type DiscountAnalysis = DiscountAnalysisMetrics[];

export interface DiscountAnalysisMetrics {
	Category: string;
	AvgDiscountPercentage: number;
	TotalDiscountedRevenue: number;
	PotentialRevenue: number;
	AvgSales: number;
	RevenueLossFromDiscount: number;
	DiscountROI: number;
}

export type PriceRatingRelation = PriceRatingAnalysis[];

export interface PriceRatingAnalysis {
	Category: string;
	PriceRatingCorrelation: number;
	AvgPrice: number;
	AvgRating: number;
	AvgReviews: number;
	PriceRatingRelationship: string;
}

export type CustomerSatisfaction = CustomerSatisfactionMetrics[];

export interface CustomerSatisfactionMetrics {
	Category: string;
	TotalReviews: number;
	TotalSales: number;
	AvgRating: number;
	HighRatingCount: number;
	ReviewToSalesRatio: number;
	CustomerEngagementLevel: string;
}
