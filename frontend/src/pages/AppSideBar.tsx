import { UsersRound, ChartArea, BadgeDollarSign } from "lucide-react";
import { Link } from "react-router-dom";

import {
	Sidebar,
	SidebarContent,
	SidebarGroup,
	SidebarGroupContent,
	SidebarGroupLabel,
	SidebarHeader,
	SidebarMenu,
	SidebarMenuButton,
	SidebarMenuItem,
	useSidebar,
} from "@/components/ui/sidebar";

const items = [
	{
		title: "Sales Analysis Performance",
		url: "/",
		icon: ChartArea,
	},
	{
		title: "Pricing Strategy Analysis",
		url: "/pricing-strategy",
		icon: BadgeDollarSign,
	},
	{
		title: "Customer Satisfaction Metrics",
		url: "/customer-satisfaction",
		icon: UsersRound,
	},
];

export function AppSidebar() {
	const { state } = useSidebar();

	return (
		<Sidebar collapsible="icon">
			<SidebarContent>
				{state === "expanded" && (
					<SidebarHeader>
						<p className="text-xl font-semibold">
							E-Commerce XYZ Business Intelligence
						</p>
					</SidebarHeader>
				)}
				<SidebarGroup>
					<SidebarGroupLabel>Menu</SidebarGroupLabel>
					<SidebarGroupContent>
						<SidebarMenu>
							{items.map((item) => (
								<SidebarMenuItem key={item.title}>
									<SidebarMenuButton asChild>
										<Link to={item.url}>
											<item.icon />
											<span>{item.title}</span>
										</Link>
									</SidebarMenuButton>
								</SidebarMenuItem>
							))}
						</SidebarMenu>
					</SidebarGroupContent>
				</SidebarGroup>
			</SidebarContent>
		</Sidebar>
	);
}
