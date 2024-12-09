import React from "react";
import { SidebarProvider, SidebarTrigger } from "./components/ui/sidebar";
import { AppSidebar } from "./pages/AppSideBar";

const Layout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
	return (
		<SidebarProvider>
			<AppSidebar />
			<main className="h-full w-full bg-neutral-200 p-4 flex">
				<SidebarTrigger />
				{children}
			</main>
		</SidebarProvider>
	);
};

export default Layout;
