import * as React from "react";
import {
  Link,
  Outlet,
  createRootRouteWithContext,
  redirect,
  useNavigate,
} from "@tanstack/react-router";

import { TanStackRouterDevtools } from "@tanstack/react-router-devtools";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";

import { queryClient } from "@/main";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar";
import { AppSidebar } from "../components/fosra-ui/app-sidebar";
import { useNavStore } from "../hooks/state-hooks";

export const Route = createRootRouteWithContext<{
  queryClient: typeof queryClient;
}>()({
  notFoundComponent: () => {
    return (
      <div>
        <p>This is the not FoundComponent configured on root route</p>
        <Link to="/">Start Over</Link>
      </div>
    );
  },

  component: Root,
});

function Root() {
  const { activeUserId, activeConvoName } = useNavStore.getState();

  const navigate = useNavigate();

  if (!activeUserId) {
    navigate({ to: "/login", from: "/convo" });
  }

  return (
    <>
      <SidebarProvider className="h-dvh">
        {activeUserId && <AppSidebar className=" text-foreground" />}
        <SidebarInset className=" text-foreground">
          <header className="flex sticky h-16 shrink-0 flex-none items-center gap-2 border-b px-4">
            <SidebarTrigger className="-ml-1" />
            <Breadcrumb>
              <BreadcrumbList></BreadcrumbList>
              <BreadcrumbItem className="hidden md:block"></BreadcrumbItem>
              <BreadcrumbItem></BreadcrumbItem>
            </Breadcrumb>
            <div className="self-auto h-auto w-full flex justify-center content-end">
              <BreadcrumbPage className="align-middle">
                {activeConvoName ?? "none"}
              </BreadcrumbPage>
            </div>
          </header>
          <Outlet />
        </SidebarInset>
      </SidebarProvider>
      <ReactQueryDevtools buttonPosition="top-right" />
      <TanStackRouterDevtools position="bottom-right" />
    </>
  );
}
