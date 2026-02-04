import * as React from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { Link } from "@tanstack/react-router";
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
  SidebarRail,
} from "@/components/ui/sidebar";

import { SettingsPanel } from "./settings-panel";
import { ConvoCreatePanel, UserPanel } from "./user-panel";
import { SearchForm } from "./search-form";
import { WorkspaceCreatePanel, WorkspaceSwitcher } from "./workspace-switcher";
import { useNavStore } from "../../hooks/state-hooks";
import { ConvosList, LogoutButton } from "./sidebar-comps";

// --- ATOMIC ITEM COMPONENT ---

export function AppSidebar({ ...props }: React.ComponentProps<typeof Sidebar>) {
  const activeConvoId = useNavStore((state) => state.activeConvoId);
  const {
    setActiveConvoId,
    activeWorkspaceId,
    setActiveWorkspaceId,
    setActiveConvoName,
  } = useNavStore();

  const [settingsOpen, setSettingsOpen] = React.useState(false);
  const [userProfileOpen, setUserProfileOpen] = React.useState(false);
  const [convoCreateOpen, setConvoCreateOpen] = React.useState(false);
  const [workspaceCreateOpen, setWorkspaceCreateOpen] = React.useState(false);
  const [isLoading, setIsLoading] = React.useState(false);

  if (isLoading) {
    return (
      <Sidebar {...props}>
        <SidebarHeader>
          <WorkspaceSwitcher />
          loading...
          <WorkspaceCreatePanel
            open={workspaceCreateOpen}
            onOpenChange={setWorkspaceCreateOpen}
          />
          loading...
          <SearchForm />
        </SidebarHeader>
        <SidebarContent>
          <SidebarGroup>
            <SidebarGroupLabel>Loading...</SidebarGroupLabel>
          </SidebarGroup>
        </SidebarContent>
      </Sidebar>
    );
  }

  return (
    <Sidebar {...props}>
      <SidebarContent className="flex flex-col h-full overflow-hidden">
        <SidebarHeader>
          <WorkspaceSwitcher />
          <WorkspaceCreatePanel
            open={workspaceCreateOpen}
            onOpenChange={setWorkspaceCreateOpen}
          />
          <SearchForm />
        </SidebarHeader>
        {activeWorkspaceId && ConvosList()}
        <SidebarGroup className="mt-auto border-t bg-sidebar">
          <SidebarMenu>
            <SidebarMenuItem>
              <SettingsPanel
                open={settingsOpen}
                onOpenChange={setSettingsOpen}
              />
              <ConvoCreatePanel
                open={convoCreateOpen}
                onOpenChange={setConvoCreateOpen}
              />
              <LogoutButton />
            </SidebarMenuItem>
          </SidebarMenu>
        </SidebarGroup>
      </SidebarContent>
      <SidebarRail />
    </Sidebar>
  );
}
