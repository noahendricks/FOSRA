import * as React from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { Link } from "@tanstack/react-router";
import {
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";

import { useNavStore } from "../../hooks/state-hooks";
import { Button } from "../ui/button";
import { getListOfConvosWorkspacesUserIdWorkspaceIdGetConvosListGetOptions } from "../../lib/api/@tanstack/react-query.gen";
import { useQuery } from "@tanstack/react-query";

export const MemoizedConvoItem = React.memo(
  ({
    id,
    title,
    isActive,
    style,
    onClick,
  }: {
    id: string;
    title: string;
    isActive: boolean;
    style: React.CSSProperties;
    onClick: (id: string, title: string) => void;
  }) => {
    const handleClick = React.useCallback(
      (e: React.MouseEvent) => {
        console.log(title);
        onClick(id, title);
      },
      [id, title, onClick],
    );

    return (
      <div style={style}>
        <SidebarMenuItem>
          <SidebarMenuButton
            asChild
            isActive={isActive}
            className="transition-none hover:bg-sidebar-accent"
          >
            <Link
              to="/convo/$ConvoId"
              params={{ ConvoId: id }}
              preload={"intent"}
              onClick={handleClick}
            >
              <span className="truncate">
                {title || "Untitled Conversation"}
              </span>
            </Link>
          </SidebarMenuButton>
        </SidebarMenuItem>
      </div>
    );
  },
  (prev, next) => {
    return (
      prev.id === next.id &&
      prev.title === next.title &&
      prev.isActive === next.isActive &&
      prev.style.transform === next.style.transform &&
      prev.style.height === next.style.height
    );
  },
);

MemoizedConvoItem.displayName = "MemoizedConvoItem";

export const ConvosList = () => {
  const {
    setActiveConvoId,
    activeWorkspaceId,
    activeConvoId,
    activeUserId,
    setActiveConvoName,
  } = useNavStore();

  const ConvosList = () => {
    const { setActiveConvoId, activeWorkspaceId, activeConvoId, activeUserId } =
      useNavStore();
  };

  const { data: convoList = [], isLoading } = useQuery({
    ...getListOfConvosWorkspacesUserIdWorkspaceIdGetConvosListGetOptions({
      path: {
        user_id: activeUserId,
        workspace_id: activeWorkspaceId,
      },
    }),
    enabled: true,
  });

  const handleClick = React.useCallback(
    (id: string, title: string) => {
      setActiveConvoId(id);
      setActiveConvoName(title);
    },
    [setActiveConvoId],
  );

  const parentRef = React.useRef<HTMLDivElement>(null);

  const rowVirtualizer = useVirtualizer({
    count: convoList.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 35,
    overscan: 3,
    measureElement:
      typeof window !== "undefined" &&
        navigator.userAgent.indexOf("Firefox") === -1
        ? (element) => element?.getBoundingClientRect()?.height
        : undefined,
    debug: false,
    enabled: true,
  });

  const virtualItems = rowVirtualizer.getVirtualItems();

  return (
    <div className="">
      <SidebarGroup className="flex-1 min-h-0 flex flex-col">
        <SidebarGroupLabel>
          Recent Chats ({convoList?.length ?? 0})
        </SidebarGroupLabel>
        <SidebarGroupContent className="flex-1 relative"></SidebarGroupContent>
      </SidebarGroup>

      <SidebarMenu className="h-full">
        <div
          ref={parentRef}
          className="h-full  w-full overflow-y-auto scrollbar-hide"
          style={{
            height: `${rowVirtualizer.getTotalSize()}px`,
            contain: "strict",
            willChange: "scroll-position",
            position: "relative",
            scrollbarGutter: "unset",
          }}
        >
          {virtualItems.map((virtualRow) => {
            const convo = convoList?.[virtualRow.index];

            if (!convo) return null;

            return (
              <MemoizedConvoItem
                key={virtualRow.key}
                id={convo.ConvoId}
                title={convo.Title ?? ""}
                isActive={activeConvoId === convo.ConvoId}
                onClick={handleClick}
                style={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  width: "100%",
                  height: `${virtualRow.size} px`,
                  transform: `translateY(${virtualRow.start}px)`,
                }}
              />
            );
          })}
        </div>
      </SidebarMenu>
    </div>
  );
};

type ButtonProps = React.ComponentProps<typeof Button>;

export const LogoutButton = () => {
  const handleLogout = () => {
    const { activeUserId, setActiveUserId } = useNavStore.getState();
    if (activeUserId != null) {
      setActiveUserId("");
    }
  };

  return <Button onClick={handleLogout}>Logout</Button>;
};
