import * as React from "react";
import { Check, ChevronsUpDown, GalleryVerticalEnd } from "lucide-react";

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";
import { nanoid } from "nanoid";
import { useGetWorkspaceList } from "../../hooks/convo_hooks";
import { useNavStore } from "../../hooks/state-hooks";
import { Button } from "../ui/button";
import {
  createNewConvoWorkspacesUserIdWorkspaceIdNewConvoPost,
  listUserWorkspacesWorkspacesUserIdListWorkspacesGet,
  WorkspaceFullResponse,
} from "../../lib/api";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  createNewConvoWorkspacesUserIdWorkspaceIdNewConvoPostMutation,
  listUserWorkspacesWorkspacesUserIdListWorkspacesGetOptions,
  listUserWorkspacesWorkspacesUserIdListWorkspacesGetQueryKey,
  newWorkspaceWorkspacesUserIdCreateWorkspacePostMutation,
} from "../../lib/api/@tanstack/react-query.gen";
import { PanelProps } from "./user-panel";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Textarea } from "../ui/textarea";
import { queryClient } from "../../main";

const MemoizedWorkspaceListItem = React.memo(
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
    onClick;
  }) => {
    return (
      <div style={style}>
        <DropdownMenuItem key={id} onClick={onClick}>
          {title}
        </DropdownMenuItem>
      </div>
    );
  },
);

MemoizedWorkspaceListItem.displayName = "MemoizedWorkspaceListItem";

export type ButtonProps = React.ComponentProps<typeof Button>;

export const WorkspaceCreatePanel = ({ open, onOpenChange }: PanelProps) => {
  const [title, setTitle] = React.useState<string | undefined>();
  const [description, setDescription] = React.useState<string | undefined>();

  const {
    activeWorkspaceId,
    activeUserId,
    setActiveWorkspaceId,
    setActiveWorkspaceName,
  } = useNavStore();

  const { mutateAsync: createWorkspace } = useMutation({
    ...newWorkspaceWorkspacesUserIdCreateWorkspacePostMutation(),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: listUserWorkspacesWorkspacesUserIdListWorkspacesGetQueryKey({
          path: { user_id: activeUserId },
        }),
      });
    },
  });

  const handleSave = React.useCallback(
    async (description: string | undefined, name: string | undefined) => {
      console.log("ran handle save");

      const post = await createWorkspace({
        query: { UserId: activeUserId, Description: description, Name: name },
      });

      const { WorkspaceId, Name } = post;

      console.log(activeWorkspaceId);
      console.log("before", WorkspaceId);
      console.log(setActiveWorkspaceId(WorkspaceId));

      console.log("after", activeWorkspaceId);
      setActiveWorkspaceName(Name ?? "");

      onOpenChange?.(false);

      setTitle("");
      setDescription("");
    },
    [
      setActiveWorkspaceId,
      activeWorkspaceId,
      activeUserId,
      createWorkspace,
      onOpenChange,
    ],
  );

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogTrigger asChild>
        <Button>New</Button>
      </DialogTrigger>
      <DialogContent className="max-w-21">
        <DialogHeader>
          <DialogTitle>New Workspace</DialogTitle>
        </DialogHeader>
        <div>
          <Label>Workspace Title</Label>
          <Textarea
            value={title || ""}
            onChange={(e) => setTitle(e.target.value)}
            onSubmit={() => {
              setTitle("");
            }}
          ></Textarea>
        </div>
        <div>
          <Label>Description</Label>
          <Textarea
            value={description || ""}
            onChange={(e) => setDescription(e.target.value)}
          ></Textarea>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange?.(false)}>
            Cancel
          </Button>
          <Button
            onClick={() => handleSave(description, title)}
            variant="outline"
          >
            Save Changes
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export function WorkspaceSwitcher() {
  const {
    activeWorkspaceId,
    setActiveWorkspaceId,
    setActiveWorkspaceName,
    activeWorkspaceName,
    activeUserId,
  } = useNavStore();

  const [workspaceCreateOpen, setWorkspaceCreateOpen] = React.useState(false);

  const { data: workspaceList = [] } = useQuery({
    ...listUserWorkspacesWorkspacesUserIdListWorkspacesGetOptions({
      path: { user_id: activeUserId },
    }),
    enabled: !!activeUserId,
  });

  const handleClick = React.useCallback(
    (id: string, title: string) => {
      console.log(id, title);
      console.log("clicked");
      setActiveWorkspaceId(id);
      setActiveWorkspaceName(title);
    },
    [setActiveWorkspaceId, setActiveWorkspaceName],
  );

  return (
    <SidebarMenu>
      <SidebarMenuItem>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <SidebarMenuButton
              size="lg"
              className="data-[state=open]:bg-sidebar-accent data-[state=open]:text-sidebar-accent-foreground"
            >
              <div className="bg-sidebar-primary text-sidebar-primary-foreground flex aspect-square size-8 items-center justify-center rounded-lg">
                <GalleryVerticalEnd className="size-4" />
              </div>

              <div className="flex flex-col gap-0.5 leading-none">
                <span className="font-medium">Workspace</span>
                {activeWorkspaceName}
              </div>

              <ChevronsUpDown className="ml-auto" />
            </SidebarMenuButton>
          </DropdownMenuTrigger>
          <DropdownMenuContent
            className="w-(--radix-dropdown-menu-trigger-width)"
            align="start"
            style={{ scrollbarWidth: "none" }}
          >
            {workspaceList?.map((row) => {
              const workspace = row;

              if (!workspace) return [];
              return (
                <MemoizedWorkspaceListItem
                  id={row.WorkspaceId}
                  title={row.Name ?? "cant get"}
                  key={row.WorkspaceId || nanoid()}
                  isActive={activeWorkspaceId === row.WorkspaceId}
                  onClick={() => handleClick(row.WorkspaceId, row.Name)}
                />
              );
            })}
          </DropdownMenuContent>
        </DropdownMenu>
      </SidebarMenuItem>
    </SidebarMenu>
  );
}
