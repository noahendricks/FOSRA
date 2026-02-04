import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import React, {
  ComponentProps,
  JSX,
  ReactNode,
  useEffect,
  useMemo,
  useState,
} from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Switch } from "@/components/ui/switch";
import { Field } from "../ui/field";
import { Textarea } from "../ui/textarea";
import {
  createNewConvoWorkspacesUserIdWorkspaceIdNewConvoPostMutation,
  getListOfConvosWorkspacesUserIdWorkspaceIdGetConvosListGetQueryKey,
} from "../../lib/api/@tanstack/react-query.gen";
import { useNavStore } from "../../hooks/state-hooks";
import { useMutation, useQuery } from "@tanstack/react-query";
import { queryClient } from "../../main";

export type SettingsPanelProps = ComponentProps<typeof Dialog> & {
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
};

export type SettingsPannelTriggerProps = ComponentProps<typeof DialogTrigger>;

export const SettingsPanelTrigger = (props: SettingsPannelTriggerProps) => (
  <DialogTrigger {...props} />
);

export type SettingsPanelContentProps = ComponentProps<typeof DialogContent> & {
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
};

const UserProfile = {
  userId: "ASHEAIRTHS5406545645HSA",
  name: "Jam Wiltor",
  enabled: true,
  created_at: Date.now(),
};

// Retrieve User Settings
//// Place in Zustand

// Map Over Types of Setting (LLM, Retriever, Vector, etc) & Set Tab Component
////  Map each Field in the Config & Set Field Name and Type of Input (Radio, Field, Toggle)
///// Fill each Field with Existing Value
//

console.log(Object.keys(UserProfile));
// for (const [title, field_val] of Object.entries(UserProfile)) {
// console.log(title, field_val, `\n`);
// for (const [field, val] of Object.entries(field_val)) {
//   console.log(field, val, `  |||${typeof val}`);
// }
// console.log(`\n`);
// }

const triggers: JSX.Element[] = [];
const contents: JSX.Element[] = [];

type FieldValue = string | number | boolean | object | null;

const fieldGenerator = (field: string, val: FieldValue): JSX.Element => {
  switch (typeof val) {
    case "string":
      return (
        <div className="space-y-2">
          <Label htmlFor={field}>{field}</Label>
          <Input id={field} defaultValue={val} />
        </div>
      );

    case "boolean":
      return (
        <div className="space-y-2">
          <Label htmlFor={field}>{field}</Label>
          <Switch id={field} checked={val} defaultChecked={false} />
        </div>
      );
    case "number":
      return (
        <div className="space-y-2">
          <Label htmlFor={field}>{field}</Label>
          <Input type="number" id={field} defaultValue={val} />
        </div>
      );

    case "bigint":
      return (
        <div className="space-y-2">
          <Label htmlFor={field}>{field}</Label>
          <Field id={field} defaultValue={"BIGINT NOT SUPPORTED"} />
        </div>
      );
    case "object":
      if (val && !Array.isArray(val)) {
        return (
          <div>
            <Label>{field}</Label>
            <Textarea defaultValue={JSON.stringify(val, null, 2)} />
          </div>
        );
      } else {
        return <></>;
      }
  }
};

export const UserPanel = ({ open, onOpenChange }: SettingsPanelProps) => {
  // const settings = useSettingsStore((state) => state.activeUserSettings);
  const profile = UserProfile;

  const { triggers, contents } = useMemo(() => {
    const triggers: JSX.Element[] = [];
    const contents: JSX.Element[] = [];

    for (const [title, field_val] of Object.entries(profile)) {
      triggers.push(
        <TabsTrigger key={title} value={title}>
          {title}
        </TabsTrigger>,
      );
      const tabFields = [];

      tabFields.push(fieldGenerator(title, field_val));
      contents.push(<TabsContent value={title}>{tabFields}</TabsContent>);
    }

    return { triggers, contents };
  }, [profile]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogTrigger asChild>
        <Button variant="default" size="default">
          User Profile
        </Button>
      </DialogTrigger>

      <DialogContent className="max-w-21">
        <DialogHeader>
          <DialogTitle>User Profile</DialogTitle>
        </DialogHeader>

        <Tabs defaultValue="account" className="w-full">
          <TabsList>{triggers}</TabsList>
          {contents}
        </Tabs>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange?.(false)}>
            Cancel
          </Button>
          <Button variant="outline" onClick={() => onOpenChange?.(false)}>
            Save Changes
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export const useCreateConvo = () => {
  return useMutation(
    createNewConvoWorkspacesUserIdWorkspaceIdNewConvoPostMutation(),
  );
};

export type PanelProps = ComponentProps<typeof Dialog> & {
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
};

export type ButtonProps = ComponentProps<typeof Button>;

export const ConvoCreatePanel = ({ open, onOpenChange }: PanelProps) => {
  const [titleText, setTitleText] = useState<string | undefined>(undefined);
  const { activeWorkspaceId, activeUserId, setActiveConvoId } = useNavStore();

  const { mutate: createConvo } = useMutation({
    ...createNewConvoWorkspacesUserIdWorkspaceIdNewConvoPostMutation(),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey:
          getListOfConvosWorkspacesUserIdWorkspaceIdGetConvosListGetQueryKey({
            path: { user_id: activeUserId, workspace_id: activeWorkspaceId },
          }),
      });
    },
  });

  const handleSave = React.useCallback(
    (title) => {
      createConvo({
        query: {
          UserId: activeUserId,
          WorkspaceId: activeWorkspaceId,
          Title: title,
        },
      });
      onOpenChange?.(false);
      setTitleText("");
      queryClient.invalidateQueries();
    },
    [onOpenChange, activeWorkspaceId, createConvo, activeUserId],
  );

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogTrigger asChild>
        <Button>Create Convo</Button>
      </DialogTrigger>

      <DialogContent className="max-w-21">
        <DialogHeader>
          <DialogTitle>New Convo</DialogTitle>
        </DialogHeader>
        <div>
          <Label>Convo Title</Label>
          <Textarea
            value={titleText || ""}
            onChange={(e) => setTitleText(e.target.value)}
          ></Textarea>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange?.(false)}>
            Cancel
          </Button>
          <Button onClick={() => handleSave(titleText)} variant="outline">
            Save Changes
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
