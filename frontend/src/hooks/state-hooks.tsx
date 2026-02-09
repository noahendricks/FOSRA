import * as React from "react";
import { create } from "zustand";
import { devtools, persist } from "zustand/middleware";

import { WorkspacePreferencesApi } from "../lib/api";

interface NavigationState {
  activeWorkspaceId: string;
  activeUserId: string;
  activeConvoName: string;
  activeConvoId: string;
  activeWorkspaceName: string;

  setActiveConvoId: (ConvoId: string) => void;
  setActiveUserId: (UserId: string) => void;
  setActiveConvoName: (ConvoName: string) => void;
  setActiveWorkspaceId: (WorkspaceId: string) => void;
  setActiveWorkspaceName: (WorkspaceName: string) => void;
}

export const useNavStore = create<NavigationState>()(
  persist(
    (set) => ({
      activeConvoId: "",
      activeConvoName: "",
      activeUserId: "",
      activeWorkspaceId: "",
      activeWorkspaceName: "",

      setActiveConvoId: (ConvoId: string) =>
        set(() => ({ activeConvoId: ConvoId })),
      setActiveUserId: (UserId: string) => set({ activeUserId: UserId }),
      setActiveConvoName: (ConvoName: string) =>
        set({
          activeConvoName: ConvoName,
        }),
      setActiveWorkspaceId: (WorkspaceId: string) =>
        set({ activeWorkspaceId: WorkspaceId }),
      setActiveWorkspaceName: (WorkspaceName: string) =>
        set({
          activeWorkspaceName: WorkspaceName,
        }),
    }),
    {
      name: "navigation-storage",
    },
  ),
);
interface SettingsState {
  activeUserSettings: WorkspacePreferencesApi;
  setActiveUserSettings: (settings: WorkspacePreferencesApi) => void;
  updateSettings: (partial: Partial<WorkspacePreferencesApi>) => void;
}

export const useSettingsStore = create<SettingsState>()((set) => ({
  activeUserSettings: {},

  setActiveUserSettings: (settings) => set({ activeUserSettings: settings }),
  updateSettings: (partial) =>
    set((state) => ({
      activeUserSettings: { ...state.activeUserSettings, ...partial },
    })),
}));
