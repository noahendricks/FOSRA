// Stub for @/config/tui
export interface TuiConfigInfo {
  keybinds: Record<string, string>
  sidebar: "auto" | "open" | "closed"
  scrollSpeed: number
}

export const TuiConfig = {
  Info: {
    keybinds: {
      session_list: "<leader>s s",
      session_new: "<leader>s n",
      model_list: "<leader>s m",
      agent_list: "<leader>s a",
      theme_list: "<leader>s t",
      status_view: "<leader>s v",
      session_parent: "<leader>s p",
      session_prev: "<leader>s [",
      session_next: "<leader>s ]",
      model_cycle_recent: "<leader>m r",
      model_cycle_favorite: "<leader>m f",
      agent_cycle: "<leader>a c",
      variant_cycle: "<leader>v c",
      terminal_suspend: "<leader>t s",
      terminal_title_toggle: "<leader>t t",
    },
    sidebar: "auto" as const,
    scrollSpeed: 3,
  },
}