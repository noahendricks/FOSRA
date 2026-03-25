// Stub for @/config/tui
export namespace TuiConfig {
  export interface Info {
    keybinds: Record<string, string>
    sidebar: "auto" | "open" | "closed"
    scrollSpeed: number
    scroll_speed: number
    scroll_acceleration: { enabled: boolean } | number
    theme: string
    diff_style: "compact" | "full" | "inline" | "stacked"
  }
}

export const TuiConfig: { Info: TuiConfig.Info } = {
  Info: {
    keybinds: {
      leader: "ctrl+space",
      command_list: "ctrl+k",
      input_submit: "enter",
      app_exit: "ctrl+c",
      session_list: "<leader> s",
      session_new: "<leader> n",
      model_list: "<leader> m",
      agent_list: "<leader> a",
      theme_list: "<leader> t",
      status_view: "<leader> v",
      session_parent: "<leader> p",
      session_prev: "<leader> [",
      session_next: "<leader> ]",
      model_cycle_recent: "<leader> r",
      model_cycle_favorite: "<leader> f",
      agent_cycle: "<leader> c",
      session_interrupt: "escape",
    },
    sidebar: "auto",
    scrollSpeed: 3,
    scroll_speed: 3,
    scroll_acceleration: { enabled: true },
    theme: "default",
    diff_style: "compact",
  },
}
