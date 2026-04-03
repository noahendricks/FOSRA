import { createMemo } from "solid-js";
import { Keybind, type KeybindInfo } from "@/util/keybind";
import { pipe, mapValues } from "remeda";
import type { TuiConfig } from "@/config/tui";
import type { ParsedKey, Renderable } from "@opentui/core";
import { createStore } from "solid-js/store";
import { useKeyboard, useRenderer } from "@opentui/solid";
import { createSimpleContext } from "./helper";
import { useTuiConfig } from "./tui-config";
import { log } from "@/util/log";

export type KeybindKey = keyof NonNullable<TuiConfig.Info["keybinds"]> & string;

export const { use: useKeybind, provider: KeybindProvider } =
  createSimpleContext({
    name: "Keybind",
    init: () => {
      const config = useTuiConfig();
      const keybinds = createMemo<Record<string, KeybindInfo[]>>(() => {
        return pipe(
          (config.keybinds ?? {}) as Record<string, string>,
          mapValues((value) => Keybind.parse(value)),
        );
      });
      const [store, setStore] = createStore({
        leader: false,
      });
      const renderer = useRenderer();

      let focus: Renderable | null;
      let timeout: NodeJS.Timeout;
      function leader(active: boolean) {
        if (active) {
          setStore("leader", true);
          focus = renderer.currentFocusedRenderable;
          focus?.blur();
          if (timeout) clearTimeout(timeout);
          timeout = setTimeout(() => {
            if (!store.leader) return;
            leader(false);
            if (!focus || focus.isDestroyed) return;
            focus.focus();
          }, 2000);
          return;
        }

        if (!active) {
          if (focus && !renderer.currentFocusedRenderable) {
            focus.focus();
          }
          setStore("leader", false);
        }
      }

      useKeyboard(async (evt) => {
        if (!store.leader && result.match("leader", evt)) {
          log.keybind.info("LEADER_ON");
          leader(true);
          return;
        }

        if (store.leader && evt.name) {
          log.keybind.info("LEADER_KEY", { name: evt.name, ctrl: evt.ctrl });
          setImmediate(() => {
            if (focus && renderer.currentFocusedRenderable === focus) {
              focus.focus();
            }
            leader(false);
          });
        }
      });

      const result = {
        get all() {
          return keybinds();
        },
        get leader() {
          return store.leader;
        },
        parse(evt: ParsedKey): KeybindInfo {
          // handle special case for Ctrl+Underscore (represented as \x1F)
          if (evt.name === "\x1F") {
            return Keybind.fromParsedKey(
              { ...evt, name: "_", ctrl: true },
              store.leader,
            );
          }
          return Keybind.fromParsedKey(evt, store.leader);
        },
        match(key: KeybindKey, evt: ParsedKey) {
          const keybind = keybinds()[key];
          if (!keybind) return false;
          const parsed: KeybindInfo = result.parse(evt);
          for (const key of keybind) {
            if (Keybind.match(key, parsed)) {
              return true;
            }
          }
        },
        print(key: KeybindKey) {
          const first = keybinds()[key]?.at(0);
          if (!first) return "";
          if (first.leader) {
            const leaderInfo = keybinds().leader?.[0];
            if (!leaderInfo) return first.name;
            return `${Keybind.toString(leaderInfo)} ${first.name}`;
          }
          return Keybind.toString(first);
        },
      };
      return result;
    },
  });
