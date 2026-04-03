import { Global } from "@/global";
import { Filesystem } from "@/util/filesystem";
import { createSignal, type Setter } from "solid-js";
import { createStore } from "solid-js/store";
import { createSimpleContext } from "./helper";
import path from "path";
import { log } from "@/util/log";

function createDebouncedWriter(filePath: string, store: Record<string, any>) {
  let writeTimer: ReturnType<typeof setTimeout> | null = null;
  let pending = false;

  function flush() {
    if (writeTimer) {
      clearTimeout(writeTimer);
      writeTimer = null;
    }
    if (pending) {
      log.config.debug("KV_FLUSH", { store });
      Filesystem.writeJson(filePath, store).catch(() => {});
      pending = false;
    }
  }

  function scheduleWrite() {
    pending = true;
    if (writeTimer) clearTimeout(writeTimer);
    writeTimer = setTimeout(flush, 500);
  }

  if (typeof process !== "undefined" && process.on) {
    process.on("SIGTERM", flush);
    process.on("SIGINT", flush);
    process.on("exit", flush);
  }

  return { scheduleWrite, flush };
}

export const { use: useKV, provider: KVProvider } = createSimpleContext({
  name: "KV",
  init: () => {
    const [ready, setReady] = createSignal(false);
    const [store, setStore] = createStore<Record<string, any>>();
    const filePath = path.join(Global.Path.state, "kv.json");

    Filesystem.readJson(filePath)
      .then((x) => {
        log.config.debug("KV_LOADED", { store: x });
        setStore(x);
      })
      .catch(() => {})
      .finally(() => {
        setReady(true);
      });

    const writer = createDebouncedWriter(filePath, store);

    const result = {
      get ready() {
        return ready();
      },
      get store() {
        return store;
      },
      signal<T>(name: string, defaultValue: T) {
        if (store[name] === undefined) setStore(name, defaultValue);
        return [
          function () {
            return result.get(name);
          },
          function setter(next: Setter<T>) {
            result.set(name, next);
          },
        ] as const;
      },
      get(key: string, defaultValue?: any) {
        return store[key] ?? defaultValue;
      },
      set(key: string, value: any) {
        log.config.debug("KV_SET", { key, value });
        setStore(key, value);
        writer.scheduleWrite();
      },
    };
    return result;
  },
});
