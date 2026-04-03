import { useRenderer } from "@opentui/solid";
import { createSimpleContext } from "./helper";
import { FormatError, FormatUnknownError } from "@/cli/error";
import { win32FlushInputBuffer } from "../win32";
import { log } from "@/util/log";
type Exit = ((reason?: unknown) => Promise<void>) & {
  message: {
    set: (value?: string) => () => void;
    clear: () => void;
    get: () => string | undefined;
  };
};

export const { use: useExit, provider: ExitProvider } = createSimpleContext({
  name: "Exit",
  init: (input: { onExit?: () => Promise<void> }) => {
    const renderer = useRenderer();
    let message: string | undefined;
    let task: Promise<void> | undefined;
    const store = {
      set: (value?: string) => {
        const prev = message;
        message = value;
        return () => {
          message = prev;
        };
      },
      clear: () => {
        message = undefined;
      },
      get: () => message,
    };
    const exit: Exit = Object.assign(
      (reason?: unknown) => {
        log.startup.info("EXIT_CALLED", { reason });
        if (task) return task;
        task = (async () => {
          // reset window title before destroying renderer
          renderer.setTerminalTitle("");
          renderer.destroy();
          win32FlushInputBuffer();
          if (reason) {
            const formatted =
              FormatError(reason as Error) ?? FormatUnknownError(reason);
            if (formatted) {
              process.stderr.write(formatted + "\n");
            }
          }
          const text = store.get();
          if (text) process.stdout.write(text + "\n");
          await input.onExit?.();
        })();
        return task;
      },
      {
        message: store,
      },
    );
    return exit;
  },
});
