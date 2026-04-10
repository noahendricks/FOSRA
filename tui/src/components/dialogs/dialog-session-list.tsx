import { useDialog } from "@tui/components/dialogs/dialog";
import { DialogSelect } from "@tui/components/dialogs/dialog-select";
import { useRoute } from "@tui/context/route";
import { useStore } from "@tui/context/store";
import {
  createMemo,
  createSignal,
  createResource,
  onMount,
  Show,
} from "solid-js";
import { Locale } from "@/util/locale";
import { useKeybind } from "@tui/context/keybind";
import { useTheme } from "@tui/context/theme";
import { useApi } from "@tui/context/api";
import { DialogSessionRename } from "@tui/components/dialogs/dialog-session-rename";
import { useKV } from "@tui/context/kv";
import { createDebouncedSignal } from "@tui/util/signal";
import { Spinner } from "@tui/components/spinner";
import { useToast } from "@tui/components/dialogs/toast";

export function DialogSessionList(
  props: { sessionId?: string; localOnly?: boolean } = {},
) {
  const dialog = useDialog();
  const route = useRoute();
  const store = useStore();
  const keybind = useKeybind();
  const { theme } = useTheme();
  const api = useApi();
  const kv = useKV();
  const toast = useToast();
  const [toDelete, setToDelete] = createSignal<string>();
  const [search, setSearch] = createDebouncedSignal("", 150);

  const [listed, listedActions] = createResource(
    () => props.sessionId,
    async (sessionId) => {
      if (!sessionId) return undefined;
      const result = await api.fosra.session.list({ roots: true });
      return result.data ?? [];
    },
  );

  const [searchResults] = createResource(search, async (query) => {
    if (!query || props.localOnly) return undefined;
    const result = await api.fosra.session.list({
      search: query,
      limit: 30,
      ...(props.sessionId ? { roots: true } : {}),
    });

    return result.data ?? [];
  });

  const currentSessionID = createMemo(() =>
    route.data.type === "session" ? route.data.sessionID : undefined,
  );

  const sessions = createMemo(() => {
    if (searchResults()) return searchResults()!;
    if (props.sessionId) return listed() ?? [];
    if (props.localOnly)
      return store.state
        .sessionsArray()
        .filter((session) => !session.session_id);
    return store.state.sessionsArray();
  });

  const options = createMemo(() => {
    const today = new Date().toDateString();
    return sessions()
      .filter((x) => {
        if (x.parentID != null) return false;
        if (props.sessionId && listed()) return true;
        if (props.sessionId) return x.session_id === props.sessionId;
        if (props.localOnly) return !x.session_id;
        return true;
      })
      .toSorted((a, b) => b.time.updated - a.time.updated)
      .map((x) => {
        const date = new Date(x.time.updated);
        let category = date.toDateString();
        if (category === today) {
          category = "Today";
        }
        // TODO: need to fix ,currently bugged
        // const isDeleting = toDelete() === x.session_id;
        const isDeleting = false;
        const sessionStatus = store.session.status(x.session_id);
        const isWorking = sessionStatus === "working";
        return {
          title: isDeleting
            ? `Press ${keybind.print("session_delete")} again to confirm`
            : x.title,
          bg: isDeleting ? theme.error : undefined,
          value: x.session_id,
          category,
          footer: Locale.time(x.time.updated),
          gutter: isWorking ? <Spinner /> : undefined,
        };
      });
  });

  onMount(() => {
    dialog.setSize("large");
  });

  return (
    <DialogSelect
      title={
        props.sessionId
          ? `Sessions`
          : props.localOnly
            ? "Local Sessions"
            : "Sessions"
      }
      options={options()}
      skipFilter={!props.localOnly}
      current={currentSessionID()}
      onFilter={setSearch}
      onMove={() => {
        setToDelete(undefined);
      }}
      onSelect={(option) => {
        route.navigate({
          type: "session",
          sessionID: option.value,
        });
        dialog.clear();
      }}
      keybind={[
        {
          keybind: keybind.all.session_delete?.[0],
          title: "delete",
          onTrigger: async (option) => {
            if (toDelete() === option.value) {
              const deleted = await api.fosra.session
                .delete({
                  sessionID: option.value,
                })
                .then(() => true)
                .catch(() => false);
              setToDelete(undefined);
              if (!deleted) {
                toast.show({
                  message: "Failed to delete session",
                  variant: "error",
                });
                return;
              }
              if (props.sessionId) {
                listedActions.mutate((sessions) =>
                  sessions?.filter(
                    (session) => session.session_id !== option.value,
                  ),
                );
                return;
              }
              return;
            }
            setToDelete(option.value);
          },
        },
        {
          keybind: keybind.all.session_rename?.[0],
          title: "rename",
          onTrigger: async (option) => {
            dialog.replace(() => (
              <DialogSessionRename session={option.value} />
            ));
          },
        },
      ]}
    />
  );
}
