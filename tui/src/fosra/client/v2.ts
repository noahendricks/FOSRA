import type {
  Event,
  Session,
  Message,
  Part,
  ProviderAuthAuthorization,
  ProviderListResponse,
  SessionSummarizeResponse,
  SessionRevertResponse,
  SessionUnrevertResponse,
  SessionShareResponse,
  SessionUnshareResponse,
  ConfigProvidersResponse,
  ConfigGetResponse,
  ConfigUpdateResponse,
  ProviderAuthResponse,
  AppAgentsResponse,
  SessionTodoResponse,
  SessionDiffResponse,
  SessionStatusResponse,
  SessionShellResponse,
  SessionCommandResponse,
  SessionPromptResponse,
  CommandListResponse,
  LspStatusResponse,
  McpStatusResponse,
  FormatterStatusResponse,
  VcsGetResponse,
  PathGetResponse,
  FindFilesResponse,
} from "../sdk-types";
import { log } from "../util/log";
export * from "../sdk-types";

type Result<T> = {
  data?: T;
  error?: { code: string; message: string };
  response?: Response;
};
type StreamResult<T> = {
  stream: AsyncIterable<T>;
  error?: { code: string; message: string };
  response?: Response;
};

export const createFosraClient = (options: {
  baseUrl?: string;
  signal?: AbortSignal;
  directory?: string;
  fetch?: typeof fetch;
  headers?: Record<string, string>;
}) => {
  const baseUrl = options.baseUrl ?? "http://localhost:8000/oc";
  const fetchFn = options.fetch ?? globalThis.fetch;
  const signal = options.signal;
  const baseHeaders = options.headers ?? {};

  async function api<T>(
    path: string,
    init?: {
      method?: string;
      body?: string;
      throwOnError?: boolean;
      signal?: AbortSignal;
    },
  ): Promise<Result<T>> {
    const headers: Record<string, string> = { ...baseHeaders };
    if (init?.body) headers["Content-Type"] = "application/json";
    const start = Date.now();

    let res: Response;
    try {
      res = await fetchFn(`${baseUrl}${path}`, {
        method: init?.method ?? "GET",
        headers,
        body: init?.body,
        signal: init?.signal ?? signal,
      });
    } catch (e) {
      log.api.error("API_FETCH_ERROR", {
        method: init?.method ?? "GET",
        path,
        error: String(e),
      });
      if (init?.throwOnError) throw new Error(String(e));
      return { error: { code: "NETWORK", message: String(e) } };
    }

    if (!res.ok) {
      const msg = await res.text().catch(() => res.statusText);
      log.api.debug("API_RESPONSE", {
        method: init?.method ?? "GET",
        path,
        status: res.status,
        durationMs: Date.now() - start,
        bodyPreview: msg.slice(0, 200),
      });
      if (init?.throwOnError) throw new Error(msg);
      return {
        error: { code: String(res.status), message: msg },
        response: res,
      };
    }

    const text = await res.text();
    if (!text) {
      log.api.debug("API_RESPONSE", {
        method: init?.method ?? "GET",
        path,
        status: res.status,
        durationMs: Date.now() - start,
        bodyEmpty: true,
      });
      return { data: undefined as T, response: res };
    }
    try {
      const data = JSON.parse(text) as T;
      log.api.debug("API_RESPONSE", {
        method: init?.method ?? "GET",
        path,
        status: res.status,
        durationMs: Date.now() - start,
        dataKeys: Object.keys(data as object),
      });
      return { data, response: res };
    } catch {
      const parseError = `JSON parse error: ${text.slice(0, 100)}`;
      log.api.error("API_PARSE_ERROR", {
        method: init?.method ?? "GET",
        path,
        error: parseError,
      });
      if (init?.throwOnError) throw new Error(parseError);
      return {
        error: {
          code: "PARSE",
          message: parseError,
        },
        response: res,
      };
    }
  }

  function opts(o?: { throwOnError?: boolean; signal?: AbortSignal }): {
    throwOnError?: boolean;
    signal?: AbortSignal;
  } {
    return {
      ...(o?.throwOnError !== undefined && { throwOnError: o.throwOnError }),
      ...(o?.signal !== undefined && { signal: o.signal }),
    };
  }

  function queryString(params: Record<string, unknown> = {}): string {
    const entries = Object.entries(params).filter(([, v]) => v !== undefined);
    if (entries.length === 0) return "";
    return "?" + new URLSearchParams(entries as [string, string][]).toString();
  }

  function sessionMethod<T>(
    sessionID: string,
    path: string,
    method: string,
    body: Record<string, unknown>,
    o?: { throwOnError?: boolean; signal?: AbortSignal },
  ): Promise<Result<T>> {
    return api<T>(`/session/${sessionID}${path}`, {
      method,
      body: JSON.stringify(body),
      ...opts(o),
    });
  }

  // stub for endpoints the backend doesn't support
  function mock<T>(data: T): Result<T> {
    return { data, response: new Response() };
  }

  return {
    event: {
      subscribe: async (
        _params?: any,
        _opts?: any,
      ): Promise<StreamResult<Event>> => {
        const res = await fetchFn(`${baseUrl}/event`, {
          headers: { ...baseHeaders, Accept: "text/event-stream" },
          signal: _opts?.signal ?? signal,
        });

        async function* parseSSE(): AsyncGenerator<Event> {
          if (!res.body) {
            log.sse.error("SSE_BODY_NULL", {});
            return;
          }
          const reader = res.body.getReader();
          const decoder = new TextDecoder();
          let buffer = "";

          while (true) {
            const { done, value } = await reader.read();
            if (done) {
              break;
            }
            buffer += decoder.decode(value, { stream: true });

            while (true) {
              const idx = buffer.indexOf("\n\n");
              if (idx === -1) break;
              const block = buffer.slice(0, idx);
              buffer = buffer.slice(idx + 2);

              let data = "";
              for (const line of block.split("\n")) {
                // skip SSE comments (lines starting with :)
                if (line.startsWith(":")) continue;
                // skip other SSE fields (event:, id:, retry:)
                if (!line.startsWith("data:")) continue;
                // extract data content
                data += line.startsWith("data: ")
                  ? line.slice(6)
                  : line.slice(5);
                data += "\n";
              }
              data = data.trim();
              if (data) {
                try {
                  yield JSON.parse(data.trim()) as Event;
                } catch (e) {
                  log.sse.error("SSE_JSON_PARSE_ERROR", {
                    dataPreview: data.slice(0, 200),
                    error: String(e),
                  });
                }
              }
            }
          }
        }

        return { stream: parseSSE(), response: res };
      },
    },
    config: {
      providers: async (_p?: any, o?: any) =>
        api<ConfigProvidersResponse>("/config/providers", opts(o)),
      get: async (_p?: any, o?: any) =>
        api<ConfigGetResponse>("/config", opts(o)),
      update: async (p?: any, o?: any) =>
        api<ConfigUpdateResponse>("/config", {
          method: "PUT",
          body: JSON.stringify(p),
          ...opts(o),
        }),
    },
    provider: {
      list: async (_p?: any, o?: any) =>
        api<ProviderListResponse>("/provider", opts(o)),
      auth: async (_p?: any, o?: any) =>
        api<ProviderAuthResponse>("/provider/auth", opts(o)),
      oauth: {
        authorize: async (_p?: any, _o?: any) =>
          mock<ProviderAuthAuthorization>({
            url: "",
            method: "auto",
            instructions: "",
          }),
        callback: async (_p?: any, _o?: any) => mock(true),
      },
    },
    app: {
      agents: async (_p?: any, o?: any) =>
        api<AppAgentsResponse>("/agent", opts(o)),
      skills: async (_p?: any, _o?: any) => mock<AppAgentsResponse>([]),
    },
    session: {
      list: async (p?: any, o?: any) =>
        api<Session[]>(`/session${queryString(p ?? {})}`, opts(o)),
      get: async (p: any, o?: any) =>
        api<Session>(`/session/${p.sessionID}`, opts(o)),
      create: async (p?: any, o?: any) =>
        api<Session>("/session", {
          method: "POST",
          body: JSON.stringify(p),
          ...opts(o),
        }),
      messages: async (p: any, o?: any) =>
        api<Array<{ info: Message; parts: Part[] }>>(
          `/session/${p.sessionID}/message${queryString(p ?? {})}`,
          opts(o),
        ),
      todo: async (p: any, o?: any) =>
        api<SessionTodoResponse>(`/session/${p.sessionID}/todo`, opts(o)),
      diff: async (p: any, o?: any) =>
        api<SessionDiffResponse>(`/session/${p.sessionID}/diff`, opts(o)),
      status: async (_p?: any, o?: any) =>
        api<SessionStatusResponse>("/session/status", opts(o)),
      abort: async (p: any, o?: any) =>
        api<boolean>(`/session/${p.sessionID}/abort`, {
          method: "POST",
          ...opts(o),
        }),
      summarize: async (p: any, o?: any) =>
        api<SessionSummarizeResponse>(`/session/${p.sessionID}/summarize`, {
          method: "POST",
          ...opts(o),
        }),
      revert: async (p: any, o?: any) =>
        api<SessionRevertResponse>(`/session/${p.sessionID}/revert`, {
          method: "POST",
          ...opts(o),
        }),
      unrevert: async (p: any, o?: any) =>
        api<SessionUnrevertResponse>(`/session/${p.sessionID}/unrevert`, {
          method: "POST",
          ...opts(o),
        }),
      fork: async (p: any, o?: any) => {
        const { sessionID, ...body } = p;
        return sessionMethod<Session>(sessionID, "/fork", "POST", body, o);
      },
      shell: async (p: any, o?: any) => {
        const { sessionID, ...body } = p;
        return sessionMethod<SessionShellResponse>(
          sessionID,
          "/shell",
          "POST",
          body,
          o,
        );
      },
      command: async (p: any, o?: any) => {
        const { sessionID, ...body } = p;
        return sessionMethod<SessionCommandResponse>(
          sessionID,
          "/command",
          "POST",
          body,
          o,
        );
      },
      update: async (p: any, o?: any) => {
        const { sessionID, ...body } = p;
        return sessionMethod<Session>(sessionID, "", "PUT", body, o);
      },
      delete: async (p: any, o?: any) =>
        api<boolean>(`/session/${p.sessionID}`, {
          method: "DELETE",
          ...opts(o),
        }),
      prompt: async (p: any, o?: any) => {
        const { sessionID, ...body } = p;
        return sessionMethod<SessionPromptResponse>(
          sessionID,
          "/message",
          "POST",
          body,
          o,
        );
      },
      share: async (p: any, o?: any) =>
        api<SessionShareResponse>(`/session/${p.sessionID}/share`, {
          method: "POST",
          ...opts(o),
        }),
      unshare: async (p: any, o?: any) =>
        api<SessionUnshareResponse>(`/session/${p.sessionID}/share`, {
          method: "DELETE",
          ...opts(o),
        }),
    },
    command: {
      list: async (_p?: any, o?: any) =>
        api<CommandListResponse>("/command", opts(o)),
    },
    permission: {
      reply: async (p: any, o?: any) =>
        api<boolean>(`/permission/${p.requestID}/reply`, {
          method: "POST",
          body: JSON.stringify(p),
          ...opts(o),
        }),
    },
    question: {
      reply: async (p: any, o?: any) =>
        api<boolean>(`/question/${p.requestID}/reply`, {
          method: "POST",
          body: JSON.stringify(p),
          ...opts(o),
        }),
      reject: async (p: any, o?: any) =>
        api<boolean>(`/question/${p.requestID}/reject`, {
          method: "POST",
          body: JSON.stringify(p),
          ...opts(o),
        }),
    },
    lsp: {
      status: async (_p?: any, o?: any) =>
        api<LspStatusResponse>("/lsp/status", opts(o)),
    },
    mcp: {
      status: async (_p?: any, o?: any) =>
        api<McpStatusResponse>("/mcp/status", opts(o)),
      connect: async (p: any, o?: any) =>
        api<{}>(`/mcp/${p.name}/connect`, { method: "POST", ...opts(o) }),
      disconnect: async (p: any, o?: any) =>
        api<boolean>(`/mcp/${p.name}/disconnect`, {
          method: "POST",
          ...opts(o),
        }),
    },
    experimental: {
      resource: {
        list: async (p: any, o?: any) =>
          api<{}>("/experimental/resource", { method: "GET", ...opts(o) }),
      },
    },
    formatter: {
      status: async (_p?: any, o?: any) =>
        api<FormatterStatusResponse>("/formatter/status", opts(o)),
    },
    vcs: {
      get: async (_p?: any, o?: any) => api<VcsGetResponse>("/vcs", opts(o)),
    },
    path: {
      get: async (_p?: any, o?: any) => api<PathGetResponse>("/path", opts(o)),
    },
    instance: {
      dispose: async (p: any, o?: any) =>
        api<boolean>("/instance/dispose", { method: "POST", ...opts(o) }),
    },
    find: {
      files: async (p: any, o?: any) =>
        api<FindFilesResponse>("/find/file", { method: "GET", ...opts(o) }),
    },
    auth: {
      set: async (p?: any, o?: any) =>
        api<boolean>("/provider/auth", {
          method: "POST",
          body: JSON.stringify(p),
          ...opts(o),
        }),
    },
  };
};
