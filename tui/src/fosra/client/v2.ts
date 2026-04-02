import type {
  Event,
  Session,
  Message,
  Part,
  ProviderAuthAuthorization,
  ProviderListResponse,
} from "../sdk-types";
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
  experimental_workspaceID?: string;
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

    let res: Response;
    try {
      res = await fetchFn(`${baseUrl}${path}`, {
        method: init?.method ?? "GET",
        headers,
        body: init?.body,
        signal: init?.signal ?? signal,
      });
    } catch (e) {
      if (init?.throwOnError) throw new Error(String(e));
      return { error: { code: "NETWORK", message: String(e) } };
    }

    if (!res.ok) {
      const msg = await res.text().catch(() => res.statusText);
      if (init?.throwOnError) throw new Error(msg);
      return {
        error: { code: String(res.status), message: msg },
        response: res,
      };
    }

    const text = await res.text();
    if (!text) return { data: undefined as T, response: res };
    try {
      const data = JSON.parse(text) as T;
      return { data, response: res };
    } catch {
      if (init?.throwOnError)
        throw new Error(`JSON parse error: ${text.slice(0, 100)}`);
      return {
        error: {
          code: "PARSE",
          message: `JSON parse error: ${text.slice(0, 100)}`,
        },
        response: res,
      };
    }
  }

  function opts(o?: any): { throwOnError?: boolean; signal?: AbortSignal } {
    return { throwOnError: o?.throwOnError, signal: o?.signal };
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
            console.error("[SSE PARSE] res.body is null - SSE stream unavailable");
            return;
          }
          const reader = res.body.getReader();
          const decoder = new TextDecoder();
          let buffer = "";

          while (true) {
            const { done, value } = await reader.read();
            if (done) {
              console.debug("[SSE PARSE] Stream ended (done=true)");
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
                if (line.startsWith("data: ")) data += line.slice(6);
                else if (line.startsWith("data:")) data += line.slice(5);
              }
              if (data) {
                try {
                  yield JSON.parse(data.trim()) as Event;
                } catch (e) {
                  console.error("[SSE PARSE] JSON parse failed:", data.slice(0, 100), e);
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
        api<any>("/config/providers", opts(o)),
      get: async (_p?: any, o?: any) => api<any>("/config", opts(o)),
      update: async (p?: any, o?: any) =>
        api<any>("/config", {
          method: "PUT",
          body: JSON.stringify(p),
          ...opts(o),
        }),
    },
    provider: {
      list: async (_p?: any, o?: any) =>
        api<ProviderListResponse>("/provider", opts(o)),
      auth: async (_p?: any, o?: any) => api<any>("/provider/auth", opts(o)),
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
      agents: async (_p?: any, o?: any) => api<any[]>("/agent", opts(o)),
      skills: async (_p?: any, _o?: any) => mock<any[]>([]),
    },
    session: {
      list: async (p?: any, o?: any) => {
        const query = p?.start ? `?start=${p.start}` : "";
        return api<Session[]>(`/session${query}`, opts(o));
      },
      get: async (p: any, o?: any) =>
        api<Session>(`/session/${p.sessionID}`, opts(o)),
      create: async (p?: any, o?: any) =>
        api<Session>("/session", {
          method: "POST",
          body: JSON.stringify(p),
          ...opts(o),
        }),
      messages: async (p: any, o?: any) => {
        const query = p?.limit ? `?limit=${p.limit}` : "";
        return api<Array<{ info: Message; parts: Part[] }>>(
          `/session/${p.sessionID}/message${query}`,
          opts(o),
        );
      },
      todo: async (p: any, o?: any) =>
        api<any[]>(`/session/${p.sessionID}/todo`, opts(o)),
      diff: async (p: any, o?: any) =>
        api<any[]>(`/session/${p.sessionID}/diff`, opts(o)),
      status: async (_p?: any, o?: any) => api<any>("/session/status", opts(o)),
      abort: async (p: any, o?: any) =>
        api<boolean>(`/session/${p.sessionID}/abort`, {
          method: "POST",
          ...opts(o),
        }),
      summarize: async (p: any, o?: any) =>
        api<{ title: string }>(`/session/${p.sessionID}/summarize`, {
          method: "POST",
          ...opts(o),
        }),
      revert: async (p: any, o?: any) =>
        api<{ ok: boolean }>(`/session/${p.sessionID}/revert`, {
          method: "POST",
          ...opts(o),
        }),
      unrevert: async (p: any, o?: any) =>
        api<{ ok: boolean }>(`/session/${p.sessionID}/unrevert`, {
          method: "POST",
          ...opts(o),
        }),
      fork: async (p: any, o?: any) => {
        const { sessionID, ...body } = p;
        return api<Session>(`/session/${sessionID}/fork`, {
          method: "POST",
          body: JSON.stringify(body),
          ...opts(o),
        });
      },
      shell: async (p: any, o?: any) => {
        const { sessionID, ...body } = p;
        return api<any>(`/session/${sessionID}/shell`, {
          method: "POST",
          body: JSON.stringify(body),
          ...opts(o),
        });
      },
      command: async (p: any, o?: any) => {
        const { sessionID, ...body } = p;
        return api<any>(`/session/${sessionID}/command`, {
          method: "POST",
          body: JSON.stringify(body),
          ...opts(o),
        });
      },
      update: async (p: any, o?: any) => {
        const { sessionID, ...body } = p;
        return api<Session>(`/session/${sessionID}`, {
          method: "PUT",
          body: JSON.stringify(body),
          ...opts(o),
        });
      },
      delete: async (p: any, o?: any) =>
        api<boolean>(`/session/${p.sessionID}`, {
          method: "DELETE",
          ...opts(o),
        }),
      prompt: async (p: any, o?: any) => {
        const { sessionID, ...body } = p;
        return api<any>(`/session/${sessionID}/prompt`, {
          method: "POST",
          body: JSON.stringify(body),
          ...opts(o),
        });
      },
      share: async (p: any, o?: any) =>
        api<{ url: string }>(`/session/${p.sessionID}/share`, {
          method: "POST",
          ...opts(o),
        }),
      unshare: async (p: any, o?: any) =>
        api<{}>(`/session/${p.sessionID}/share`, {
          method: "DELETE",
          ...opts(o),
        }),
    },
    command: {
      list: async (_p?: any, o?: any) => api<any[]>("/command", opts(o)),
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
      status: async (_p?: any, o?: any) => api<any[]>("/lsp/status", opts(o)),
    },
    mcp: {
      status: async (_p?: any, o?: any) => api<any>("/mcp/status", opts(o)),
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
      workspace: {
        list: async (p: any, o?: any) =>
          api<any[]>("/experimental/workspace", { method: "GET", ...opts(o) }),
        create: async (p: any, o?: any) =>
          api<{}>("/experimental/workspace", {
            method: "POST",
            body: JSON.stringify(p),
            ...opts(o),
          }),
        remove: async (p: any, o?: any) =>
          api<boolean>(`/experimental/workspace/${p.id}`, {
            method: "DELETE",
            ...opts(o),
          }),
      },
    },
    formatter: {
      status: async (_p?: any, o?: any) =>
        api<any[]>("/formatter/status", opts(o)),
    },
    vcs: {
      get: async (_p?: any, o?: any) => api<any>("/vcs", opts(o)),
    },
    path: {
      get: async (_p?: any, o?: any) => api<any>("/path", opts(o)),
    },
    instance: {
      dispose: async (p: any, o?: any) =>
        api<boolean>("/instance/dispose", { method: "POST", ...opts(o) }),
    },
    find: {
      files: async (p: any, o?: any) =>
        api<any[]>("/find/file", { method: "GET", ...opts(o) }),
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
