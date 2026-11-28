import { createContext, useContext, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { useInfiniteQuery, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { QueryKey } from "@tanstack/react-query";
import { useOutletContext } from "react-router";
import type { Failure, Job, Page } from "./models";

export const JOBS = "/api/v1/jobs";
const SESSION_KEY = "ats-workspace-token";

export const jobApi = (jobId: string) => `${JOBS}/${encodeURIComponent(jobId)}`;
export const jobUrl = (jobId: string) => `/jobs/${encodeURIComponent(jobId)}`;
export const jobKey = (jobId: string) => ["job", jobId] as const;
export const useJob = () => useOutletContext<Job>();

export class ApiError extends Error {
  constructor(public status: number, public failure: Failure) {
    super(failure.message);
  }
}

export interface Command {
  path: string;
  method: "POST" | "PUT";
  body: object | FormData;
}

function requestKey() {
  return Array.from(crypto.getRandomValues(new Uint8Array(16)), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function createApi(token: string, unauthorized?: () => void) {
  async function request(path: string, signal?: AbortSignal, command?: Command): Promise<Response> {
    const url = new URL(path, location.origin);
    if (url.origin !== location.origin || (url.pathname !== JOBS && !url.pathname.startsWith(`${JOBS}/`))) {
      throw new Error("The requested link is outside this workspace.");
    }
    const headers = new Headers({ Authorization: `Bearer ${token}`, Accept: "application/json" });
    const form = command?.body instanceof FormData;
    if (command && !form) headers.set("Content-Type", "application/json");
    if (command?.method === "POST") headers.set("Idempotency-Key", requestKey());
    const options: RequestInit = {
      method: command?.method ?? "GET",
      headers,
      body: command ? (form ? command.body as FormData : JSON.stringify(command.body)) : undefined,
      signal,
      redirect: "error",
      cache: "no-store",
    };
    let response: Response;
    try {
      response = await fetch(url, options);
    } catch (error) {
      if (signal?.aborted) throw error;
      // If the response was lost, retry the POST using the same idempotency key.
      try {
        if (command?.method !== "POST") throw error;
        response = await fetch(url, options);
      } catch {
        throw new Error("Could not reach the job API. Check your connection and try again.");
      }
    }
    if (response.status === 401) unauthorized?.();
    if (!response.ok) {
      const payload: Partial<Failure> = await response.json().catch(() => ({}));
      throw new ApiError(response.status, {
        code: payload.code ?? `HTTP_${response.status}`,
        message: payload.message ?? (response.status >= 500
          ? "The job API is unavailable. Check that the backend is running."
          : `The request could not be completed (${response.status}).`),
        retryable: payload.retryable ?? false,
        field_errors: payload.field_errors,
      });
    }
    return response;
  }
  async function json<T>(path: string, signal?: AbortSignal, command?: Command): Promise<T> {
    const response = await request(path, signal, command);
    if (!response.headers.get("Content-Type")?.includes("application/json")) {
      throw new Error("The workspace did not receive a JSON response. Check the API proxy configuration.");
    }
    return response.json() as Promise<T>;
  }
  return {
    read: <T,>(path: string, signal?: AbortSignal) => json<T>(path, signal),
    write: <T,>(command: Command) => json<T>(command.path, undefined, command),
    download: async (path: string, signal?: AbortSignal) => (await request(path, signal)).blob(),
  };
}

function storedToken() {
  try { return sessionStorage.getItem(SESSION_KEY) ?? ""; } catch { return ""; }
}

interface Session {
  connected: boolean;
  expired: boolean;
  connect: (value: string) => Promise<void>;
  disconnect: () => void;
  api: ReturnType<typeof createApi>;
}

const SessionContext = createContext<Session | null>(null);

export function SessionProvider({ children }: { children: ReactNode }) {
  const cache = useQueryClient();
  const [token, setToken] = useState(storedToken);
  const current = useRef(token);
  const [expired, setExpired] = useState(false);
  const session = useMemo(() => {
    function setSession(next: string, wasExpired = false) {
      current.current = next;
      try {
        if (next) sessionStorage.setItem(SESSION_KEY, next);
        else sessionStorage.removeItem(SESSION_KEY);
      } catch { /* This tab can still use an in-memory session. */ }
      cache.clear();
      setExpired(wasExpired);
      setToken(next);
    }
    return {
      connected: Boolean(token),
      expired,
      async connect(value: string) {
        const next = value.trim();
        await createApi(next).read<Page<unknown>>(`${JOBS}?limit=1`);
        setSession(next);
      },
      disconnect: () => setSession(""),
      api: createApi(token, () => {
        if (token && token === current.current) setSession("", true);
      }),
    };
  }, [cache, expired, token]);
  return <SessionContext.Provider value={session}>{children}</SessionContext.Provider>;
}

export function useSession(): Session {
  const session = useContext(SessionContext);
  if (!session) throw new Error("Missing workspace session.");
  return session;
}

export function useRead<T>(key: QueryKey, path: string | null, poll?: (data: T | undefined) => number | false) {
  const { api } = useSession();
  return useQuery<T>({
    queryKey: key,
    enabled: path !== null,
    queryFn: ({ signal }) => api.read<T>(path!, signal),
    refetchInterval: poll ? (query) => query.state.status === "error" ? false : poll(query.state.data) : false,
  });
}

export function useCollection<T>(key: QueryKey, path: string | null, poll = false) {
  const { api } = useSession();
  const query = useInfiniteQuery({
    queryKey: key,
    enabled: path !== null,
    initialPageParam: null as string | null,
    queryFn: ({ pageParam, signal }) => {
      const url = new URL(path!, location.origin);
      url.searchParams.set("limit", "25");
      if (pageParam != null) url.searchParams.set("cursor", pageParam);
      return api.read<Page<T>>(`${url.pathname}${url.search}`, signal);
    },
    getNextPageParam: (page) => page.next_cursor ?? undefined,
    refetchInterval: poll ? (query) => query.state.status === "error" ? false : 3500 : false,
  });
  return { ...query, items: query.data?.pages.flatMap((page) => page.items) ?? [] };
}

export function useCommand(jobId?: string) {
  const cache = useQueryClient();
  const { api } = useSession();
  const mutation = useMutation({
    mutationFn: (command: Command) => api.write<unknown>(command),
  });
  async function submit<T>(method: Command["method"], path: string, body: Command["body"], refresh = true) {
    try {
      const result = await mutation.mutateAsync({ method, path, body }) as T;
      if (refresh) {
        void cache.invalidateQueries({ queryKey: ["jobs"] });
        if (jobId) void cache.invalidateQueries({ queryKey: jobKey(jobId) });
      }
      return result;
    } catch {
      return undefined;
    }
  }
  return { submit, error: mutation.error, pending: mutation.isPending, reset: mutation.reset };
}
