import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter } from "react-router";
import "@fontsource-variable/inter/wght.css";
import "@fontsource-variable/manrope/wght.css";
import { SessionProvider, ApiError } from "./api";
import { App } from "./App";
import "./styles.css";

const queries = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1500,
      refetchOnWindowFocus: false,
      retry: (count, error) => count < 1 && (!(error instanceof ApiError) || error.status >= 500),
    },
  },
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queries}>
      <SessionProvider>
        <BrowserRouter><App /></BrowserRouter>
      </SessionProvider>
    </QueryClientProvider>
  </StrictMode>,
);
