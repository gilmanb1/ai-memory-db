"use client";

import { createContext, useContext, useState, useEffect, useCallback, type ReactNode } from "react";

interface ScopeItem {
  scope: string;
  display_name: string;
  count: number;
}

interface ScopeContextValue {
  selectedScope: string | null;  // null = all scopes
  setSelectedScope: (scope: string | null) => void;
  scopes: ScopeItem[];
  scopeParam: Record<string, string>;  // ready to spread into API params
}

const ScopeContext = createContext<ScopeContextValue>({
  selectedScope: null,
  setSelectedScope: () => {},
  scopes: [],
  scopeParam: {},
});

export function ScopeProvider({ children }: { children: ReactNode }) {
  const [selectedScope, setSelectedScope] = useState<string | null>(null);
  const [scopes, setScopes] = useState<ScopeItem[]>([]);

  // Fetch scopes on mount
  useEffect(() => {
    const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:9111";
    fetch(`${API_BASE}/api/v1/scopes`)
      .then((r) => r.json())
      .then((data) => setScopes(data.items || []))
      .catch(() => {});
  }, []);

  // Persist selection in localStorage
  useEffect(() => {
    const saved = localStorage.getItem("memory-dashboard-scope");
    if (saved) setSelectedScope(saved === "__all__" ? null : saved);
  }, []);

  const handleSetScope = useCallback((scope: string | null) => {
    setSelectedScope(scope);
    localStorage.setItem("memory-dashboard-scope", scope || "__all__");
  }, []);

  // Build param object for API calls
  const scopeParam: Record<string, string> = selectedScope ? { scope: selectedScope } : {};

  return (
    <ScopeContext value={{ selectedScope, setSelectedScope: handleSetScope, scopes, scopeParam }}>
      {children}
    </ScopeContext>
  );
}

export function useScope() {
  return useContext(ScopeContext);
}
