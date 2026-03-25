const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:9111";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });
  if (!res.ok) {
    const msg = await res.text().catch(() => res.statusText);
    throw new Error(`API ${res.status}: ${msg}`);
  }
  return res.json();
}

export const api = {
  // Stats
  getStats: () => request<any>("/api/v1/stats"),

  // Scopes
  getScopes: () => request<any>("/api/v1/scopes"),

  // Facts
  getFacts: (params?: Record<string, string>) => {
    const qs = params ? "?" + new URLSearchParams(params).toString() : "";
    return request<any>(`/api/v1/facts${qs}`);
  },
  getFact: (id: string) => request<any>(`/api/v1/facts/${id}`),
  createFact: (data: any) =>
    request<any>("/api/v1/facts", { method: "POST", body: JSON.stringify(data) }),
  updateFact: (id: string, data: any) =>
    request<any>(`/api/v1/facts/${id}`, { method: "PUT", body: JSON.stringify(data) }),
  deleteFact: (id: string) =>
    request<any>(`/api/v1/facts/${id}`, { method: "DELETE" }),

  // Decisions
  getDecisions: (params?: Record<string, string>) => {
    const qs = params ? "?" + new URLSearchParams(params).toString() : "";
    return request<any>(`/api/v1/decisions${qs}`);
  },
  createDecision: (data: any) =>
    request<any>("/api/v1/decisions", { method: "POST", body: JSON.stringify(data) }),
  updateDecision: (id: string, data: any) =>
    request<any>(`/api/v1/decisions/${id}`, { method: "PUT", body: JSON.stringify(data) }),
  deleteDecision: (id: string) =>
    request<any>(`/api/v1/decisions/${id}`, { method: "DELETE" }),

  // Entities
  getEntities: (params?: Record<string, string>) => {
    const qs = params ? "?" + new URLSearchParams(params).toString() : "";
    return request<any>(`/api/v1/entities${qs}`);
  },
  createEntity: (data: any) =>
    request<any>("/api/v1/entities", { method: "POST", body: JSON.stringify(data) }),
  updateEntity: (id: string, data: any) =>
    request<any>(`/api/v1/entities/${id}`, { method: "PUT", body: JSON.stringify(data) }),
  deleteEntity: (id: string) =>
    request<any>(`/api/v1/entities/${id}`, { method: "DELETE" }),

  // Relationships
  getRelationships: (params?: Record<string, string>) => {
    const qs = params ? "?" + new URLSearchParams(params).toString() : "";
    return request<any>(`/api/v1/relationships${qs}`);
  },
  getGraph: (params?: Record<string, string>) => {
    const qs = params ? "?" + new URLSearchParams(params).toString() : "";
    return request<any>(`/api/v1/relationships/graph${qs}`);
  },
  createRelationship: (data: any) =>
    request<any>("/api/v1/relationships", { method: "POST", body: JSON.stringify(data) }),
  updateRelationship: (id: string, data: any) =>
    request<any>(`/api/v1/relationships/${id}`, { method: "PUT", body: JSON.stringify(data) }),
  deleteRelationship: (id: string) =>
    request<any>(`/api/v1/relationships/${id}`, { method: "DELETE" }),

  // Guardrails
  getGuardrails: (params?: Record<string, string>) => {
    const qs = params ? "?" + new URLSearchParams(params).toString() : "";
    return request<any>(`/api/v1/guardrails${qs}`);
  },
  createGuardrail: (data: any) =>
    request<any>("/api/v1/guardrails", { method: "POST", body: JSON.stringify(data) }),
  updateGuardrail: (id: string, data: any) =>
    request<any>(`/api/v1/guardrails/${id}`, { method: "PUT", body: JSON.stringify(data) }),
  deleteGuardrail: (id: string) =>
    request<any>(`/api/v1/guardrails/${id}`, { method: "DELETE" }),

  // Procedures
  getProcedures: (params?: Record<string, string>) => {
    const qs = params ? "?" + new URLSearchParams(params).toString() : "";
    return request<any>(`/api/v1/procedures${qs}`);
  },
  createProcedure: (data: any) =>
    request<any>("/api/v1/procedures", { method: "POST", body: JSON.stringify(data) }),
  updateProcedure: (id: string, data: any) =>
    request<any>(`/api/v1/procedures/${id}`, { method: "PUT", body: JSON.stringify(data) }),
  deleteProcedure: (id: string) =>
    request<any>(`/api/v1/procedures/${id}`, { method: "DELETE" }),

  // Error Solutions
  getErrorSolutions: (params?: Record<string, string>) => {
    const qs = params ? "?" + new URLSearchParams(params).toString() : "";
    return request<any>(`/api/v1/error_solutions${qs}`);
  },
  createErrorSolution: (data: any) =>
    request<any>("/api/v1/error_solutions", { method: "POST", body: JSON.stringify(data) }),
  updateErrorSolution: (id: string, data: any) =>
    request<any>(`/api/v1/error_solutions/${id}`, { method: "PUT", body: JSON.stringify(data) }),
  deleteErrorSolution: (id: string) =>
    request<any>(`/api/v1/error_solutions/${id}`, { method: "DELETE" }),

  // Observations
  getObservations: (params?: Record<string, string>) => {
    const qs = params ? "?" + new URLSearchParams(params).toString() : "";
    return request<any>(`/api/v1/observations${qs}`);
  },
  deleteObservation: (id: string) =>
    request<any>(`/api/v1/observations/${id}`, { method: "DELETE" }),

  // Sessions
  getSessions: (params?: Record<string, string>) => {
    const qs = params ? "?" + new URLSearchParams(params).toString() : "";
    return request<any>(`/api/v1/sessions${qs}`);
  },

  // Ideas
  getIdeas: (params?: Record<string, string>) => {
    const qs = params ? "?" + new URLSearchParams(params).toString() : "";
    return request<any>(`/api/v1/ideas${qs}`);
  },
  createIdea: (data: any) =>
    request<any>("/api/v1/ideas", { method: "POST", body: JSON.stringify(data) }),
  updateIdea: (id: string, data: any) =>
    request<any>(`/api/v1/ideas/${id}`, { method: "PUT", body: JSON.stringify(data) }),
  deleteIdea: (id: string) =>
    request<any>(`/api/v1/ideas/${id}`, { method: "DELETE" }),

  // Questions
  getQuestions: (params?: Record<string, string>) => {
    const qs = params ? "?" + new URLSearchParams(params).toString() : "";
    return request<any>(`/api/v1/questions${qs}`);
  },
  createQuestion: (data: any) =>
    request<any>("/api/v1/questions", { method: "POST", body: JSON.stringify(data) }),
  updateQuestion: (id: string, data: any) =>
    request<any>(`/api/v1/questions/${id}`, { method: "PUT", body: JSON.stringify(data) }),
  deleteQuestion: (id: string) =>
    request<any>(`/api/v1/questions/${id}`, { method: "DELETE" }),

  // Search
  search: (data: { query: string; types?: string[]; scope?: string; limit?: number }) =>
    request<any>("/api/v1/search", { method: "POST", body: JSON.stringify(data) }),
};
