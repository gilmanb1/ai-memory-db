"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Terminal, Globe, Copy, Check } from "lucide-react";

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      onClick={() => { navigator.clipboard.writeText(text); setCopied(true); setTimeout(() => setCopied(false), 1500); }}
      className="absolute top-2 right-2 p-1 rounded bg-white/10 hover:bg-white/20 transition-colors"
      title="Copy"
    >
      {copied ? <Check className="w-3 h-3 text-green-400" /> : <Copy className="w-3 h-3 text-muted-foreground" />}
    </button>
  );
}

function CodeBlock({ children, language = "" }: { children: string; language?: string }) {
  return (
    <div className="relative">
      <CopyButton text={children} />
      <pre className="p-3 rounded-lg bg-black/40 border border-white/10 text-[12px] font-mono text-emerald-200 overflow-x-auto">
        {children}
      </pre>
    </div>
  );
}

const MCP_TOOLS = [
  {
    name: "memory_search",
    desc: "Search all memory types by semantic query",
    params: [
      { name: "query", type: "string", required: true, desc: "Search query text" },
      { name: "types", type: "string[]", required: false, desc: "Filter: facts, guardrails, procedures, error_solutions, decisions" },
      { name: "limit", type: "integer", required: false, desc: "Max results per type (default 10)" },
    ],
    example: `memory_search(query="retry logic", types=["facts", "guardrails"], limit=5)`,
  },
  {
    name: "memory_store",
    desc: "Store a fact, decision, guardrail, procedure, or error_solution",
    params: [
      { name: "text", type: "string", required: true, desc: "Content to store" },
      { name: "type", type: "string", required: true, desc: "One of: fact, decision, guardrail, procedure, error_solution" },
      { name: "importance", type: "integer", required: false, desc: "1-10 score (default 7)" },
      { name: "file_paths", type: "string[]", required: false, desc: "Associated file paths" },
    ],
    example: `memory_store(text="API uses gRPC", type="fact", importance=8)`,
  },
  {
    name: "memory_guardrail",
    desc: "Create a guardrail (protective rule for specific files)",
    params: [
      { name: "warning", type: "string", required: true, desc: "The guardrail warning text" },
      { name: "rationale", type: "string", required: false, desc: "Why this guardrail exists" },
      { name: "consequence", type: "string", required: false, desc: "What happens if violated" },
      { name: "file_paths", type: "string[]", required: false, desc: "Files this applies to" },
    ],
    example: `memory_guardrail(warning="Don't modify auth.py", file_paths=["src/auth.py"])`,
  },
  {
    name: "memory_check_file",
    desc: "Get all memory associated with a file path",
    params: [
      { name: "file_path", type: "string", required: true, desc: "File path to check" },
    ],
    example: `memory_check_file(file_path="src/auth.py")`,
  },
];

const REST_ENDPOINTS = [
  { method: "GET", path: "/api/v1/facts", desc: "List facts", params: "scope, temporal_class, limit, offset" },
  { method: "GET", path: "/api/v1/decisions", desc: "List decisions", params: "scope, limit, offset" },
  { method: "GET", path: "/api/v1/entities", desc: "List entities", params: "scope, limit, offset" },
  { method: "GET", path: "/api/v1/relationships", desc: "List relationships", params: "scope, limit, offset" },
  { method: "GET", path: "/api/v1/relationships/graph", desc: "Cytoscape graph data", params: "scope, limit" },
  { method: "GET", path: "/api/v1/guardrails", desc: "List guardrails", params: "scope, limit, offset" },
  { method: "GET", path: "/api/v1/procedures", desc: "List procedures", params: "scope, limit, offset" },
  { method: "GET", path: "/api/v1/error-solutions", desc: "List error solutions", params: "scope, limit, offset" },
  { method: "GET", path: "/api/v1/observations", desc: "List observations", params: "scope, limit, offset" },
  { method: "GET", path: "/api/v1/sessions", desc: "List sessions", params: "limit, offset" },
  { method: "GET", path: "/api/v1/scopes", desc: "List scopes with counts", params: "" },
  { method: "GET", path: "/api/v1/stats", desc: "Database statistics", params: "" },
  { method: "GET", path: "/api/v1/knowledge-graph", desc: "Unified knowledge graph", params: "types, scope, limit, cluster_by" },
  { method: "GET", path: "/api/v1/code-graph/graph", desc: "Code dependency graph", params: "scope, limit" },
  { method: "GET", path: "/api/v1/code-graph/stats", desc: "Code graph statistics", params: "" },
  { method: "POST", path: "/api/v1/search", desc: "Semantic search", params: "query, types, scope, limit (body)" },
  { method: "POST", path: "/api/v1/chat/stream", desc: "Streaming chat (NDJSON)", params: "query, scope, conversation_history (body)" },
  { method: "GET", path: "/api/v1/review", desc: "Review queue", params: "status, limit" },
  { method: "POST", path: "/api/v1/review/{id}/approve", desc: "Approve review item", params: "" },
  { method: "POST", path: "/api/v1/review/{id}/reject", desc: "Reject review item", params: "" },
];

export default function ApiDocsPage() {
  const [tab, setTab] = useState<"mcp" | "rest">("mcp");

  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold mb-2">API Documentation</h1>
      <p className="text-sm text-muted-foreground mb-6">
        MCP tools for AI agents and REST API for the dashboard and integrations
      </p>

      <div className="flex gap-2 mb-6">
        <Button variant={tab === "mcp" ? "secondary" : "ghost"} size="sm" onClick={() => setTab("mcp")}>
          <Terminal className="w-4 h-4 mr-1.5" /> MCP Tools
        </Button>
        <Button variant={tab === "rest" ? "secondary" : "ghost"} size="sm" onClick={() => setTab("rest")}>
          <Globe className="w-4 h-4 mr-1.5" /> REST API
        </Button>
      </div>

      {tab === "mcp" && (
        <div className="space-y-6">
          <div className="p-4 rounded-lg border border-border bg-card/60">
            <h3 className="text-sm font-semibold mb-2">About the MCP Server</h3>
            <p className="text-xs text-muted-foreground leading-relaxed">
              The memory system includes an MCP (Model Context Protocol) server that gives any
              MCP-compatible AI tool on-demand read/write access to the knowledge base.
              It communicates via stdio using JSON-RPC 2.0. Compatible with Claude Code, Cursor,
              Windsurf, Cline, and other MCP-compatible agents. All share the same DuckDB database.
            </p>
          </div>

          {MCP_TOOLS.map((tool) => (
            <div key={tool.name} className="border border-border rounded-lg overflow-hidden">
              <div className="p-4 bg-card/80 border-b border-border">
                <h3 className="text-sm font-bold font-mono text-blue-400">{tool.name}</h3>
                <p className="text-xs text-muted-foreground mt-1">{tool.desc}</p>
              </div>
              <div className="p-4 space-y-3">
                <div>
                  <h4 className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold mb-2">Parameters</h4>
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="text-muted-foreground text-left">
                        <th className="pb-1 pr-3 font-medium">Name</th>
                        <th className="pb-1 pr-3 font-medium">Type</th>
                        <th className="pb-1 pr-3 font-medium">Required</th>
                        <th className="pb-1 font-medium">Description</th>
                      </tr>
                    </thead>
                    <tbody>
                      {tool.params.map((p) => (
                        <tr key={p.name} className="border-t border-border/50">
                          <td className="py-1.5 pr-3 font-mono text-emerald-300">{p.name}</td>
                          <td className="py-1.5 pr-3 text-muted-foreground">{p.type}</td>
                          <td className="py-1.5 pr-3">{p.required ? <span className="text-amber-400">yes</span> : <span className="text-muted-foreground">no</span>}</td>
                          <td className="py-1.5">{p.desc}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div>
                  <h4 className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold mb-1">Example</h4>
                  <CodeBlock>{tool.example}</CodeBlock>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {tab === "rest" && (
        <div className="space-y-4">
          <div className="p-4 rounded-lg border border-border bg-card/60">
            <h3 className="text-sm font-semibold mb-2">REST API</h3>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Base URL: <code className="px-1 py-0.5 rounded bg-white/10 text-emerald-300">http://localhost:9111</code>.
              All endpoints return JSON. List endpoints support <code className="px-1 py-0.5 rounded bg-white/10 text-emerald-300">scope</code> filtering
              and pagination via <code className="px-1 py-0.5 rounded bg-white/10 text-emerald-300">limit</code> / <code className="px-1 py-0.5 rounded bg-white/10 text-emerald-300">offset</code>.
              POST endpoints for CRUD accept JSON body. The chat endpoint streams NDJSON.
            </p>
          </div>

          <div className="border border-border rounded-lg overflow-hidden">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-muted-foreground text-left bg-card/80 border-b border-border">
                  <th className="p-3 font-medium w-16">Method</th>
                  <th className="p-3 font-medium">Endpoint</th>
                  <th className="p-3 font-medium">Description</th>
                  <th className="p-3 font-medium">Parameters</th>
                </tr>
              </thead>
              <tbody>
                {REST_ENDPOINTS.map((ep, i) => (
                  <tr key={i} className="border-t border-border/50 hover:bg-card/40 transition-colors">
                    <td className="p-3">
                      <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                        ep.method === "GET" ? "bg-green-500/20 text-green-400"
                        : "bg-blue-500/20 text-blue-400"
                      }`}>{ep.method}</span>
                    </td>
                    <td className="p-3 font-mono text-emerald-300">{ep.path}</td>
                    <td className="p-3">{ep.desc}</td>
                    <td className="p-3 text-muted-foreground">{ep.params || "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
