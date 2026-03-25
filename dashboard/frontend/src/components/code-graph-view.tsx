"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import cytoscape from "cytoscape";
import fcose from "cytoscape-fcose";
import { api } from "@/lib/api";
import { usePolling } from "@/hooks/use-polling";
import { X } from "lucide-react";

cytoscape.use(fcose);

const LANG_COLORS: Record<string, string> = {
  python: "#3b82f6",
  typescript: "#22c55e",
  javascript: "#eab308",
  go: "#06b6d4",
  rust: "#f97316",
  unknown: "#6b7280",
};

function getLangColor(language: string): string {
  return LANG_COLORS[language?.toLowerCase()] || LANG_COLORS.unknown;
}

function scaleSize(symbolCount: number): number {
  // Scale symbol_count to node size between 20 and 60
  const minSize = 20;
  const maxSize = 60;
  const clamped = Math.max(0, Math.min(symbolCount, 50));
  return minSize + (clamped / 50) * (maxSize - minSize);
}

interface SymbolRow {
  id: string;
  symbol_name: string;
  symbol_type: string;
  language: string;
  line_number: number;
  signature: string | null;
  docstring: string | null;
}

export function CodeGraphView() {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<cytoscape.Core | null>(null);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [symbols, setSymbols] = useState<SymbolRow[]>([]);
  const [loadingSymbols, setLoadingSymbols] = useState(false);

  const fetcher = useCallback(() => api.getCodeGraph(), []);
  const { data } = usePolling<{ nodes: any[]; edges: any[] }>(fetcher, 10000);

  // Initialize Cytoscape
  useEffect(() => {
    if (!containerRef.current) return;

    const cy = cytoscape({
      container: containerRef.current,
      style: [
        {
          selector: "node",
          style: {
            label: "data(label)",
            "text-valign": "bottom",
            "text-halign": "center",
            color: "#e5e7eb",
            "font-size": "9px",
            "text-outline-color": "#111827",
            "text-outline-width": 2,
            "background-color": "data(color)",
            width: "data(size)",
            height: "data(size)",
            "text-margin-y": 4,
          },
        },
        {
          selector: "edge",
          style: {
            label: "data(label)",
            "font-size": "7px",
            color: "#9ca3af",
            "text-outline-color": "#111827",
            "text-outline-width": 1,
            "line-color": "#4b5563",
            "target-arrow-color": "#6b7280",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
            width: 1.5,
            "arrow-scale": 0.8,
          },
        },
        {
          selector: "node:selected",
          style: {
            "border-width": 3,
            "border-color": "#f59e0b",
          },
        },
      ],
      layout: { name: "grid" },
      wheelSensitivity: 0.3,
    });

    cy.on("tap", "node", (evt) => {
      const nodeId = evt.target.id();
      setSelectedFile(nodeId);
    });

    cyRef.current = cy;

    return () => {
      cy.destroy();
      cyRef.current = null;
    };
  }, []);

  // Update graph data
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy || !data) return;

    const nodes = (data.nodes || []).map((n: any) => ({
      data: {
        id: n.id,
        label: n.label,
        color: getLangColor(n.language),
        size: scaleSize(n.symbol_count || 0),
      },
    }));

    const edges = (data.edges || []).map((e: any) => ({
      data: {
        id: e.id,
        source: e.source,
        target: e.target,
        label: e.import_name || "",
      },
    }));

    cy.json({ elements: { nodes, edges } });

    if (nodes.length > 0) {
      const layout = cy.layout({
        name: "fcose",
        animate: false,
        nodeDimensionsIncludeLabels: true,
        idealEdgeLength: 150,
        nodeRepulsion: 10000,
        padding: 30,
      } as any);
      layout.run();
    }
  }, [data]);

  // Fetch symbols when a file is selected
  useEffect(() => {
    if (!selectedFile) {
      setSymbols([]);
      return;
    }
    setLoadingSymbols(true);
    api
      .getCodeFileSymbols(selectedFile)
      .then((res: any) => setSymbols(res.items || []))
      .catch(() => setSymbols([]))
      .finally(() => setLoadingSymbols(false));
  }, [selectedFile]);

  // Compute dependencies and dependents from graph data
  const dependencies =
    data?.edges?.filter((e: any) => e.source === selectedFile) || [];
  const dependents =
    data?.edges?.filter((e: any) => e.target === selectedFile) || [];

  return (
    <div className="relative flex" style={{ height: "calc(100vh - 280px)" }}>
      {/* Graph area */}
      <div
        ref={containerRef}
        className="border rounded-md flex-1"
        style={{ backgroundColor: "#111827" }}
      />

      {/* Slide-out panel */}
      {selectedFile && (
        <div className="w-[380px] border-l border-border bg-card overflow-y-auto flex-shrink-0">
          <div className="flex items-center justify-between p-3 border-b border-border sticky top-0 bg-card z-10">
            <h3 className="text-sm font-semibold truncate" title={selectedFile}>
              {selectedFile.split("/").pop()}
            </h3>
            <button
              onClick={() => setSelectedFile(null)}
              className="p-1 hover:bg-accent rounded"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          <div className="p-3 text-xs text-muted-foreground border-b border-border">
            <span className="font-mono">{selectedFile}</span>
          </div>

          {/* Symbols section */}
          <div className="p-3 border-b border-border">
            <h4 className="text-xs font-semibold uppercase text-muted-foreground mb-2">
              Symbols ({symbols.length})
            </h4>
            {loadingSymbols ? (
              <div className="text-xs text-muted-foreground">Loading...</div>
            ) : symbols.length === 0 ? (
              <div className="text-xs text-muted-foreground">No symbols found</div>
            ) : (
              <div className="space-y-1 max-h-[300px] overflow-y-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="text-muted-foreground text-left">
                      <th className="pb-1 pr-2">Name</th>
                      <th className="pb-1 pr-2">Type</th>
                      <th className="pb-1 pr-2 text-right">Line</th>
                    </tr>
                  </thead>
                  <tbody>
                    {symbols.map((s) => (
                      <tr key={s.id} className="hover:bg-accent/30">
                        <td className="py-0.5 pr-2 font-mono truncate max-w-[160px]" title={s.signature || s.symbol_name}>
                          {s.symbol_name}
                        </td>
                        <td className="py-0.5 pr-2">
                          <span
                            className={`inline-block px-1 rounded text-[10px] ${
                              s.symbol_type === "function"
                                ? "bg-blue-500/20 text-blue-400"
                                : s.symbol_type === "class"
                                ? "bg-purple-500/20 text-purple-400"
                                : "bg-gray-500/20 text-gray-400"
                            }`}
                          >
                            {s.symbol_type}
                          </span>
                        </td>
                        <td className="py-0.5 text-right text-muted-foreground">
                          {s.line_number}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Dependencies section */}
          <div className="p-3 border-b border-border">
            <h4 className="text-xs font-semibold uppercase text-muted-foreground mb-2">
              Dependencies ({dependencies.length})
            </h4>
            {dependencies.length === 0 ? (
              <div className="text-xs text-muted-foreground">None</div>
            ) : (
              <ul className="space-y-1">
                {dependencies.map((d: any, i: number) => (
                  <li
                    key={i}
                    className="text-xs font-mono truncate cursor-pointer hover:text-blue-400"
                    title={d.target}
                    onClick={() => setSelectedFile(d.target)}
                  >
                    {d.target.split("/").pop()}
                    {d.import_name && (
                      <span className="text-muted-foreground ml-1">
                        ({d.import_name})
                      </span>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </div>

          {/* Dependents section */}
          <div className="p-3">
            <h4 className="text-xs font-semibold uppercase text-muted-foreground mb-2">
              Dependents ({dependents.length})
            </h4>
            {dependents.length === 0 ? (
              <div className="text-xs text-muted-foreground">None</div>
            ) : (
              <ul className="space-y-1">
                {dependents.map((d: any, i: number) => (
                  <li
                    key={i}
                    className="text-xs font-mono truncate cursor-pointer hover:text-blue-400"
                    title={d.source}
                    onClick={() => setSelectedFile(d.source)}
                  >
                    {d.source.split("/").pop()}
                    {d.import_name && (
                      <span className="text-muted-foreground ml-1">
                        ({d.import_name})
                      </span>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
