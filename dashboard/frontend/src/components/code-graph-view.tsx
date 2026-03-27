"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import cytoscape from "cytoscape";
import fcose from "cytoscape-fcose";
import { api } from "@/lib/api";
import { usePolling } from "@/hooks/use-polling";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  X,
  ZoomIn,
  ZoomOut,
  Maximize,
  Download,
  Search,
} from "lucide-react";

cytoscape.use(fcose);

const LANG_COLORS: Record<string, string> = {
  python: "#3b82f6",
  typescript: "#22c55e",
  javascript: "#eab308",
  go: "#06b6d4",
  rust: "#f97316",
  unknown: "#6b7280",
};

const EDGE_COLORS: Record<string, string> = {
  internal: "#34d399",
  external: "#fb923c",
};

const EDGE_WIDTHS: Record<string, number> = {
  internal: 2,
  external: 1.5,
};

const LAYOUTS: Record<string, any> = {
  fcose: {
    name: "fcose",
    animate: true,
    animationDuration: 400,
    nodeDimensionsIncludeLabels: true,
    idealEdgeLength: 150,
    nodeRepulsion: 10000,
    padding: 30,
  },
  circle: { name: "circle", animate: true, animationDuration: 400, padding: 30 },
  concentric: {
    name: "concentric",
    animate: true,
    animationDuration: 400,
    padding: 30,
    concentric: (node: any) => node.data("size") || 20,
    levelWidth: () => 2,
  },
  breadthfirst: {
    name: "breadthfirst",
    animate: true,
    animationDuration: 400,
    padding: 30,
    directed: true,
    spacingFactor: 1.2,
  },
  grid: { name: "grid", animate: true, animationDuration: 400, padding: 30 },
  cose: {
    name: "cose",
    animate: true,
    animationDuration: 400,
    padding: 30,
    nodeRepulsion: () => 10000,
    idealEdgeLength: () => 150,
  },
};

function getLangColor(language: string): string {
  return LANG_COLORS[language?.toLowerCase()] || LANG_COLORS.unknown;
}

function computeNodeSize(degree: number, symbolCount: number): number {
  const raw = 20 + Math.sqrt(degree || 0) * 8 + Math.sqrt(symbolCount || 0) * 3;
  return Math.max(20, Math.min(80, raw));
}

function getEdgeColor(depType: string): string {
  return EDGE_COLORS[depType] || EDGE_COLORS.external;
}

function getEdgeWidth(depType: string): number {
  return EDGE_WIDTHS[depType] || EDGE_WIDTHS.external;
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
  const layoutDoneRef = useRef(false);
  const prevDataRef = useRef<string>("");

  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [symbols, setSymbols] = useState<SymbolRow[]>([]);
  const [loadingSymbols, setLoadingSymbols] = useState(false);
  const [layout, setLayout] = useState("fcose");
  const [nodeLimit, setNodeLimit] = useState(200);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchOpen, setSearchOpen] = useState(false);
  const [isFocusMode, setIsFocusMode] = useState(false);
  const [showLabels, setShowLabels] = useState(true);

  const fetcher = useCallback(
    () => api.getCodeGraph({ limit: String(nodeLimit) }),
    [nodeLimit]
  );
  const { data } = usePolling<{ nodes: any[]; edges: any[] }>(fetcher, 10000, [nodeLimit]);

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
          selector: ":parent",
          style: {
            "background-opacity": 0.06,
            "background-color": "#94a3b8",
            "border-width": 1,
            "border-color": "#475569",
            "border-style": "dashed",
            "border-opacity": 0.4,
            label: "data(label)",
            "font-size": "11px",
            color: "#64748b",
            "text-valign": "top",
            "text-halign": "center",
            "text-margin-y": -8,
            padding: "20px",
          },
        },
        {
          selector: "edge",
          style: {
            "font-size": "7px",
            color: "#9ca3af",
            "text-outline-color": "#111827",
            "text-outline-width": 1,
            "line-color": "data(edgeColor)",
            "target-arrow-color": "data(edgeColor)",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
            width: "data(edgeWidth)",
            "arrow-scale": 0.8,
          },
        },
        {
          selector: "node:selected",
          style: { "border-width": 3, "border-color": "#f59e0b" },
        },
        {
          selector: ".dimmed",
          style: { opacity: 0.15 },
        },
        {
          selector: ".focused",
          style: { opacity: 1, "border-width": 4, "border-color": "#f59e0b", "z-index": 10 },
        },
        {
          selector: ".connected",
          style: { opacity: 1 },
        },
        {
          selector: ".connection",
          style: { opacity: 1, "line-color": "#f59e0b", "target-arrow-color": "#f59e0b", width: 2.5 },
        },
        {
          selector: ".highlighted",
          style: { "border-width": 4, "border-color": "#22d3ee", "z-index": 10 },
        },
      ],
      layout: { name: "grid" },
      wheelSensitivity: 0.3,
    });

    // Single click to select file (ignore compound/parent nodes)
    cy.on("tap", "node", (evt) => {
      const node = evt.target;
      if (node.isParent()) return;
      setSelectedFile(node.id());
    });

    // Double-click focus mode (ignore compound/parent nodes)
    cy.on("dblclick", "node", (evt) => {
      const node = evt.target;
      if (node.isParent()) return;
      setIsFocusMode(true);
      cy.elements().removeClass("dimmed focused connected connection");
      cy.elements().addClass("dimmed");
      node.removeClass("dimmed").addClass("focused");
      const neighborhood = node.neighborhood();
      neighborhood.nodes().removeClass("dimmed").addClass("connected");
      neighborhood.edges().removeClass("dimmed").addClass("connection");
      cy.animate(
        { fit: { eles: node.union(neighborhood), padding: 80 } },
        { duration: 600, easing: "ease-out-cubic" as any }
      );
    });

    // Click background to exit focus
    cy.on("tap", (evt) => {
      if (evt.target === cy) {
        setIsFocusMode(false);
        cy.elements().removeClass("dimmed focused connected connection");
        cy.animate(
          { fit: { eles: cy.elements(), padding: 30 } },
          { duration: 400, easing: "ease-out" as any }
        );
      }
    });

    cyRef.current = cy;
    return () => {
      cy.destroy();
      cyRef.current = null;
    };
  }, []);

  // Incremental data update
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy || !data) return;

    const incomingIds = [
      ...(data.nodes || []).map((n: any) => n.id),
      ...(data.edges || []).map((e: any) => e.id),
    ].sort().join(",");

    if (incomingIds === prevDataRef.current) return;
    prevDataRef.current = incomingIds;

    const incomingNodeIds = new Set((data.nodes || []).map((n: any) => n.id));
    const incomingEdgeIds = new Set((data.edges || []).map((e: any) => e.id));
    const existingNodeIds = new Set(cy.nodes().filter((n) => !n.isParent()).map((n) => n.id()));

    // Collect unique directories for compound nodes
    const directories = new Set<string>();
    for (const n of data.nodes || []) {
      if (n.directory) {
        directories.add(n.directory);
      }
    }

    // Remove stale file nodes and edges
    cy.nodes().filter((n) => !n.isParent()).forEach((n) => {
      if (!incomingNodeIds.has(n.id())) n.remove();
    });
    cy.edges().forEach((e) => {
      if (!incomingEdgeIds.has(e.id())) e.remove();
    });

    // Add/update compound (directory) parent nodes
    const existingParentIds = new Set(cy.nodes().filter((n) => n.isParent()).map((n) => n.id()));
    const neededParentIds = new Set<string>();
    for (const dir of directories) {
      const parentId = `dir-${dir}`;
      neededParentIds.add(parentId);
      if (!existingParentIds.has(parentId)) {
        cy.add({ data: { id: parentId, label: dir } });
      }
    }

    // Add/update file nodes
    for (const n of data.nodes || []) {
      const existing = cy.getElementById(n.id);
      const color = getLangColor(n.language);
      const size = computeNodeSize(n.degree || 0, n.symbol_count || 0);
      const parent = n.directory ? `dir-${n.directory}` : undefined;
      if (existing.length > 0 && !existing.isParent()) {
        existing.data({ label: n.label, color, size, parent });
      } else if (existing.length === 0) {
        const nodeData: any = { id: n.id, label: n.label, color, size };
        if (parent) nodeData.parent = parent;
        cy.add({ data: nodeData });
      }
    }

    // Remove orphan parent nodes (directories with no children)
    cy.nodes().filter((n) => n.isParent()).forEach((n) => {
      if (!neededParentIds.has(n.id()) || n.children().length === 0) {
        n.remove();
      }
    });

    // Add edges
    for (const e of data.edges || []) {
      if (cy.getElementById(e.id).length === 0) {
        const depType = e.dep_type || "external";
        cy.add({
          data: {
            id: e.id,
            source: e.source,
            target: e.target,
            label: e.import_name || "",
            edgeColor: getEdgeColor(depType),
            edgeWidth: getEdgeWidth(depType),
          },
        });
      }
    }

    const nodesChanged =
      incomingNodeIds.size !== existingNodeIds.size ||
      [...incomingNodeIds].some((id) => !existingNodeIds.has(id));

    if (!layoutDoneRef.current || nodesChanged) {
      if (cy.nodes().length > 0) {
        cy.layout({ ...LAYOUTS[layout], animate: layoutDoneRef.current } as any).run();
      }
      layoutDoneRef.current = true;
    }
  }, [data, layout]);

  // Re-layout
  function runLayout(name: string) {
    setLayout(name);
    const cy = cyRef.current;
    if (!cy || cy.nodes().length === 0) return;
    cy.layout({ ...LAYOUTS[name], animate: true } as any).run();
  }

  // Toggle labels
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.style()
      .selector("node")
      .style("label", showLabels ? "data(label)" : "")
      .update();
  }, [showLabels]);

  // Search/highlight
  function handleSearch(query: string) {
    setSearchQuery(query);
    const cy = cyRef.current;
    if (!cy) return;
    cy.elements().removeClass("highlighted");
    if (!query.trim()) return;
    const q = query.toLowerCase();
    cy.nodes().forEach((n) => {
      if (n.isParent()) return;
      if (n.data("id")?.toLowerCase().includes(q) || n.data("label")?.toLowerCase().includes(q)) {
        n.addClass("highlighted");
      }
    });
    const highlighted = cy.nodes(".highlighted");
    if (highlighted.length > 0) {
      cy.animate({ fit: { eles: highlighted, padding: 80 } }, { duration: 400 });
    }
  }

  // Export PNG
  function exportPng() {
    const cy = cyRef.current;
    if (!cy) return;
    const png = cy.png({ full: true, scale: 2, bg: "#111827" });
    const link = document.createElement("a");
    link.href = png;
    link.download = "code-graph.png";
    link.click();
  }

  // Fetch symbols for selected file
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

  const dependencies = data?.edges?.filter((e: any) => e.source === selectedFile) || [];
  const dependents = data?.edges?.filter((e: any) => e.target === selectedFile) || [];

  return (
    <div className="relative flex flex-col" style={{ height: "calc(100vh - 280px)" }}>
      {/* Controls bar */}
      <div className="flex items-center gap-2 p-2 border-b border-border bg-card/80 backdrop-blur-sm z-10 flex-wrap">
        <Select value={layout} onValueChange={(v) => v && runLayout(v)}>
          <SelectTrigger className="w-36 h-8 text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="fcose">Force Directed</SelectItem>
            <SelectItem value="circle">Circle</SelectItem>
            <SelectItem value="concentric">Concentric</SelectItem>
            <SelectItem value="breadthfirst">Hierarchy</SelectItem>
            <SelectItem value="grid">Grid</SelectItem>
            <SelectItem value="cose">CoSE</SelectItem>
          </SelectContent>
        </Select>

        <div className="flex items-center gap-1 text-xs text-muted-foreground">
          <span>Files:</span>
          <input
            type="range"
            min={20}
            max={500}
            step={10}
            value={nodeLimit}
            onChange={(e) => {
              setNodeLimit(parseInt(e.target.value));
              layoutDoneRef.current = false;
            }}
            className="w-20 h-1 accent-blue-500"
          />
          <span className="w-6 text-center">{nodeLimit}</span>
        </div>

        <div className="h-4 w-px bg-border" />

        <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => cyRef.current?.zoom(cyRef.current.zoom() * 1.3)}>
          <ZoomIn className="w-3.5 h-3.5" />
        </Button>
        <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => cyRef.current?.zoom(cyRef.current.zoom() * 0.7)}>
          <ZoomOut className="w-3.5 h-3.5" />
        </Button>
        <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => cyRef.current?.fit(undefined, 30)}>
          <Maximize className="w-3.5 h-3.5" />
        </Button>

        <div className="h-4 w-px bg-border" />

        <Button variant={showLabels ? "secondary" : "ghost"} size="sm" className="h-7 text-xs px-2" onClick={() => setShowLabels(!showLabels)}>
          Labels
        </Button>

        {searchOpen ? (
          <div className="flex items-center gap-1">
            <Input
              placeholder="Search files..."
              value={searchQuery}
              onChange={(e) => handleSearch(e.target.value)}
              className="h-7 w-40 text-xs"
              autoFocus
            />
            <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => { setSearchOpen(false); setSearchQuery(""); cyRef.current?.elements().removeClass("highlighted"); }}>
              <X className="w-3.5 h-3.5" />
            </Button>
          </div>
        ) : (
          <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => setSearchOpen(true)}>
            <Search className="w-3.5 h-3.5" />
          </Button>
        )}

        <Button variant="ghost" size="icon" className="h-7 w-7" onClick={exportPng}>
          <Download className="w-3.5 h-3.5" />
        </Button>

        {/* Dual legend: Language colors + Edge types */}
        <div className="flex items-center gap-3 ml-auto text-[10px] text-muted-foreground">
          <div className="flex items-center gap-2">
            {Object.entries(LANG_COLORS).filter(([k]) => k !== "unknown").map(([lang, color]) => (
              <div key={lang} className="flex items-center gap-1">
                <div className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
                {lang}
              </div>
            ))}
          </div>
          <div className="h-3 w-px bg-border" />
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1">
              <div className="w-3 h-0.5 rounded" style={{ backgroundColor: EDGE_COLORS.internal }} />
              internal
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-0.5 rounded" style={{ backgroundColor: EDGE_COLORS.external }} />
              external
            </div>
          </div>
        </div>

        {isFocusMode && (
          <span className="text-[10px] text-amber-400">
            Focus mode — click background to reset
          </span>
        )}
      </div>

      {/* Graph + panel */}
      <div className="flex flex-1 overflow-hidden">
        <div ref={containerRef} className="flex-1" style={{ backgroundColor: "#111827" }} />

        {/* Slide-out panel */}
        {selectedFile && (
          <div className="w-[380px] border-l border-border bg-card overflow-y-auto flex-shrink-0">
            <div className="flex items-center justify-between p-3 border-b border-border sticky top-0 bg-card z-10">
              <h3 className="text-sm font-semibold truncate" title={selectedFile}>
                {selectedFile.split("/").pop()}
              </h3>
              <button onClick={() => setSelectedFile(null)} className="p-1 hover:bg-accent rounded">
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="p-3 text-xs text-muted-foreground border-b border-border">
              <span className="font-mono">{selectedFile}</span>
            </div>

            <div className="p-3 border-b border-border">
              <h4 className="text-xs font-semibold uppercase text-muted-foreground mb-2">
                Symbols ({symbols.length})
              </h4>
              {loadingSymbols ? (
                <div className="text-xs text-muted-foreground">Loading...</div>
              ) : symbols.length === 0 ? (
                <div className="text-xs text-muted-foreground">No symbols found</div>
              ) : (
                <div className="max-h-[300px] overflow-y-auto">
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
                            <span className={`inline-block px-1 rounded text-[10px] ${
                              s.symbol_type === "function" ? "bg-blue-500/20 text-blue-400"
                                : s.symbol_type === "class" ? "bg-purple-500/20 text-purple-400"
                                : "bg-gray-500/20 text-gray-400"
                            }`}>
                              {s.symbol_type}
                            </span>
                          </td>
                          <td className="py-0.5 text-right text-muted-foreground">{s.line_number}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            <div className="p-3 border-b border-border">
              <h4 className="text-xs font-semibold uppercase text-muted-foreground mb-2">
                Dependencies ({dependencies.length})
              </h4>
              {dependencies.length === 0 ? (
                <div className="text-xs text-muted-foreground">None</div>
              ) : (
                <ul className="space-y-1">
                  {dependencies.map((d: any, i: number) => (
                    <li key={i} className="text-xs font-mono truncate cursor-pointer hover:text-blue-400" title={d.target} onClick={() => setSelectedFile(d.target)}>
                      {d.target.split("/").pop()}
                      {d.import_name && <span className="text-muted-foreground ml-1">({d.import_name})</span>}
                    </li>
                  ))}
                </ul>
              )}
            </div>

            <div className="p-3">
              <h4 className="text-xs font-semibold uppercase text-muted-foreground mb-2">
                Dependents ({dependents.length})
              </h4>
              {dependents.length === 0 ? (
                <div className="text-xs text-muted-foreground">None</div>
              ) : (
                <ul className="space-y-1">
                  {dependents.map((d: any, i: number) => (
                    <li key={i} className="text-xs font-mono truncate cursor-pointer hover:text-blue-400" title={d.source} onClick={() => setSelectedFile(d.source)}>
                      {d.source.split("/").pop()}
                      {d.import_name && <span className="text-muted-foreground ml-1">({d.import_name})</span>}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        )}
      </div>

      <div className="absolute bottom-3 left-3 text-[10px] text-muted-foreground/50 z-10">
        Drag to pan &middot; Scroll to zoom &middot; Click node for details &middot; Double-click to focus
      </div>
    </div>
  );
}
