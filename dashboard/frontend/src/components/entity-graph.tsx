"use client";

import { useEffect, useRef, useCallback, useState } from "react";
import cytoscape from "cytoscape";
import fcose from "cytoscape-fcose";
import { api } from "@/lib/api";
import { usePolling } from "@/hooks/use-polling";
import { GraphNode, GraphEdge } from "@/lib/types";
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
  ZoomIn,
  ZoomOut,
  Maximize,
  Download,
  Search,
  X,
} from "lucide-react";

cytoscape.use(fcose);

const TYPE_COLORS: Record<string, string> = {
  person: "#3b82f6",
  technology: "#22c55e",
  organization: "#a855f7",
  general: "#6b7280",
};

const LAYOUTS: Record<string, any> = {
  fcose: {
    name: "fcose",
    animate: true,
    animationDuration: 400,
    nodeDimensionsIncludeLabels: true,
    idealEdgeLength: 120,
    nodeRepulsion: 8000,
    padding: 30,
  },
  circle: { name: "circle", animate: true, animationDuration: 400, padding: 30 },
  concentric: {
    name: "concentric",
    animate: true,
    animationDuration: 400,
    padding: 30,
    concentric: (node: any) => node.data("size") || 30,
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
    nodeRepulsion: () => 8000,
    idealEdgeLength: () => 120,
  },
};

function getColor(entityType: string): string {
  return TYPE_COLORS[entityType.toLowerCase()] || TYPE_COLORS.general;
}

interface EntityGraphProps {
  height?: string;
}

export function EntityGraph({ height }: EntityGraphProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<cytoscape.Core | null>(null);
  const layoutDoneRef = useRef(false);
  const prevDataRef = useRef<string>("");

  const [layout, setLayout] = useState("fcose");
  const [nodeLimit, setNodeLimit] = useState(100);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchOpen, setSearchOpen] = useState(false);
  const [isFocusMode, setIsFocusMode] = useState(false);
  const [showLabels, setShowLabels] = useState(true);

  const fetcher = useCallback(
    () => api.getGraph({ limit: String(nodeLimit) }),
    [nodeLimit]
  );
  const { data } = usePolling<{ nodes: GraphNode[]; edges: GraphEdge[] }>(
    fetcher,
    5000,
    [nodeLimit]
  );

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
            "text-valign": "center",
            "text-halign": "center",
            color: "#e5e7eb",
            "font-size": "10px",
            "text-outline-color": "#1f2937",
            "text-outline-width": 2,
            "background-color": "data(color)",
            width: "data(size)",
            height: "data(size)",
          },
        },
        {
          selector: "edge",
          style: {
            label: "data(label)",
            "font-size": "8px",
            color: "#9ca3af",
            "text-outline-color": "#1f2937",
            "text-outline-width": 1,
            "line-color": "#4b5563",
            "target-arrow-color": "#4b5563",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
            width: 1.5,
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
          style: {
            opacity: 1,
            "border-width": 4,
            "border-color": "#f59e0b",
            "z-index": 10,
          },
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
          style: {
            "border-width": 4,
            "border-color": "#22d3ee",
            "z-index": 10,
          },
        },
      ],
      layout: { name: "grid" },
      wheelSensitivity: 0.3,
    });

    // Double-click focus mode
    cy.on("dblclick", "node", (evt) => {
      const node = evt.target;
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
      ...(data.nodes || []).map((n) => n.id),
      ...(data.edges || []).map((e) => e.id),
    ].sort().join(",");

    if (incomingIds === prevDataRef.current) return;
    prevDataRef.current = incomingIds;

    const incomingNodeIds = new Set((data.nodes || []).map((n) => n.id));
    const incomingEdgeIds = new Set((data.edges || []).map((e) => e.id));
    const existingNodeIds = new Set(cy.nodes().map((n) => n.id()));

    cy.nodes().forEach((n) => {
      if (!incomingNodeIds.has(n.id())) n.remove();
    });
    cy.edges().forEach((e) => {
      if (!incomingEdgeIds.has(e.id())) e.remove();
    });

    for (const n of data.nodes || []) {
      const existing = cy.getElementById(n.id);
      const color = getColor(n.entity_type);
      const size = Math.max(25, Math.min(60, 25 + (n.session_count || 1) * 5));
      if (existing.length > 0) {
        existing.data({ label: n.label, color, size });
      } else {
        cy.add({ data: { id: n.id, label: n.label, color, size } });
      }
    }

    for (const e of data.edges || []) {
      if (cy.getElementById(e.id).length === 0) {
        cy.add({ data: { id: e.id, source: e.source, target: e.target, label: e.rel_type } });
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

  // Re-layout when layout changes
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
    cy.style()
      .selector("edge")
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
      if (n.data("label")?.toLowerCase().includes(q)) {
        n.addClass("highlighted");
      }
    });
    // Zoom to highlighted
    const highlighted = cy.nodes(".highlighted");
    if (highlighted.length > 0) {
      cy.animate(
        { fit: { eles: highlighted, padding: 80 } },
        { duration: 400 }
      );
    }
  }

  // Export PNG
  function exportPng() {
    const cy = cyRef.current;
    if (!cy) return;
    const png = cy.png({ full: true, scale: 2, bg: "#111827" });
    const link = document.createElement("a");
    link.href = png;
    link.download = "entity-graph.png";
    link.click();
  }

  return (
    <div className="relative flex flex-col" style={{ height: height || "calc(100vh - 200px)" }}>
      {/* Controls bar */}
      <div className="flex items-center gap-2 p-2 border-b border-border bg-card/80 backdrop-blur-sm z-10 flex-wrap">
        {/* Layout selector */}
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

        {/* Node limit */}
        <div className="flex items-center gap-1 text-xs text-muted-foreground">
          <span>Nodes:</span>
          <input
            type="range"
            min={20}
            max={300}
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

        {/* Zoom controls */}
        <Button
          variant="ghost" size="icon" className="h-7 w-7"
          onClick={() => cyRef.current?.zoom(cyRef.current.zoom() * 1.3)}
        >
          <ZoomIn className="w-3.5 h-3.5" />
        </Button>
        <Button
          variant="ghost" size="icon" className="h-7 w-7"
          onClick={() => cyRef.current?.zoom(cyRef.current.zoom() * 0.7)}
        >
          <ZoomOut className="w-3.5 h-3.5" />
        </Button>
        <Button
          variant="ghost" size="icon" className="h-7 w-7"
          onClick={() => cyRef.current?.fit(undefined, 30)}
        >
          <Maximize className="w-3.5 h-3.5" />
        </Button>

        <div className="h-4 w-px bg-border" />

        {/* Labels toggle */}
        <Button
          variant={showLabels ? "secondary" : "ghost"}
          size="sm"
          className="h-7 text-xs px-2"
          onClick={() => setShowLabels(!showLabels)}
        >
          Labels
        </Button>

        {/* Search */}
        {searchOpen ? (
          <div className="flex items-center gap-1">
            <Input
              placeholder="Search nodes..."
              value={searchQuery}
              onChange={(e) => handleSearch(e.target.value)}
              className="h-7 w-40 text-xs"
              autoFocus
            />
            <Button
              variant="ghost" size="icon" className="h-7 w-7"
              onClick={() => {
                setSearchOpen(false);
                setSearchQuery("");
                cyRef.current?.elements().removeClass("highlighted");
              }}
            >
              <X className="w-3.5 h-3.5" />
            </Button>
          </div>
        ) : (
          <Button
            variant="ghost" size="icon" className="h-7 w-7"
            onClick={() => setSearchOpen(true)}
          >
            <Search className="w-3.5 h-3.5" />
          </Button>
        )}

        {/* Export */}
        <Button
          variant="ghost" size="icon" className="h-7 w-7"
          onClick={exportPng}
        >
          <Download className="w-3.5 h-3.5" />
        </Button>

        {/* Focus mode indicator */}
        {isFocusMode && (
          <span className="text-[10px] text-amber-400 ml-auto">
            Focus mode — click background to reset
          </span>
        )}
      </div>

      {/* Graph */}
      <div
        ref={containerRef}
        className="flex-1 w-full"
        style={{ backgroundColor: "#111827" }}
      />

      {/* Help hint */}
      <div className="absolute bottom-3 right-3 text-[10px] text-muted-foreground/50 z-10">
        Drag to pan &middot; Scroll to zoom &middot; Double-click node to focus
      </div>
    </div>
  );
}
