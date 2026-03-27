"use client";

import { useEffect, useRef, useCallback, useState, useMemo } from "react";
import cytoscape from "cytoscape";
import fcose from "cytoscape-fcose";
import { api } from "@/lib/api";
import { usePolling } from "@/hooks/use-polling";
import { KnowledgeNode, KnowledgeEdge, KnowledgeGraphData } from "@/lib/types";
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
  Flame,
} from "lucide-react";

cytoscape.use(fcose);

const NODE_TYPE_COLORS: Record<string, string> = {
  entity: "#3b82f6",
  fact: "#22c55e",
  decision: "#a855f7",
  observation: "#f59e0b",
  guardrail: "#ef4444",
  procedure: "#06b6d4",
  error_solution: "#f97316",
  file: "#6b7280",
};

const EDGE_PALETTE = [
  "#f87171", "#fb923c", "#fbbf24", "#a3e635", "#34d399", "#22d3ee",
  "#60a5fa", "#a78bfa", "#f472b6", "#e879f9", "#94a3b8", "#fcd34d",
];

const ALL_NODE_TYPES = [
  "entity", "fact", "decision", "observation", "guardrail", "procedure", "error_solution", "file",
];

function getNodeColor(nodeType: string): string {
  return NODE_TYPE_COLORS[nodeType] || NODE_TYPE_COLORS.entity;
}

function buildEdgeColorMap(edges: KnowledgeEdge[]): Map<string, string> {
  const map = new Map<string, string>();
  let idx = 0;
  for (const e of edges) {
    if (!map.has(e.edge_type)) {
      map.set(e.edge_type, EDGE_PALETTE[idx % EDGE_PALETTE.length]);
      idx++;
    }
  }
  return map;
}

function countEdgeTypes(edges: KnowledgeEdge[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (const e of edges) {
    counts.set(e.edge_type, (counts.get(e.edge_type) || 0) + 1);
  }
  return counts;
}

function nodeSize(sizeMetric: number): number {
  return Math.max(25, Math.min(80, 25 + Math.sqrt(sizeMetric) * 8));
}

function edgeWidth(strength: number): number {
  return Math.max(1, Math.min(4, strength * 2));
}

// ── Heatmap ──────────────────────────────────────────────────────────────

const HEATMAP_METRICS = ["degree", "size_metric", "recency"] as const;
type HeatmapMetric = (typeof HEATMAP_METRICS)[number];

const HEATMAP_LABELS: Record<HeatmapMetric, string> = {
  degree: "Connections",
  size_metric: "Importance",
  recency: "Recency",
};

// Cool-to-hot gradient: deep blue → cyan → green → yellow → orange → red
const HEATMAP_STOPS = [
  { pos: 0.0, r: 30,  g: 58,  b: 138 },  // deep blue
  { pos: 0.2, r: 6,   g: 182, b: 212 },  // cyan
  { pos: 0.4, r: 34,  g: 197, b: 94  },  // green
  { pos: 0.6, r: 250, g: 204, b: 21  },  // yellow
  { pos: 0.8, r: 249, g: 115, b: 22  },  // orange
  { pos: 1.0, r: 239, g: 68,  b: 68  },  // red
];

function heatColor(t: number): string {
  const clamped = Math.max(0, Math.min(1, t));
  // Find the two stops to interpolate between
  let low = HEATMAP_STOPS[0], high = HEATMAP_STOPS[HEATMAP_STOPS.length - 1];
  for (let i = 0; i < HEATMAP_STOPS.length - 1; i++) {
    if (clamped >= HEATMAP_STOPS[i].pos && clamped <= HEATMAP_STOPS[i + 1].pos) {
      low = HEATMAP_STOPS[i];
      high = HEATMAP_STOPS[i + 1];
      break;
    }
  }
  const range = high.pos - low.pos || 1;
  const f = (clamped - low.pos) / range;
  const r = Math.round(low.r + (high.r - low.r) * f);
  const g = Math.round(low.g + (high.g - low.g) * f);
  const b = Math.round(low.b + (high.b - low.b) * f);
  return `rgb(${r},${g},${b})`;
}

function computeHeatValues(
  nodes: KnowledgeNode[],
  metric: HeatmapMetric,
): Map<string, number> {
  const map = new Map<string, number>();
  let values: number[] = [];

  for (const n of nodes) {
    let val = 0;
    if (metric === "degree") {
      val = n.degree ?? 0;
    } else if (metric === "size_metric") {
      val = n.size_metric ?? 0;
    } else if (metric === "recency") {
      // Placeholder — backend doesn't send timestamps directly,
      // but nodes are sorted by size_metric desc which correlates with activity.
      // Use the node's index position as a proxy (first = most active)
      val = 0; // filled below
    }
    values.push(val);
  }

  if (metric === "recency") {
    // Reverse index: last items in the array are oldest
    values = nodes.map((_, i) => nodes.length - i);
  }

  const maxVal = Math.max(...values, 1);
  const minVal = Math.min(...values, 0);
  const range = maxVal - minVal || 1;

  nodes.forEach((n, i) => {
    map.set(n.id, (values[i] - minVal) / range);
  });

  return map;
}

/** Compute density per cluster for heatmap backgrounds */
function computeClusterDensity(nodes: KnowledgeNode[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (const n of nodes) {
    if (n.cluster) counts.set(n.cluster, (counts.get(n.cluster) || 0) + 1);
  }
  const maxCount = Math.max(...counts.values(), 1);
  const density = new Map<string, number>();
  for (const [c, count] of counts) {
    density.set(c, count / maxCount);
  }
  return density;
}

function buildLayouts(): Record<string, any> {
  return {
    fcose: {
      name: "fcose",
      animate: true,
      animationDuration: 400,
      nodeDimensionsIncludeLabels: true,
      idealEdgeLength: (edge: any) => {
        const src = edge.source().data("cluster");
        const tgt = edge.target().data("cluster");
        return src && tgt && src === tgt ? 80 : 180;
      },
      nodeRepulsion: () => 8000,
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
}

interface KnowledgeGraphProps {
  height?: string;
  onNodeSelect?: (nodeId: string, nodeType: string, label: string) => void;
  highlightedNodes?: Set<string>;
}

export function KnowledgeGraph({ height, onNodeSelect, highlightedNodes }: KnowledgeGraphProps) {
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
  const [highlightedEdgeType, setHighlightedEdgeType] = useState<string | null>(null);
  const [clusterBy, setClusterBy] = useState("type");
  const [heatmapEnabled, setHeatmapEnabled] = useState(false);
  const [heatmapMetric, setHeatmapMetric] = useState<HeatmapMetric>("degree");
  const [enabledTypes, setEnabledTypes] = useState<Set<string>>(
    new Set(["entity", "fact", "decision"])
  );

  const fetcher = useCallback(
    () =>
      api.getKnowledgeGraph({
        limit: String(nodeLimit),
        types: Array.from(enabledTypes).join(","),
        cluster_by: clusterBy,
      }),
    [nodeLimit, enabledTypes, clusterBy]
  );
  const { data } = usePolling<KnowledgeGraphData>(fetcher, 5000, [
    nodeLimit,
    enabledTypes,
    clusterBy,
  ]);

  const edgeColorMap = useMemo(
    () => buildEdgeColorMap(data?.edges || []),
    [data?.edges]
  );
  const edgeTypeCounts = useMemo(
    () => countEdgeTypes(data?.edges || []),
    [data?.edges]
  );

  const heatValues = useMemo(
    () => (data?.nodes ? computeHeatValues(data.nodes, heatmapMetric) : new Map<string, number>()),
    [data?.nodes, heatmapMetric]
  );

  const clusterDensity = useMemo(
    () => (data?.nodes ? computeClusterDensity(data.nodes) : new Map<string, number>()),
    [data?.nodes]
  );

  const clusterColorMap = useMemo(() => {
    const map = new Map<string, string>();
    if (!data?.nodes) return map;
    let idx = 0;
    for (const n of data.nodes) {
      const c = n.cluster || "";
      if (c && !map.has(c)) {
        map.set(c, EDGE_PALETTE[idx % EDGE_PALETTE.length]);
        idx++;
      }
    }
    return map;
  }, [data?.nodes]);

  // Initialize Cytoscape
  useEffect(() => {
    if (!containerRef.current) return;

    const cy = cytoscape({
      container: containerRef.current,
      style: [
        {
          selector: "node.cluster-parent",
          style: {
            "background-opacity": 0,
            "border-width": 1,
            "border-style": "dashed" as any,
            "border-color": "#374151",
            "border-opacity": 0.5,
            label: "",
            padding: "20px" as any,
            shape: "round-rectangle",
          },
        },
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
            "border-width": 2,
            "border-color": "data(borderColor)",
            "border-opacity": 0.6,
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
            "line-color": "data(edgeColor)",
            "target-arrow-color": "data(edgeColor)",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
            width: "data(edgeWidth)",
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
        {
          selector: ".chat-highlight",
          style: {
            "border-width": 4,
            "border-color": "#f472b6",
            "z-index": 20,
            "border-style": "double" as any,
          },
        },
        {
          selector: ".edge-dimmed",
          style: { opacity: 0.08 },
        },
        {
          selector: ".edge-highlighted",
          style: {
            opacity: 1,
            width: 4,
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
      if (node.hasClass("cluster-parent")) return;
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

    // Single tap on node -> call onNodeSelect
    cy.on("tap", "node", (evt) => {
      const node = evt.target;
      if (node.hasClass("cluster-parent")) return;
      const nodeId = node.id();
      const nodeType = node.data("nodeType") || "entity";
      const label = node.data("label") || "";
      onNodeSelect?.(nodeId, nodeType, label);
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Update onNodeSelect ref without re-creating cy
  const onNodeSelectRef = useRef(onNodeSelect);
  useEffect(() => {
    onNodeSelectRef.current = onNodeSelect;
  }, [onNodeSelect]);

  // Highlighted nodes from chat
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.nodes().removeClass("chat-highlight");
    if (highlightedNodes && highlightedNodes.size > 0) {
      highlightedNodes.forEach((id) => {
        const node = cy.getElementById(id);
        if (node.length > 0) {
          node.addClass("chat-highlight");
        }
      });
    }
  }, [highlightedNodes]);

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
    const existingNodeIds = new Set(
      cy.nodes().filter((n) => !n.hasClass("cluster-parent")).map((n) => n.id())
    );

    const clusters = new Set<string>();
    for (const n of data.nodes || []) {
      if (n.cluster) clusters.add(n.cluster);
    }

    cy.nodes(".cluster-parent").remove();

    cy.nodes().forEach((n) => {
      if (!n.hasClass("cluster-parent") && !incomingNodeIds.has(n.id())) n.remove();
    });
    cy.edges().forEach((e) => {
      if (!incomingEdgeIds.has(e.id())) e.remove();
    });

    for (const clusterName of clusters) {
      const parentId = `cluster-${clusterName}`;
      if (cy.getElementById(parentId).length === 0) {
        cy.add({
          group: "nodes",
          data: { id: parentId, label: "" },
          classes: "cluster-parent",
        });
      }
    }

    for (const n of data.nodes || []) {
      const existing = cy.getElementById(n.id);
      const baseColor = getNodeColor(n.node_type);
      const heat = heatValues.get(n.id) ?? 0;
      const color = heatmapEnabled ? heatColor(heat) : baseColor;
      const size = heatmapEnabled
        ? Math.max(25, Math.min(90, 25 + heat * 65))  // heatmap: size also driven by metric
        : nodeSize(n.size_metric);
      const borderColor = heatmapEnabled
        ? heatColor(Math.min(1, heat + 0.15))
        : (n.cluster ? (clusterColorMap.get(n.cluster) || "#374151") : "#374151");
      const parentId = n.cluster ? `cluster-${n.cluster}` : undefined;

      if (existing.length > 0) {
        existing.data({
          label: n.label,
          color,
          size,
          borderColor,
          cluster: n.cluster || "",
          nodeType: n.node_type,
        });
        if (parentId && existing.data("parent") !== parentId) {
          existing.move({ parent: parentId });
        } else if (!parentId && existing.data("parent")) {
          existing.move({ parent: null });
        }
      } else {
        cy.add({
          data: {
            id: n.id,
            label: n.label,
            color,
            size,
            borderColor,
            cluster: n.cluster || "",
            nodeType: n.node_type,
            parent: parentId,
          },
        });
      }
    }

    // Heatmap: color cluster backgrounds by density
    if (heatmapEnabled) {
      cy.nodes(".cluster-parent").forEach((parent) => {
        const clusterName = parent.id().replace("cluster-", "");
        const density = clusterDensity.get(clusterName) ?? 0;
        const bgColor = heatColor(density);
        parent.style({
          "background-color": bgColor,
          "background-opacity": 0.12 + density * 0.15,
          "border-color": bgColor,
          "border-opacity": 0.4 + density * 0.3,
          "border-width": 2,
        });
      });
    } else {
      cy.nodes(".cluster-parent").forEach((parent) => {
        parent.style({
          "background-color": "#374151",
          "background-opacity": 0,
          "border-color": "#374151",
          "border-opacity": 0.5,
          "border-width": 1,
        });
      });
    }

    for (const e of data.edges || []) {
      const eColor = edgeColorMap.get(e.edge_type) || "#4b5563";
      const eWidth = edgeWidth(e.strength ?? 0.5);
      if (cy.getElementById(e.id).length === 0) {
        cy.add({
          data: {
            id: e.id,
            source: e.source,
            target: e.target,
            label: e.edge_type,
            edgeColor: eColor,
            edgeWidth: eWidth,
            edgeType: e.edge_type,
          },
        });
      } else {
        cy.getElementById(e.id).data({ edgeColor: eColor, edgeWidth: eWidth, edgeType: e.edge_type });
      }
    }

    const nodesChanged =
      incomingNodeIds.size !== existingNodeIds.size ||
      [...incomingNodeIds].some((id) => !existingNodeIds.has(id));

    if (!layoutDoneRef.current || nodesChanged) {
      const regularNodes = cy.nodes().filter((n) => !n.hasClass("cluster-parent"));
      if (regularNodes.length > 0) {
        const layouts = buildLayouts();
        cy.layout({ ...layouts[layout], animate: layoutDoneRef.current } as any).run();
      }
      layoutDoneRef.current = true;
    }
  }, [data, layout, edgeColorMap, clusterColorMap, heatmapEnabled, heatValues, clusterDensity]);

  function runLayout(name: string) {
    setLayout(name);
    const cy = cyRef.current;
    if (!cy || cy.nodes().filter((n) => !n.hasClass("cluster-parent")).length === 0) return;
    const layouts = buildLayouts();
    cy.layout({ ...layouts[name], animate: true } as any).run();
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
    cy.style()
      .selector("node.cluster-parent")
      .style("label", "")
      .update();
  }, [showLabels]);

  function handleLegendClick(edgeType: string) {
    const cy = cyRef.current;
    if (!cy) return;

    if (highlightedEdgeType === edgeType) {
      setHighlightedEdgeType(null);
      cy.elements().removeClass("edge-dimmed edge-highlighted");
      return;
    }

    setHighlightedEdgeType(edgeType);
    cy.elements().removeClass("edge-dimmed edge-highlighted");
    cy.edges().addClass("edge-dimmed");
    cy.nodes().filter((n) => !n.hasClass("cluster-parent")).addClass("edge-dimmed");

    cy.edges().forEach((e) => {
      if (e.data("edgeType") === edgeType) {
        e.removeClass("edge-dimmed").addClass("edge-highlighted");
        e.source().removeClass("edge-dimmed");
        e.target().removeClass("edge-dimmed");
      }
    });
  }

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
    const highlighted = cy.nodes(".highlighted");
    if (highlighted.length > 0) {
      cy.animate(
        { fit: { eles: highlighted, padding: 80 } },
        { duration: 400 }
      );
    }
  }

  function exportPng() {
    const cy = cyRef.current;
    if (!cy) return;
    const png = cy.png({ full: true, scale: 2, bg: "#111827" });
    const link = document.createElement("a");
    link.href = png;
    link.download = "knowledge-graph.png";
    link.click();
  }

  function toggleType(type: string) {
    setEnabledTypes((prev) => {
      const next = new Set(prev);
      if (next.has(type)) {
        next.delete(type);
      } else {
        next.add(type);
      }
      layoutDoneRef.current = false;
      return next;
    });
  }

  const legendEntries = useMemo(() => {
    const entries: { edgeType: string; color: string; count: number }[] = [];
    for (const [edgeType, color] of edgeColorMap) {
      entries.push({ edgeType, color, count: edgeTypeCounts.get(edgeType) || 0 });
    }
    entries.sort((a, b) => b.count - a.count);
    return entries;
  }, [edgeColorMap, edgeTypeCounts]);

  return (
    <div className="relative flex flex-col" style={{ height: height || "100%" }}>
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

        {/* Cluster-by dropdown */}
        <Select value={clusterBy} onValueChange={(v) => { if (v) { setClusterBy(v); layoutDoneRef.current = false; } }}>
          <SelectTrigger className="w-32 h-8 text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="type">By Type</SelectItem>
            <SelectItem value="community">By Community</SelectItem>
            <SelectItem value="session">By Session</SelectItem>
            <SelectItem value="scope">By Scope</SelectItem>
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

        {/* Heatmap toggle + metric */}
        <div className="h-4 w-px bg-border" />
        <Button
          variant={heatmapEnabled ? "secondary" : "ghost"}
          size="sm"
          className={`h-7 text-xs px-2 gap-1 ${heatmapEnabled ? "text-orange-400" : ""}`}
          onClick={() => {
            setHeatmapEnabled(!heatmapEnabled);
            prevDataRef.current = "";  // force re-render
          }}
        >
          <Flame className="w-3 h-3" />
          Heat
        </Button>
        {heatmapEnabled && (
          <Select value={heatmapMetric} onValueChange={(v) => {
            if (v) {
              setHeatmapMetric(v as HeatmapMetric);
              prevDataRef.current = "";
            }
          }}>
            <SelectTrigger className="w-28 h-7 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {HEATMAP_METRICS.map((m) => (
                <SelectItem key={m} value={m}>{HEATMAP_LABELS[m]}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}

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

      {/* Type filter toggles */}
      <div className="flex items-center gap-1.5 px-3 py-1.5 border-b border-border bg-card/60 backdrop-blur-sm z-10 overflow-x-auto flex-wrap">
        <span className="text-[10px] text-muted-foreground/70 uppercase tracking-wider shrink-0">
          Types
        </span>
        {ALL_NODE_TYPES.map((type) => {
          const active = enabledTypes.has(type);
          const count = data?.type_counts?.[type] || 0;
          return (
            <button
              key={type}
              onClick={() => toggleType(type)}
              className={`flex items-center gap-1.5 text-[11px] px-2 py-0.5 rounded-full transition-all shrink-0 border ${
                active
                  ? "border-white/20 bg-white/10 text-gray-200"
                  : "border-transparent bg-white/5 text-gray-500 opacity-50"
              }`}
            >
              <span
                className="inline-block w-2 h-2 rounded-full shrink-0"
                style={{ backgroundColor: NODE_TYPE_COLORS[type] || "#6b7280" }}
              />
              <span>{type.replace("_", " ")}</span>
              {count > 0 && <span className="text-gray-500">{count}</span>}
            </button>
          );
        })}
      </div>

      {/* Edge-type legend */}
      {legendEntries.length > 0 && (
        <div className="flex items-center gap-3 px-3 py-1.5 border-b border-border bg-card/60 backdrop-blur-sm z-10 overflow-x-auto flex-wrap">
          <span className="text-[10px] text-muted-foreground/70 uppercase tracking-wider shrink-0">
            Edge types
          </span>
          {legendEntries.map(({ edgeType, color, count }) => (
            <button
              key={edgeType}
              onClick={() => handleLegendClick(edgeType)}
              className={`flex items-center gap-1.5 text-[11px] px-1.5 py-0.5 rounded transition-all hover:bg-white/10 shrink-0 ${
                highlightedEdgeType === edgeType
                  ? "bg-white/15 ring-1 ring-white/20"
                  : highlightedEdgeType !== null
                  ? "opacity-40"
                  : ""
              }`}
            >
              <span
                className="inline-block w-2.5 h-0.5 shrink-0 rounded"
                style={{ backgroundColor: color }}
              />
              <span className="text-gray-300">{edgeType}</span>
              <span className="text-gray-500">{count}</span>
            </button>
          ))}
        </div>
      )}

      {/* Graph */}
      <div
        ref={containerRef}
        className="flex-1 w-full"
        style={{ backgroundColor: "#111827" }}
      />

      {/* Heatmap gradient legend */}
      {heatmapEnabled && (
        <div className="absolute bottom-10 left-1/2 -translate-x-1/2 z-10 flex items-center gap-2 bg-black/70 backdrop-blur-sm rounded-lg px-3 py-1.5 border border-white/10">
          <span className="text-[10px] text-muted-foreground">Low</span>
          <div
            className="h-3 w-32 rounded-sm"
            style={{
              background: `linear-gradient(to right, ${HEATMAP_STOPS.map(s => `rgb(${s.r},${s.g},${s.b}) ${s.pos * 100}%`).join(", ")})`,
            }}
          />
          <span className="text-[10px] text-muted-foreground">High</span>
          <span className="text-[10px] text-muted-foreground/60 ml-1">{HEATMAP_LABELS[heatmapMetric]}</span>
        </div>
      )}

      {/* Help hint */}
      <div className="absolute bottom-3 right-3 text-[10px] text-muted-foreground/50 z-10">
        Drag to pan &middot; Scroll to zoom &middot; Double-click node to focus &middot; Click node to chat
      </div>
    </div>
  );
}
