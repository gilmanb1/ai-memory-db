"use client";

import { useEffect, useRef, useCallback } from "react";
import cytoscape from "cytoscape";
import fcose from "cytoscape-fcose";
import { api } from "@/lib/api";
import { usePolling } from "@/hooks/use-polling";
import { GraphNode, GraphEdge } from "@/lib/types";

cytoscape.use(fcose);

const TYPE_COLORS: Record<string, string> = {
  person: "#3b82f6",
  technology: "#22c55e",
  organization: "#a855f7",
  general: "#6b7280",
};

function getColor(entityType: string): string {
  return TYPE_COLORS[entityType.toLowerCase()] || TYPE_COLORS.general;
}

export function EntityGraph() {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<cytoscape.Core | null>(null);

  const fetcher = useCallback(() => api.getGraph(), []);
  const { data } = usePolling<{ nodes: GraphNode[]; edges: GraphEdge[] }>(
    fetcher,
    5000
  );

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
          style: {
            "border-width": 3,
            "border-color": "#f59e0b",
          },
        },
      ],
      layout: { name: "grid" },
      wheelSensitivity: 0.3,
    });

    cyRef.current = cy;

    return () => {
      cy.destroy();
      cyRef.current = null;
    };
  }, []);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy || !data) return;

    const nodes = (data.nodes || []).map((n) => ({
      data: {
        id: n.id,
        label: n.label,
        color: getColor(n.entity_type),
        size: Math.max(25, Math.min(60, 25 + (n.session_count || 1) * 5)),
      },
    }));

    const edges = (data.edges || []).map((e) => ({
      data: {
        id: e.id,
        source: e.source,
        target: e.target,
        label: e.rel_type,
      },
    }));

    cy.json({ elements: { nodes, edges } });

    if (nodes.length > 0) {
      const layout = cy.layout({
        name: "fcose",
        animate: false,
        nodeDimensionsIncludeLabels: true,
        idealEdgeLength: 120,
        nodeRepulsion: 8000,
        padding: 30,
      } as any);
      layout.run();
    }
  }, [data]);

  return (
    <div
      ref={containerRef}
      className="w-full border rounded-md"
      style={{ height: "calc(100vh - 200px)", backgroundColor: "#111827" }}
    />
  );
}
