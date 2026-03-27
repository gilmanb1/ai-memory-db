"use client";

import dynamic from "next/dynamic";
import { useState, useRef, useCallback, useEffect } from "react";
import { ChatPanel } from "@/components/chat-panel";
import { Button } from "@/components/ui/button";
import { MessageSquare } from "lucide-react";

const KnowledgeGraph = dynamic(
  () => import("@/components/knowledge-graph").then((m) => ({ default: m.KnowledgeGraph })),
  { ssr: false }
);

export default function KnowledgePage() {
  const [highlightedNodes, setHighlightedNodes] = useState<Set<string>>(new Set());
  const [chatQuery, setChatQuery] = useState("");
  const [showChat, setShowChat] = useState(true);
  const [graphWidthPercent, setGraphWidthPercent] = useState(70);
  const containerRef = useRef<HTMLDivElement>(null);
  const isDragging = useRef(false);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    isDragging.current = true;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  }, []);

  useEffect(() => {
    function handleMouseMove(e: MouseEvent) {
      if (!isDragging.current || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const pct = Math.max(30, Math.min(85, (x / rect.width) * 100));
      setGraphWidthPercent(pct);
    }
    function handleMouseUp() {
      if (isDragging.current) {
        isDragging.current = false;
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
      }
    }
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, []);

  return (
    <div className="flex flex-col h-[calc(100vh-theme(spacing.12))]">
      <div className="flex items-center justify-between p-4 border-b border-border">
        <h1 className="text-2xl font-bold">Knowledge Graph</h1>
        <Button
          variant={showChat ? "secondary" : "ghost"}
          size="sm"
          onClick={() => setShowChat(!showChat)}
        >
          <MessageSquare className="w-4 h-4 mr-1" /> Chat
        </Button>
      </div>
      <div ref={containerRef} className="flex flex-1 overflow-hidden">
        {/* Graph panel */}
        <div style={{ width: showChat ? `${graphWidthPercent}%` : "100%" }}>
          <KnowledgeGraph
            highlightedNodes={highlightedNodes}
            onNodeSelect={(id, type, label) => {
              setChatQuery(`Tell me about ${label}`);
              if (!showChat) setShowChat(true);
            }}
          />
        </div>

        {/* Drag handle */}
        {showChat && (
          <div
            onMouseDown={handleMouseDown}
            className="w-1.5 bg-border hover:bg-blue-500/50 active:bg-blue-500 cursor-col-resize transition-colors shrink-0 relative group"
          >
            <div className="absolute inset-y-0 -left-1 -right-1" />
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-1 h-8 rounded-full bg-muted-foreground/30 group-hover:bg-blue-400/60 transition-colors" />
          </div>
        )}

        {/* Chat panel */}
        {showChat && (
          <div
            className="border-l border-border flex flex-col min-w-[250px]"
            style={{ width: `${100 - graphWidthPercent}%` }}
          >
            <ChatPanel
              onSourcesReceived={(ids) => setHighlightedNodes(new Set(ids))}
              initialQuery={chatQuery}
            />
          </div>
        )}
      </div>
    </div>
  );
}
