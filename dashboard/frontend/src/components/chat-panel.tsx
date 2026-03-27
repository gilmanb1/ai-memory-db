"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { chatStream } from "@/lib/api";
import { ChatMessage } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Send, ChevronDown, ChevronRight } from "lucide-react";

const NODE_TYPE_COLORS: Record<string, string> = {
  entity: "#3b82f6",
  fact: "#22c55e",
  decision: "#a855f7",
  observation: "#f59e0b",
  guardrail: "#ef4444",
  procedure: "#06b6d4",
  error_solution: "#f97316",
  file: "#6b7280",
  facts: "#22c55e",
  decisions: "#a855f7",
  observations: "#f59e0b",
  guardrails: "#ef4444",
  procedures: "#06b6d4",
  error_solutions: "#f97316",
};

/** Extract citation node IDs from text like [fact:abc123] or [facts:abc123] */
function extractCitations(text: string): { fullMatch: string; nodeType: string; nodeId: string }[] {
  const re = /\[([\w_]+):([\w-]+)\]/g;
  const results: { fullMatch: string; nodeType: string; nodeId: string }[] = [];
  let match: RegExpExecArray | null;
  while ((match = re.exec(text)) !== null) {
    results.push({ fullMatch: match[0], nodeType: match[1], nodeId: match[2] });
  }
  return results;
}

/** Custom react-markdown components that render citation badges inline */
function createMarkdownComponents(onHover?: (nodeIds: string[]) => void) {
  return {
    // Override paragraph to handle citation badges
    p: ({ children, ...props }: any) => {
      return (
        <p className="mb-2 last:mb-0 leading-relaxed" {...props}>
          {processChildren(children, onHover)}
        </p>
      );
    },
    li: ({ children, ...props }: any) => {
      return (
        <li className="ml-4" {...props}>
          {processChildren(children, onHover)}
        </li>
      );
    },
    // Block code: react-markdown wraps fenced code in <pre><code>
    // so we style <pre> as the block container and <code> inside it inherits
    pre: ({ children, ...props }: any) => (
      <pre className="my-2 p-3 rounded-lg bg-black/30 border border-white/10 overflow-x-auto text-[11px] font-mono text-emerald-200" {...props}>
        {children}
      </pre>
    ),
    // Inline code: <code> NOT inside a <pre>
    code: ({ children, ...props }: any) => (
      <code className="px-1.5 py-0.5 rounded bg-white/10 text-[11px] font-mono text-emerald-300" {...props}>
        {children}
      </code>
    ),
    h1: ({ children, ...props }: any) => <h1 className="text-base font-bold mt-3 mb-1.5 text-foreground" {...props}>{children}</h1>,
    h2: ({ children, ...props }: any) => <h2 className="text-sm font-bold mt-2.5 mb-1 text-foreground" {...props}>{children}</h2>,
    h3: ({ children, ...props }: any) => <h3 className="text-sm font-semibold mt-2 mb-1 text-foreground" {...props}>{children}</h3>,
    ul: ({ children, ...props }: any) => <ul className="list-disc pl-4 mb-2 space-y-0.5" {...props}>{children}</ul>,
    ol: ({ children, ...props }: any) => <ol className="list-decimal pl-4 mb-2 space-y-0.5" {...props}>{children}</ol>,
    blockquote: ({ children, ...props }: any) => (
      <blockquote className="border-l-2 border-blue-500/50 pl-3 my-2 text-muted-foreground italic" {...props}>
        {children}
      </blockquote>
    ),
    table: ({ children, ...props }: any) => (
      <div className="overflow-x-auto my-2">
        <table className="text-xs border-collapse w-full" {...props}>{children}</table>
      </div>
    ),
    th: ({ children, ...props }: any) => (
      <th className="border border-border px-2 py-1 text-left font-semibold bg-muted/50" {...props}>{children}</th>
    ),
    td: ({ children, ...props }: any) => (
      <td className="border border-border px-2 py-1" {...props}>{children}</td>
    ),
    strong: ({ children, ...props }: any) => <strong className="font-semibold text-foreground" {...props}>{children}</strong>,
    hr: () => <hr className="my-3 border-border" />,
  };
}

/** Walk react children and replace citation strings with badge components */
function processChildren(children: React.ReactNode, onHover?: (nodeIds: string[]) => void): React.ReactNode {
  if (!children) return children;
  if (typeof children === "string") {
    return renderCitationsInText(children, onHover);
  }
  if (Array.isArray(children)) {
    return children.map((child, i) => {
      if (typeof child === "string") {
        return <span key={i}>{renderCitationsInText(child, onHover)}</span>;
      }
      return child;
    });
  }
  return children;
}

function renderCitationsInText(text: string, onHover?: (nodeIds: string[]) => void): React.ReactNode {
  const parts = text.split(/(\[[\w_]+:[\w-]+\])/g);
  if (parts.length === 1) return text;

  return parts.map((part, i) => {
    const citMatch = part.match(/^\[([\w_]+):([\w-]+)\]$/);
    if (citMatch) {
      const nodeType = citMatch[1];
      const nodeId = `${nodeType}:${citMatch[2]}`;
      const color = NODE_TYPE_COLORS[nodeType] || "#6b7280";
      return (
        <span
          key={i}
          className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-mono cursor-pointer transition-opacity hover:opacity-80 mx-0.5 align-baseline"
          style={{ backgroundColor: color + "30", color, border: `1px solid ${color}40` }}
          onMouseEnter={() => onHover?.([nodeId])}
          onMouseLeave={() => onHover?.([])}
          title={`${nodeType}: ${citMatch[2]}`}
        >
          {nodeType}:{citMatch[2].slice(0, 8)}
        </span>
      );
    }
    return <span key={i}>{part}</span>;
  });
}

interface ChatPanelProps {
  onSourcesReceived?: (nodeIds: string[]) => void;
  initialQuery?: string;
  scope?: string;
}

export function ChatPanel({ onSourcesReceived, initialQuery, scope }: ChatPanelProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedSources, setExpandedSources] = useState<Set<number>>(new Set());
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const processedInitialRef = useRef<string>("");

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Handle initialQuery changes
  useEffect(() => {
    if (initialQuery && initialQuery !== processedInitialRef.current && !isStreaming) {
      processedInitialRef.current = initialQuery;
      setInput(initialQuery);
    }
  }, [initialQuery, isStreaming]);

  const sendMessage = useCallback(async (text: string) => {
    if (!text.trim() || isStreaming) return;

    const userMsg: ChatMessage = { role: "user", content: text };
    const updatedMessages = [...messages, userMsg];
    setMessages(updatedMessages);
    setInput("");
    setError(null);
    setIsStreaming(true);

    try {
      const response = await chatStream({
        query: text,
        scope,
        conversation_history: updatedMessages,
      });

      if (!response.ok) {
        const errText = await response.text().catch(() => response.statusText);
        throw new Error(`API ${response.status}: ${errText}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error("No response stream");

      const decoder = new TextDecoder();
      let assistantContent = "";
      let sources: ChatMessage["sources"] = [];
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.trim()) continue;
          let parsed: any;
          try {
            parsed = JSON.parse(line);
          } catch {
            continue;
          }

          if (parsed.type === "text") {
            assistantContent += parsed.data || "";
            setMessages([
              ...updatedMessages,
              { role: "assistant", content: assistantContent, sources },
            ]);
          } else if (parsed.type === "sources") {
            sources = parsed.data || [];
            const nodeIds = sources?.map((s: any) => s.id) || [];
            onSourcesReceived?.(nodeIds);
          } else if (parsed.type === "error") {
            throw new Error(parsed.data || "Stream error");
          } else if (parsed.type === "done") {
            // Stream complete
          }
        }
      }

      // Final message with all sources
      if (assistantContent) {
        setMessages([
          ...updatedMessages,
          { role: "assistant", content: assistantContent, sources },
        ]);
      }
    } catch (e: any) {
      setError(e.message || "Failed to send message");
      setMessages([
        ...updatedMessages,
        { role: "assistant", content: "Sorry, I encountered an error. Please try again." },
      ]);
    } finally {
      setIsStreaming(false);
    }
  }, [messages, isStreaming, scope, onSourcesReceived]);

  function toggleSourceExpanded(idx: number) {
    setExpandedSources((prev) => {
      const next = new Set(prev);
      next.has(idx) ? next.delete(idx) : next.add(idx);
      return next;
    });
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    sendMessage(input);
  }

  const mdComponents = createMarkdownComponents(onSourcesReceived);

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-3 py-2 border-b border-border bg-card/80 backdrop-blur-sm">
        <h3 className="text-sm font-medium text-foreground">Knowledge Chat</h3>
        <p className="text-[10px] text-muted-foreground">Ask questions about your knowledge base</p>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full text-sm text-muted-foreground">
            <div className="text-center">
              <p className="mb-1">No messages yet</p>
              <p className="text-xs">Click a node in the graph or type a question below</p>
            </div>
          </div>
        )}

        {messages.map((msg, idx) => (
          <div key={idx}>
            <div className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
              <div
                className={`max-w-[95%] rounded-lg px-3 py-2 text-sm ${
                  msg.role === "user"
                    ? "bg-blue-600/80 text-white"
                    : "bg-muted text-foreground"
                }`}
              >
                {msg.role === "assistant" ? (
                  <ReactMarkdown remarkPlugins={[remarkGfm]} components={mdComponents}>
                    {msg.content}
                  </ReactMarkdown>
                ) : (
                  <p>{msg.content}</p>
                )}
              </div>
            </div>

            {/* Sources panel */}
            {msg.role === "assistant" && msg.sources && msg.sources.length > 0 && (
              <div className="ml-1 mt-1">
                <button
                  onClick={() => toggleSourceExpanded(idx)}
                  className="flex items-center gap-1 text-[10px] text-muted-foreground hover:text-foreground transition-colors"
                >
                  {expandedSources.has(idx) ? (
                    <ChevronDown className="w-3 h-3" />
                  ) : (
                    <ChevronRight className="w-3 h-3" />
                  )}
                  Sources ({msg.sources.length})
                </button>
                {expandedSources.has(idx) && (
                  <div className="mt-1 space-y-1 pl-3 border-l border-border">
                    {msg.sources.map((src, sIdx) => {
                      const color = NODE_TYPE_COLORS[src.node_type] || "#6b7280";
                      return (
                        <div
                          key={sIdx}
                          className="text-[11px] p-1.5 rounded bg-card/60 border border-border cursor-pointer hover:bg-card/80 transition-colors"
                          onMouseEnter={() => onSourcesReceived?.([src.id])}
                          onMouseLeave={() => onSourcesReceived?.([])}
                        >
                          <div className="flex items-center gap-1.5 mb-0.5">
                            <span
                              className="inline-block w-2 h-2 rounded-full shrink-0"
                              style={{ backgroundColor: color }}
                            />
                            <span className="font-medium" style={{ color }}>
                              {src.node_type}
                            </span>
                            {src.score !== undefined && (
                              <span className="text-muted-foreground ml-auto">
                                {(src.score * 100).toFixed(0)}%
                              </span>
                            )}
                          </div>
                          <p className="text-muted-foreground line-clamp-2">{src.text}</p>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            )}
          </div>
        ))}

        {/* Streaming indicator */}
        {isStreaming && (
          <div className="flex justify-start">
            <div className="bg-muted rounded-lg px-3 py-2 text-sm">
              <span className="inline-flex gap-1">
                <span className="w-1.5 h-1.5 bg-muted-foreground rounded-full animate-pulse" />
                <span className="w-1.5 h-1.5 bg-muted-foreground rounded-full animate-pulse [animation-delay:150ms]" />
                <span className="w-1.5 h-1.5 bg-muted-foreground rounded-full animate-pulse [animation-delay:300ms]" />
              </span>
            </div>
          </div>
        )}

        {/* Error display */}
        {error && (
          <div className="text-xs text-red-400 bg-red-950/30 border border-red-900/50 rounded px-2 py-1.5">
            {error}
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input area */}
      <form onSubmit={handleSubmit} className="p-3 border-t border-border bg-card/80">
        <div className="flex gap-2">
          <Input
            placeholder="Ask about your knowledge..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={isStreaming}
            className="text-sm"
          />
          <Button
            type="submit"
            size="icon"
            disabled={isStreaming || !input.trim()}
            className="shrink-0"
          >
            <Send className="w-4 h-4" />
          </Button>
        </div>
      </form>
    </div>
  );
}
