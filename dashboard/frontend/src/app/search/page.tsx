"use client";

import { useState } from "react";
import { api } from "@/lib/api";
import { SearchResult } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Search } from "lucide-react";
import { toast } from "sonner";

const SEARCH_TYPES = [
  "fact",
  "decision",
  "observation",
  "guardrail",
  "procedure",
  "error_solution",
  "entity",
  "relationship",
];

const TYPE_COLORS: Record<string, string> = {
  fact: "border-blue-500/30 text-blue-400 bg-blue-500/10",
  decision: "border-purple-500/30 text-purple-400 bg-purple-500/10",
  observation: "border-cyan-500/30 text-cyan-400 bg-cyan-500/10",
  guardrail: "border-red-500/30 text-red-400 bg-red-500/10",
  procedure: "border-green-500/30 text-green-400 bg-green-500/10",
  error_solution: "border-orange-500/30 text-orange-400 bg-orange-500/10",
  entity: "border-yellow-500/30 text-yellow-400 bg-yellow-500/10",
  relationship: "border-pink-500/30 text-pink-400 bg-pink-500/10",
};

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [selectedTypes, setSelectedTypes] = useState<Set<string>>(new Set());
  const [results, setResults] = useState<SearchResult[]>([]);
  const [searching, setSearching] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);

  function toggleType(type: string) {
    setSelectedTypes((prev) => {
      const next = new Set(prev);
      if (next.has(type)) {
        next.delete(type);
      } else {
        next.add(type);
      }
      return next;
    });
  }

  async function handleSearch() {
    if (!query.trim()) return;
    setSearching(true);
    setHasSearched(true);
    try {
      const types = selectedTypes.size > 0 ? Array.from(selectedTypes) : undefined;
      const data = await api.search({ query: query.trim(), types, limit: 50 });
      setResults(data.results || data.items || []);
    } catch (e: any) {
      toast.error(e.message);
    } finally {
      setSearching(false);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter") {
      handleSearch();
    }
  }

  return (
    <div>
      <div className="mb-4">
        <h2 className="text-xl font-semibold mb-4">Search Memory</h2>

        <div className="flex gap-2 mb-3">
          <Input
            placeholder="Search query..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            className="max-w-lg"
          />
          <Button onClick={handleSearch} disabled={searching || !query.trim()}>
            <Search className="w-4 h-4 mr-1" />
            {searching ? "Searching..." : "Search"}
          </Button>
        </div>

        <div className="flex flex-wrap gap-2 mb-4">
          {SEARCH_TYPES.map((type) => (
            <button
              key={type}
              onClick={() => toggleType(type)}
              className={`px-2.5 py-1 text-xs rounded-md border transition-colors ${
                selectedTypes.has(type)
                  ? "bg-primary text-primary-foreground border-primary"
                  : "bg-muted/50 text-muted-foreground border-border hover:bg-muted"
              }`}
            >
              {type.replace("_", " ")}
            </button>
          ))}
        </div>
      </div>

      <div className="space-y-2">
        {results.map((result, i) => (
          <div
            key={`${result.id}-${i}`}
            className="border rounded-lg p-3 hover:bg-muted/50 transition-colors"
          >
            <div className="flex items-start justify-between mb-1">
              <div className="flex items-center gap-2">
                <Badge
                  variant="outline"
                  className={`text-[10px] ${TYPE_COLORS[result.type] || ""}`}
                >
                  {result.type.replace("_", " ")}
                </Badge>
                {result.scope && (
                  <span className="text-[10px] text-muted-foreground font-mono">
                    {result.scope}
                  </span>
                )}
              </div>
              {result.score != null && (
                <span className="text-xs font-medium text-muted-foreground">
                  {(result.score * 100).toFixed(1)}%
                </span>
              )}
            </div>
            <p className="text-sm">{result.text}</p>
          </div>
        ))}

        {hasSearched && results.length === 0 && !searching && (
          <div className="text-center text-muted-foreground py-8">
            No results found.
          </div>
        )}

        {!hasSearched && (
          <div className="text-center text-muted-foreground py-8">
            Enter a query and press Search to find memories.
          </div>
        )}
      </div>
    </div>
  );
}
