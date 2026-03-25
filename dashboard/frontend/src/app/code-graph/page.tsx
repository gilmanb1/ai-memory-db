"use client";

import { useCallback } from "react";
import dynamic from "next/dynamic";
import { api } from "@/lib/api";
import { usePolling } from "@/hooks/use-polling";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { FileCode2, Box, GitFork } from "lucide-react";

const CodeGraphView = dynamic(
  () => import("@/components/code-graph-view").then((m) => m.CodeGraphView),
  { ssr: false }
);

const LANG_BADGE_COLORS: Record<string, string> = {
  python: "bg-blue-500/20 text-blue-400 border-blue-500/30",
  typescript: "bg-green-500/20 text-green-400 border-green-500/30",
  javascript: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
  go: "bg-cyan-500/20 text-cyan-400 border-cyan-500/30",
  rust: "bg-orange-500/20 text-orange-400 border-orange-500/30",
};

export default function CodeGraphPage() {
  const fetcher = useCallback(() => api.getCodeGraphStats(), []);
  const { data: stats, loading } = usePolling(fetcher, 10000);

  return (
    <div>
      <h2 className="text-xl font-semibold mb-4">Code Graph</h2>

      {/* Stats cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
        <Card className="bg-gradient-to-br from-blue-500/10 to-blue-600/5 border-border/50">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-1.5">
              <FileCode2 className="w-4 h-4 text-blue-500" />
              Files
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {loading ? "..." : stats?.files ?? 0}
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-green-500/10 to-green-600/5 border-border/50">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-1.5">
              <Box className="w-4 h-4 text-green-500" />
              Symbols
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {loading ? "..." : stats?.symbols ?? 0}
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-purple-500/10 to-purple-600/5 border-border/50">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-1.5">
              <GitFork className="w-4 h-4 text-purple-500" />
              Dependencies
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {loading ? "..." : stats?.dependencies ?? 0}
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-gray-500/10 to-gray-600/5 border-border/50">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              By Language
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-1">
              {stats?.by_language?.length > 0 ? (
                stats.by_language.map((l: any) => (
                  <Badge
                    key={l.language}
                    variant="outline"
                    className={`text-[10px] ${
                      LANG_BADGE_COLORS[l.language?.toLowerCase()] ||
                      "bg-gray-500/20 text-gray-400 border-gray-500/30"
                    }`}
                  >
                    {l.language}: {l.files}f / {l.symbols}s
                  </Badge>
                ))
              ) : (
                <span className="text-xs text-muted-foreground">
                  {loading ? "..." : "No data"}
                </span>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Graph */}
      <CodeGraphView />
    </div>
  );
}
