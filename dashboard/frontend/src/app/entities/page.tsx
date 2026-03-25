"use client";

import { useCallback, useState } from "react";
import dynamic from "next/dynamic";
import { api } from "@/lib/api";
import { usePolling } from "@/hooks/use-polling";
import { Entity } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ConfirmDialog } from "@/components/confirm-dialog";
import { Plus, Pencil, Trash2, TableIcon, Network } from "lucide-react";
import { toast } from "sonner";

const EntityGraph = dynamic(
  () => import("@/components/entity-graph").then((m) => m.EntityGraph),
  { ssr: false }
);

const ENTITY_TYPES = ["person", "technology", "organization", "general"];

const TYPE_COLORS: Record<string, string> = {
  person: "bg-blue-500/20 text-blue-400 border-blue-500/30",
  technology: "bg-green-500/20 text-green-400 border-green-500/30",
  organization: "bg-purple-500/20 text-purple-400 border-purple-500/30",
  general: "bg-gray-500/20 text-gray-400 border-gray-500/30",
};

function formatDate(iso: string): string {
  if (!iso) return "-";
  return new Date(iso).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

export default function EntitiesPage() {
  const [view, setView] = useState<"table" | "graph">("table");
  const [textFilter, setTextFilter] = useState("");
  const [typeFilter, setTypeFilter] = useState<string>("all");
  const [createOpen, setCreateOpen] = useState(false);
  const [editItem, setEditItem] = useState<Entity | null>(null);
  const [deleteId, setDeleteId] = useState<string | null>(null);

  // Create form
  const [newName, setNewName] = useState("");
  const [newType, setNewType] = useState("general");

  const params: Record<string, string> = { limit: "200" };
  if (typeFilter !== "all") params.entity_type = typeFilter;

  const fetcher = useCallback(() => api.getEntities(params), [typeFilter]);
  const { data, refetch } = usePolling(fetcher, 3000, [typeFilter]);

  const items: Entity[] = data?.items || [];
  const filtered = textFilter
    ? items.filter((e) =>
        e.name.toLowerCase().includes(textFilter.toLowerCase())
      )
    : items;

  async function handleCreate() {
    try {
      await api.createEntity({
        name: newName,
        entity_type: newType,
      });
      toast.success("Entity created");
      setCreateOpen(false);
      setNewName("");
      setNewType("general");
      refetch();
    } catch (e: any) {
      toast.error(e.message);
    }
  }

  async function handleUpdate() {
    if (!editItem) return;
    try {
      await api.updateEntity(editItem.id, {
        name: editItem.name,
        entity_type: editItem.entity_type,
      });
      toast.success("Entity updated");
      setEditItem(null);
      refetch();
    } catch (e: any) {
      toast.error(e.message);
    }
  }

  async function handleDelete() {
    if (!deleteId) return;
    try {
      await api.deleteEntity(deleteId);
      toast.success("Entity deleted");
      setDeleteId(null);
      refetch();
    } catch (e: any) {
      toast.error(e.message);
    }
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold">
          Entities ({data?.total ?? 0})
        </h2>
        <div className="flex gap-2">
          <div className="flex rounded-md border overflow-hidden">
            <Button
              variant={view === "table" ? "default" : "ghost"}
              size="sm"
              className="rounded-none"
              onClick={() => setView("table")}
            >
              <TableIcon className="w-4 h-4 mr-1" /> Table
            </Button>
            <Button
              variant={view === "graph" ? "default" : "ghost"}
              size="sm"
              className="rounded-none"
              onClick={() => setView("graph")}
            >
              <Network className="w-4 h-4 mr-1" /> Graph
            </Button>
          </div>
          <Button size="sm" onClick={() => setCreateOpen(true)}>
            <Plus className="w-4 h-4 mr-1" /> Add Entity
          </Button>
        </div>
      </div>

      {view === "table" && (
        <>
          <div className="flex gap-2 mb-4">
            <Input
              placeholder="Filter by name..."
              value={textFilter}
              onChange={(e) => setTextFilter(e.target.value)}
              className="max-w-sm"
            />
            <Select value={typeFilter} onValueChange={(v) => v && setTypeFilter(v)}>
              <SelectTrigger className="w-40">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All types</SelectItem>
                {ENTITY_TYPES.map((t) => (
                  <SelectItem key={t} value={t}>
                    {t}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="border rounded-md">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-[30%]">Name</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead className="text-center">Sessions</TableHead>
                  <TableHead>First Seen</TableHead>
                  <TableHead>Last Seen</TableHead>
                  <TableHead className="w-16"></TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filtered.map((entity) => (
                  <TableRow key={entity.id} className="group">
                    <TableCell className="text-sm font-medium">
                      {entity.name}
                    </TableCell>
                    <TableCell>
                      <Badge
                        variant="outline"
                        className={`text-[10px] ${TYPE_COLORS[entity.entity_type] || TYPE_COLORS.general}`}
                      >
                        {entity.entity_type}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-center text-sm text-muted-foreground">
                      {entity.session_count}
                    </TableCell>
                    <TableCell className="text-sm text-muted-foreground">
                      {formatDate(entity.first_seen_at)}
                    </TableCell>
                    <TableCell className="text-sm text-muted-foreground">
                      {formatDate(entity.last_seen_at)}
                    </TableCell>
                    <TableCell>
                      <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-7 w-7"
                          onClick={() => setEditItem({ ...entity })}
                        >
                          <Pencil className="w-3.5 h-3.5" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-7 w-7 text-destructive"
                          onClick={() => setDeleteId(entity.id)}
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </>
      )}

      {view === "graph" && <EntityGraph />}

      {/* Create Dialog */}
      <Dialog open={createOpen} onOpenChange={setCreateOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Add Entity</DialogTitle>
          </DialogHeader>
          <div className="space-y-3">
            <Input
              placeholder="Entity name..."
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
            />
            <Select value={newType} onValueChange={(v) => v && setNewType(v)}>
              <SelectTrigger>
                <SelectValue placeholder="Type" />
              </SelectTrigger>
              <SelectContent>
                {ENTITY_TYPES.map((t) => (
                  <SelectItem key={t} value={t}>
                    {t}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setCreateOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleCreate} disabled={!newName.trim()}>
              Create
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Edit Dialog */}
      <Dialog open={!!editItem} onOpenChange={(o) => !o && setEditItem(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit Entity</DialogTitle>
          </DialogHeader>
          {editItem && (
            <div className="space-y-3">
              <Input
                value={editItem.name}
                onChange={(e) =>
                  setEditItem({ ...editItem, name: e.target.value })
                }
              />
              <Select
                value={editItem.entity_type}
                onValueChange={(v) =>
                  v && setEditItem({ ...editItem, entity_type: v })
                }
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {ENTITY_TYPES.map((t) => (
                    <SelectItem key={t} value={t}>
                      {t}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}
          <DialogFooter>
            <Button variant="outline" onClick={() => setEditItem(null)}>
              Cancel
            </Button>
            <Button onClick={handleUpdate}>Save</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Confirm */}
      <ConfirmDialog
        open={!!deleteId}
        onOpenChange={(o) => !o && setDeleteId(null)}
        title="Delete Entity"
        description="This will delete the entity and may affect related relationships."
        onConfirm={handleDelete}
      />
    </div>
  );
}
