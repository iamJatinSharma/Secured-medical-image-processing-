"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Users, Search, Loader2, AlertCircle, Trash2, Power, ShieldOff } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { usersApi } from "@/lib/api";
import { extractApiError } from "@/lib/utils";
import { useAuthStore } from "@/store/authStore";
import type { User } from "@/lib/types";

const ROLES: User["role"][] = ["admin", "doctor", "researcher", "viewer"];

const ROLE_COLORS: Record<User["role"], string> = {
  admin: "bg-purple-100 text-purple-700 border-purple-200 dark:bg-purple-900/20 dark:text-purple-400",
  doctor: "bg-blue-100 text-blue-700 border-blue-200 dark:bg-blue-900/20 dark:text-blue-400",
  researcher: "bg-amber-100 text-amber-700 border-amber-200 dark:bg-amber-900/20 dark:text-amber-400",
  viewer: "bg-gray-100 text-gray-700 border-gray-200 dark:bg-gray-900/20 dark:text-gray-400",
};

export default function UsersPage() {
  const me = useAuthStore((s) => s.user);
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>("");
  const [search, setSearch] = useState("");
  const [busyUserId, setBusyUserId] = useState<number | null>(null);

  const load = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await usersApi.list();
      setUsers(res.users);
    } catch (err) {
      setError(extractApiError(err, "Failed to load users. Admin privileges required."));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const updateRole = async (user: User, role: User["role"]) => {
    if (user.role === role) return;
    setBusyUserId(user.id);
    setError("");
    try {
      const updated = await usersApi.updateRole(user.id, role);
      setUsers((prev) => prev.map((u) => (u.id === updated.id ? updated : u)));
    } catch (err) {
      setError(extractApiError(err, "Failed to update role."));
    } finally {
      setBusyUserId(null);
    }
  };

  const toggleActive = async (user: User) => {
    setBusyUserId(user.id);
    setError("");
    try {
      const updated = await usersApi.setActive(user.id, !user.is_active);
      setUsers((prev) => prev.map((u) => (u.id === updated.id ? updated : u)));
    } catch (err) {
      setError(extractApiError(err, "Failed to toggle active state."));
    } finally {
      setBusyUserId(null);
    }
  };

  const deleteUser = async (user: User) => {
    if (!confirm(`Delete user "${user.username}"? This cannot be undone.`)) return;
    setBusyUserId(user.id);
    setError("");
    try {
      await usersApi.delete(user.id);
      setUsers((prev) => prev.filter((u) => u.id !== user.id));
    } catch (err) {
      setError(extractApiError(err, "Failed to delete user."));
    } finally {
      setBusyUserId(null);
    }
  };

  const filtered = users.filter(
    (u) =>
      u.username.toLowerCase().includes(search.toLowerCase()) ||
      u.email.toLowerCase().includes(search.toLowerCase()),
  );

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="p-3 rounded-xl bg-primary/10">
            <Users className="w-6 h-6 text-primary" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-foreground">User Management</h1>
            <p className="text-muted-foreground">Manage platform users, roles and access</p>
          </div>
        </div>
      </div>

      {error && (
        <div className="p-3 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-600 dark:text-red-400 text-sm flex items-center gap-2">
          <AlertCircle className="w-4 h-4 shrink-0" /> {error}
        </div>
      )}

      <div className="flex gap-2">
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search by username or email..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-10 pr-4 py-2 rounded-lg border border-input bg-background"
          />
        </div>
      </div>

      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <Card className="border-0 shadow-md">
          <CardHeader>
            <CardTitle>Users</CardTitle>
            <CardDescription>
              {loading ? "Loading..." : `${filtered.length} of ${users.length} users`}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="flex items-center justify-center py-12 text-muted-foreground">
                <Loader2 className="w-5 h-5 animate-spin mr-2" /> Loading users...
              </div>
            ) : filtered.length === 0 ? (
              <p className="text-center py-8 text-muted-foreground">
                {users.length === 0 ? "No users found." : "No users match your search."}
              </p>
            ) : (
              <div className="space-y-3">
                {filtered.map((u) => {
                  const isMe = me?.id === u.id;
                  const isBusy = busyUserId === u.id;
                  return (
                    <div
                      key={u.id}
                      className="flex flex-col md:flex-row md:items-center justify-between p-4 rounded-lg bg-muted/50 hover:bg-muted transition-colors gap-3"
                    >
                      <div className="flex items-center gap-4 min-w-0">
                        <Avatar>
                          <AvatarFallback className="bg-primary text-primary-foreground">
                            {u.username.slice(0, 2).toUpperCase()}
                          </AvatarFallback>
                        </Avatar>
                        <div className="min-w-0">
                          <div className="flex items-center gap-2">
                            <p className="font-medium text-foreground truncate">{u.username}</p>
                            {isMe && <Badge variant="outline" className="text-xs">You</Badge>}
                            {!u.is_active && <Badge variant="outline" className="text-xs text-red-600 border-red-200">Inactive</Badge>}
                          </div>
                          <p className="text-sm text-muted-foreground truncate">{u.email}</p>
                          <p className="text-xs text-muted-foreground mt-0.5">
                            Joined {new Date(u.created_at).toLocaleDateString()}
                          </p>
                        </div>
                      </div>

                      <div className="flex items-center gap-2 flex-wrap">
                        <select
                          value={u.role}
                          onChange={(e) => updateRole(u, e.target.value as User["role"])}
                          disabled={isBusy}
                          className={`px-3 py-1.5 rounded-md text-xs font-medium border ${ROLE_COLORS[u.role]}`}
                        >
                          {ROLES.map((r) => (
                            <option key={r} value={r}>{r}</option>
                          ))}
                        </select>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => toggleActive(u)}
                          disabled={isBusy || isMe}
                          title={u.is_active ? "Deactivate" : "Activate"}
                        >
                          {u.is_active ? <Power className="w-4 h-4 text-emerald-600" /> : <ShieldOff className="w-4 h-4 text-red-500" />}
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => deleteUser(u)}
                          disabled={isBusy || isMe}
                          title={isMe ? "You can't delete yourself" : "Delete user"}
                        >
                          {isBusy ? <Loader2 className="w-4 h-4 animate-spin" /> : <Trash2 className="w-4 h-4 text-red-500" />}
                        </Button>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}
