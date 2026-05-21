"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { Home, Activity, Shield, Image as ImageIcon, FileText, Stethoscope, Upload } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useAuthStore } from "@/store/authStore";
import { api } from "@/lib/api";

interface DashboardData {
  total_images: number;
  total_predictions: number;
  total_reports: number;
  recent_predictions: {
    id: number;
    model_name: string;
    prediction_label: string;
    confidence: number;
    created_at: string;
    processing_time_ms: number;
  }[];
}

export default function HomePage() {
  const user = useAuthStore((s) => s.user);
  const [data, setData] = useState<DashboardData | null>(null);

  useEffect(() => {
    api.get("/images/stats").then((res) => setData(res.data)).catch(() => {
      // Backend not running — use empty state
      setData({ total_images: 0, total_predictions: 0, total_reports: 0, recent_predictions: [] });
    });
  }, []);

  const stats = [
    { label: "Images Uploaded", value: data?.total_images ?? "-", icon: ImageIcon, color: "text-blue-600" },
    { label: "Detections Made", value: data?.total_predictions ?? "-", icon: Stethoscope, color: "text-teal-600" },
    { label: "Reports Generated", value: data?.total_reports ?? "-", icon: FileText, color: "text-purple-600" },
  ];

  return (
    <div className="space-y-6">
      {/* Welcome */}
      <div className="flex items-center gap-4">
        <div className="p-3 rounded-xl bg-primary/10">
          <Home className="w-6 h-6 text-primary" />
        </div>
        <div>
          <h1 className="text-2xl font-bold text-foreground">
            Welcome{user ? `, ${user.username}` : ""}
          </h1>
          <p className="text-muted-foreground">SecureMed AI — Secure Medical Imaging Platform</p>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {stats.map((stat, i) => {
          const Icon = stat.icon;
          return (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.1 }}
            >
              <Card className="border-0 shadow-md hover:shadow-lg transition-shadow">
                <CardContent className="p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-muted-foreground">{stat.label}</p>
                      <p className="text-3xl font-bold text-foreground mt-1">{stat.value}</p>
                    </div>
                    <div className="p-3 rounded-xl bg-muted">
                      <Icon className={`w-6 h-6 ${stat.color}`} />
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          );
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Activity */}
        <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.3 }}>
          <Card className="border-0 shadow-md h-full">
            <CardHeader>
              <CardTitle>Recent Predictions</CardTitle>
              <CardDescription>Latest disease detection results</CardDescription>
            </CardHeader>
            <CardContent>
              {data && data.recent_predictions.length > 0 ? (
                <div className="space-y-3">
                  {data.recent_predictions.map((pred) => (
                    <div key={pred.id} className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
                      <div>
                        <p className="font-medium text-foreground">{pred.prediction_label}</p>
                        <p className="text-sm text-muted-foreground">{pred.model_name}</p>
                      </div>
                      <div className="text-right">
                        <Badge variant="outline" className="bg-teal-50 text-teal-700 border-teal-200">
                          {(pred.confidence * 100).toFixed(1)}%
                        </Badge>
                        <p className="text-xs text-muted-foreground mt-1">{pred.processing_time_ms}ms</p>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  <Activity className="w-10 h-10 mx-auto mb-3 opacity-30" />
                  <p>No predictions yet</p>
                  <p className="text-sm mt-1">Upload an image to get started</p>
                </div>
              )}
            </CardContent>
          </Card>
        </motion.div>

        {/* Quick Actions */}
        <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.4 }}>
          <Card className="border-0 shadow-md h-full">
            <CardHeader>
              <CardTitle>Quick Actions</CardTitle>
              <CardDescription>Common tasks and shortcuts</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-3">
                {[
                  { label: "Upload Image", href: "/processing", icon: Upload },
                  { label: "New Detection", href: "/detection", icon: Stethoscope },
                  { label: "View Reports", href: "/reports", icon: FileText },
                  { label: "Security", href: "/security", icon: Shield },
                ].map((action) => {
                  const Icon = action.icon;
                  return (
                    <Link key={action.label} href={action.href}>
                      <motion.div
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        className="p-4 rounded-lg bg-primary/5 hover:bg-primary/10 border border-primary/10 text-center transition-colors cursor-pointer"
                      >
                        <Icon className="w-5 h-5 mx-auto mb-2 text-primary" />
                        <span className="font-medium text-foreground text-sm">{action.label}</span>
                      </motion.div>
                    </Link>
                  );
                })}
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </div>
  );
}
