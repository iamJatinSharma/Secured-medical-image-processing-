"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { LineChart as LineChartIcon, Loader2, AlertCircle } from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  AreaChart,
  Area,
} from "recharts";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { metricsApi } from "@/lib/api";
import { extractApiError } from "@/lib/utils";
import type { ModelMetrics } from "@/lib/types";

const MODELS = [
  { id: "resnet50_breast", name: "ResNet-50 (Breast)" },
  { id: "resnet50_lung", name: "ResNet-50 (Lung)" },
  { id: "resnet50_brain", name: "ResNet-50 (Brain)" },
  { id: "resnet50_skin", name: "ResNet-50 (Skin)" },
];

export default function VisualizationsPage() {
  const [selectedModel, setSelectedModel] = useState(MODELS[0].id);
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>("");

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError("");
    metricsApi.getModelMetrics(selectedModel)
      .then((m) => { if (!cancelled) setMetrics(m); })
      .catch((err) => { if (!cancelled) setError(extractApiError(err, "Failed to load model metrics.")); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [selectedModel]);

  // Prepare data for training history chart
  const trainingData = metrics?.training_history
    ? metrics.training_history.epochs.map((e, i) => ({
        epoch: e,
        train_loss: metrics.training_history!.train_loss[i],
        val_loss: metrics.training_history!.val_loss[i],
        train_acc: metrics.training_history!.train_acc[i],
        val_acc: metrics.training_history!.val_acc[i],
      }))
    : [];

  // Prepare ROC data
  const rocData = metrics?.roc_data
    ? metrics.roc_data.fpr.map((fpr, i) => ({
        fpr,
        tpr: metrics.roc_data!.tpr[i],
        baseline: fpr,
      }))
    : [];

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <div className="p-3 rounded-xl bg-primary/10">
          <LineChartIcon className="w-6 h-6 text-primary" />
        </div>
        <div>
          <h1 className="text-2xl font-bold text-foreground">Interactive Visualizations</h1>
          <p className="text-muted-foreground">Training curves, ROC analysis, and class distributions</p>
        </div>
      </div>

      {/* Model selector */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <Card className="border-0 shadow-md">
          <CardContent className="p-4">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-sm font-medium text-muted-foreground">Model:</span>
              {MODELS.map((m) => (
                <button
                  key={m.id}
                  onClick={() => setSelectedModel(m.id)}
                  className={`px-3 py-1.5 rounded-md text-sm font-medium border-2 transition-all ${
                    selectedModel === m.id ? "border-primary bg-primary/5 text-primary" : "border-transparent bg-muted/50 hover:bg-muted"
                  }`}
                >
                  {m.name}
                </button>
              ))}
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {loading && (
        <div className="flex items-center justify-center py-16 text-muted-foreground">
          <Loader2 className="w-5 h-5 animate-spin mr-2" /> Loading visualizations...
        </div>
      )}

      {error && (
        <div className="p-3 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-600 dark:text-red-400 text-sm flex items-center gap-2">
          <AlertCircle className="w-4 h-4 shrink-0" /> {error}
        </div>
      )}

      {!loading && metrics && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Training Loss */}
          {trainingData.length > 0 && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
              <Card className="border-0 shadow-md">
                <CardHeader>
                  <CardTitle>Training Loss Curve</CardTitle>
                  <CardDescription>Loss over epochs (lower is better)</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={260}>
                    <LineChart data={trainingData}>
                      <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                      <XAxis dataKey="epoch" label={{ value: "Epoch", position: "insideBottom", offset: -5 }} fontSize={12} />
                      <YAxis fontSize={12} />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="train_loss" stroke="#2563eb" strokeWidth={2} dot={false} name="Train Loss" />
                      <Line type="monotone" dataKey="val_loss" stroke="#f59e0b" strokeWidth={2} dot={false} name="Val Loss" />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </motion.div>
          )}

          {/* Training Accuracy */}
          {trainingData.length > 0 && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
              <Card className="border-0 shadow-md">
                <CardHeader>
                  <CardTitle>Training Accuracy Curve</CardTitle>
                  <CardDescription>Accuracy over epochs (higher is better)</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={260}>
                    <LineChart data={trainingData}>
                      <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                      <XAxis dataKey="epoch" label={{ value: "Epoch", position: "insideBottom", offset: -5 }} fontSize={12} />
                      <YAxis fontSize={12} domain={[0, 1]} />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="train_acc" stroke="#10b981" strokeWidth={2} dot={false} name="Train Acc" />
                      <Line type="monotone" dataKey="val_acc" stroke="#8b5cf6" strokeWidth={2} dot={false} name="Val Acc" />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </motion.div>
          )}

          {/* ROC Curve */}
          {rocData.length > 0 && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
              <Card className="border-0 shadow-md">
                <CardHeader>
                  <CardTitle>ROC Curve</CardTitle>
                  <CardDescription>AUC = {metrics.roc_data?.auc.toFixed(3) ?? "—"}</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={260}>
                    <AreaChart data={rocData}>
                      <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                      <XAxis dataKey="fpr" type="number" domain={[0, 1]} label={{ value: "False Positive Rate", position: "insideBottom", offset: -5 }} fontSize={12} />
                      <YAxis type="number" domain={[0, 1]} label={{ value: "TPR", angle: -90, position: "insideLeft" }} fontSize={12} />
                      <Tooltip />
                      <Legend />
                      <Area type="monotone" dataKey="tpr" stroke="#2563eb" fill="#2563eb" fillOpacity={0.2} name="ROC" />
                      <Line type="monotone" dataKey="baseline" stroke="#9ca3af" strokeDasharray="4 4" dot={false} name="Random" />
                    </AreaChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </motion.div>
          )}

          {/* Confusion Matrix as a heatmap-like grid */}
          {metrics.confusion_matrix && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}>
              <Card className="border-0 shadow-md">
                <CardHeader>
                  <CardTitle>Confusion Matrix</CardTitle>
                  <CardDescription>Diagonal = correct predictions</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center justify-center py-4">
                    <ConfusionMatrixHeatmap
                      matrix={metrics.confusion_matrix}
                      labels={metrics.labels ?? metrics.confusion_matrix.map((_, i) => `Class ${i}`)}
                    />
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          )}
        </div>
      )}
    </div>
  );
}

function ConfusionMatrixHeatmap({ matrix, labels }: { matrix: number[][]; labels: string[] }) {
  const max = Math.max(...matrix.flat());
  const colorFor = (i: number, j: number, value: number) => {
    const intensity = max > 0 ? value / max : 0;
    if (i === j) {
      // diagonal: green tint
      const alpha = Math.max(0.15, intensity);
      return `rgba(16, 185, 129, ${alpha})`;
    }
    const alpha = Math.max(0.05, intensity * 0.7);
    return `rgba(244, 63, 94, ${alpha})`;
  };

  return (
    <div className="inline-block">
      <table className="border-collapse">
        <thead>
          <tr>
            <th className="px-2 py-1"></th>
            {labels.map((l) => (
              <th key={l} className="px-3 py-2 text-xs font-medium text-muted-foreground">{l}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <th className="px-3 py-2 text-xs font-medium text-muted-foreground text-right whitespace-nowrap">{labels[i]}</th>
              {row.map((cell, j) => (
                <td
                  key={j}
                  className="w-20 h-20 border text-center font-semibold text-sm"
                  style={{ backgroundColor: colorFor(i, j, cell) }}
                >
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-2 flex justify-between text-xs text-muted-foreground px-3">
        <span>← Predicted →</span>
        <span>↑ True ↓</span>
      </div>
    </div>
  );
}
