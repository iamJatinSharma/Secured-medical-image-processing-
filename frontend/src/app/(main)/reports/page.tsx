"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { FileText, Download, Trash2, Loader2, AlertCircle, Plus, X } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { reportsApi, predictApi } from "@/lib/api";
import { extractApiError } from "@/lib/utils";
import type { ReportSummary, Prediction } from "@/lib/types";

export default function ReportsPage() {
  const [reports, setReports] = useState<ReportSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>("");
  const [showGenerate, setShowGenerate] = useState(false);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [predLoading, setPredLoading] = useState(false);
  const [generatingId, setGeneratingId] = useState<number | null>(null);

  const loadReports = async () => {
    setLoading(true);
    setError("");
    try {
      const list = await reportsApi.list();
      setReports(list);
    } catch (err) {
      setError(extractApiError(err, "Failed to load reports."));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadReports();
  }, []);

  const openGenerate = async () => {
    setShowGenerate(true);
    setPredLoading(true);
    try {
      const list = await predictApi.list();
      setPredictions(list);
    } catch (err) {
      setError(extractApiError(err, "Failed to load predictions."));
    } finally {
      setPredLoading(false);
    }
  };

  const generateFor = async (predictionId: number) => {
    setGeneratingId(predictionId);
    setError("");
    try {
      const r = await reportsApi.generate(predictionId);
      setReports((prev) => [r, ...prev]);
      setShowGenerate(false);
    } catch (err) {
      setError(extractApiError(err, "Failed to generate report."));
    } finally {
      setGeneratingId(null);
    }
  };

  const downloadReport = async (r: ReportSummary) => {
    try {
      const blob = await reportsApi.download(r.id);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `report_${r.id}.pdf`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(extractApiError(err, "Failed to download report."));
    }
  };

  const deleteReport = async (r: ReportSummary) => {
    if (!confirm(`Delete report #${r.id}?`)) return;
    try {
      await reportsApi.delete(r.id);
      setReports((prev) => prev.filter((x) => x.id !== r.id));
    } catch (err) {
      setError(extractApiError(err, "Failed to delete report."));
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="p-3 rounded-xl bg-primary/10">
            <FileText className="w-6 h-6 text-primary" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-foreground">Reports</h1>
            <p className="text-muted-foreground">Diagnostic PDF reports from predictions</p>
          </div>
        </div>
        <Button className="bg-gradient-to-r from-teal-600 to-teal-500" onClick={openGenerate}>
          <Plus className="w-4 h-4 mr-2" /> Generate Report
        </Button>
      </div>

      {error && (
        <div className="p-3 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-600 dark:text-red-400 text-sm flex items-center gap-2">
          <AlertCircle className="w-4 h-4 shrink-0" /> {error}
        </div>
      )}

      {/* Generate dialog (inline panel) */}
      {showGenerate && (
        <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
          <Card className="border-0 shadow-md border-l-4 border-l-primary">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Select a prediction</CardTitle>
                  <CardDescription>Pick a prediction to generate a PDF report from</CardDescription>
                </div>
                <Button variant="ghost" size="icon" onClick={() => setShowGenerate(false)}>
                  <X className="w-4 h-4" />
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              {predLoading ? (
                <div className="flex items-center justify-center py-8 text-muted-foreground">
                  <Loader2 className="w-5 h-5 animate-spin mr-2" /> Loading predictions...
                </div>
              ) : predictions.length === 0 ? (
                <p className="text-sm text-muted-foreground py-4 text-center">
                  No predictions yet. Run a detection from the Disease Detection page first.
                </p>
              ) : (
                <div className="space-y-2 max-h-80 overflow-y-auto">
                  {predictions.map((p) => (
                    <div
                      key={p.id}
                      className="flex items-center justify-between p-3 rounded-lg bg-muted/50 hover:bg-muted transition-colors"
                    >
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2">
                          <p className="font-medium text-foreground text-sm">{p.prediction_label}</p>
                          <Badge variant="outline" className="text-xs">{p.model_name}</Badge>
                        </div>
                        <p className="text-xs text-muted-foreground mt-0.5">
                          {(p.confidence * 100).toFixed(1)}% confidence · {p.created_at ? new Date(p.created_at).toLocaleString() : ""}
                        </p>
                      </div>
                      <Button
                        size="sm"
                        onClick={() => generateFor(p.id)}
                        disabled={generatingId === p.id}
                      >
                        {generatingId === p.id ? (
                          <><Loader2 className="w-3 h-3 mr-1 animate-spin" /> Generating...</>
                        ) : (
                          "Generate"
                        )}
                      </Button>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </motion.div>
      )}

      {/* Reports list */}
      {loading ? (
        <div className="flex items-center justify-center py-16 text-muted-foreground">
          <Loader2 className="w-5 h-5 animate-spin mr-2" /> Loading reports...
        </div>
      ) : reports.length === 0 ? (
        <Card className="border-0 shadow-md">
          <CardContent className="p-12 text-center text-muted-foreground">
            <FileText className="w-10 h-10 mx-auto mb-3 opacity-30" />
            <p>No reports yet</p>
            <p className="text-sm mt-1">Click &quot;Generate Report&quot; to create one from a prediction.</p>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-3">
          {reports.map((report, index) => (
            <motion.div
              key={report.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: Math.min(index * 0.05, 0.4) }}
            >
              <Card className="border-0 shadow-md">
                <CardContent className="p-4">
                  <div className="flex items-center justify-between gap-4">
                    <div className="flex items-center gap-4 min-w-0">
                      <div className="p-2 rounded-lg bg-primary/10 shrink-0">
                        <FileText className="w-5 h-5 text-primary" />
                      </div>
                      <div className="min-w-0">
                        <h3 className="font-medium text-foreground truncate">
                          Report #{report.id}
                          {report.prediction_label && <span className="text-muted-foreground"> · {report.prediction_label}</span>}
                        </h3>
                        <p className="text-sm text-muted-foreground">
                          {report.created_at ? new Date(report.created_at).toLocaleString() : "—"}
                          {report.model_name && <span> · {report.model_name}</span>}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      <Badge variant="outline" className="text-xs">PDF</Badge>
                      <Button size="sm" variant="ghost" onClick={() => downloadReport(report)}>
                        <Download className="w-4 h-4" />
                      </Button>
                      <Button size="sm" variant="ghost" onClick={() => deleteReport(report)}>
                        <Trash2 className="w-4 h-4 text-red-500" />
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>
      )}
    </div>
  );
}
