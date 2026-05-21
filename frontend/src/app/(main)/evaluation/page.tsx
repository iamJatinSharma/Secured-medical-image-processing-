"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { motion } from "framer-motion";
import { BarChart3, TrendingUp, Target, Gauge, Upload, Loader2, AlertCircle, CheckCircle2, ImageIcon } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { metricsApi, processingApi } from "@/lib/api";
import { extractApiError } from "@/lib/utils";
import type { ModelMetrics, CompareResponse } from "@/lib/types";

const MODELS = [
  { id: "resnet50_breast", name: "ResNet-50 (Breast)" },
  { id: "resnet50_lung", name: "ResNet-50 (Lung)" },
  { id: "resnet50_brain", name: "ResNet-50 (Brain)" },
  { id: "resnet50_skin", name: "ResNet-50 (Skin)" },
];

function ImagePicker({
  label,
  file,
  previewUrl,
  onFileSelect,
}: {
  label: string;
  file: File | null;
  previewUrl: string;
  onFileSelect: (file: File) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  return (
    <div className="space-y-2">
      <Badge variant="outline" className="text-xs">{label}</Badge>
      {!previewUrl ? (
        <div
          onClick={() => inputRef.current?.click()}
          onDrop={(e) => {
            e.preventDefault();
            const f = e.dataTransfer.files[0];
            if (f?.type.startsWith("image/")) onFileSelect(f);
          }}
          onDragOver={(e) => e.preventDefault()}
          className="border-2 border-dashed border-muted-foreground/25 rounded-xl p-8 text-center hover:border-primary/50 cursor-pointer aspect-square flex flex-col items-center justify-center"
        >
          <Upload className="w-10 h-10 mx-auto text-muted-foreground/50 mb-2" />
          <p className="text-sm text-muted-foreground">Click to upload</p>
        </div>
      ) : (
        <div className="relative rounded-xl overflow-hidden bg-black/5 flex items-center justify-center aspect-square">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={previewUrl} alt={label} className="w-full h-full object-contain" />
          <button
            onClick={() => inputRef.current?.click()}
            className="absolute top-2 right-2 bg-black/60 text-white text-xs px-2 py-1 rounded hover:bg-black/80"
          >
            Change
          </button>
        </div>
      )}
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onFileSelect(f);
        }}
      />
      {file && (
        <p className="text-xs text-muted-foreground flex items-center gap-1.5">
          <ImageIcon className="w-3 h-3" /> {file.name}
        </p>
      )}
    </div>
  );
}

export default function EvaluationPage() {
  const [selectedModel, setSelectedModel] = useState(MODELS[0].id);
  const [modelMetrics, setModelMetrics] = useState<ModelMetrics | null>(null);
  const [modelLoading, setModelLoading] = useState(false);
  const [modelError, setModelError] = useState<string>("");

  const [file1, setFile1] = useState<File | null>(null);
  const [file2, setFile2] = useState<File | null>(null);
  const [preview1, setPreview1] = useState<string>("");
  const [preview2, setPreview2] = useState<string>("");
  const [compareResult, setCompareResult] = useState<CompareResponse | null>(null);
  const [compareLoading, setCompareLoading] = useState(false);
  const [compareError, setCompareError] = useState<string>("");

  // Load model metrics on selection change
  useEffect(() => {
    let cancelled = false;
    setModelLoading(true);
    setModelError("");
    metricsApi.getModelMetrics(selectedModel)
      .then((m) => { if (!cancelled) setModelMetrics(m); })
      .catch((err) => { if (!cancelled) setModelError(extractApiError(err, "Failed to load model metrics.")); })
      .finally(() => { if (!cancelled) setModelLoading(false); });
    return () => { cancelled = true; };
  }, [selectedModel]);

  // Cleanup blob URLs
  useEffect(() => {
    return () => {
      if (preview1) URL.revokeObjectURL(preview1);
      if (preview2) URL.revokeObjectURL(preview2);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleFile1 = useCallback((f: File) => {
    if (preview1) URL.revokeObjectURL(preview1);
    setFile1(f);
    setPreview1(URL.createObjectURL(f));
    setCompareResult(null);
  }, [preview1]);

  const handleFile2 = useCallback((f: File) => {
    if (preview2) URL.revokeObjectURL(preview2);
    setFile2(f);
    setPreview2(URL.createObjectURL(f));
    setCompareResult(null);
  }, [preview2]);

  const compare = async () => {
    if (!file1 || !file2) return;
    setCompareError("");
    setCompareLoading(true);
    try {
      const result = await processingApi.compare(file1, file2);
      setCompareResult(result);
    } catch (err) {
      setCompareError(extractApiError(err, "Comparison failed."));
    } finally {
      setCompareLoading(false);
    }
  };

  const modelStats = modelMetrics ? [
    { name: "Accuracy", value: `${(modelMetrics.accuracy * 100).toFixed(1)}%`, icon: Target },
    { name: "F1 Score", value: `${(modelMetrics.f1_score * 100).toFixed(1)}%`, icon: BarChart3 },
    { name: "AUC", value: modelMetrics.auc ? modelMetrics.auc.toFixed(3) : "—", icon: TrendingUp },
    { name: "Classes", value: String(modelMetrics.labels?.length ?? 0), icon: Gauge },
  ] : [];

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <div className="p-3 rounded-xl bg-primary/10">
          <BarChart3 className="w-6 h-6 text-primary" />
        </div>
        <div>
          <h1 className="text-2xl font-bold text-foreground">Evaluation Metrics</h1>
          <p className="text-muted-foreground">Model performance and image quality metrics</p>
        </div>
      </div>

      {/* Model selector */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <Card className="border-0 shadow-md">
          <CardHeader>
            <CardTitle>Model Performance</CardTitle>
            <CardDescription>Select a model to view its performance metrics</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-wrap gap-2">
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

            {modelLoading && (
              <div className="flex items-center justify-center py-8 text-muted-foreground">
                <Loader2 className="w-5 h-5 animate-spin mr-2" /> Loading metrics...
              </div>
            )}

            {modelError && (
              <div className="p-3 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-600 dark:text-red-400 text-sm flex items-center gap-2">
                <AlertCircle className="w-4 h-4 shrink-0" /> {modelError}
              </div>
            )}

            {!modelLoading && modelMetrics && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {modelStats.map((s) => {
                  const Icon = s.icon;
                  return (
                    <div key={s.name} className="p-4 rounded-lg bg-muted/50">
                      <div className="flex items-center justify-between mb-2">
                        <Icon className="w-4 h-4 text-primary" />
                      </div>
                      <p className="text-xs text-muted-foreground">{s.name}</p>
                      <p className="text-2xl font-bold text-foreground">{s.value}</p>
                    </div>
                  );
                })}
              </div>
            )}

            {modelMetrics?.confusion_matrix && (
              <div>
                <p className="text-sm font-medium mb-2">Confusion Matrix</p>
                <div className="inline-block border rounded-lg overflow-hidden">
                  <table className="border-collapse">
                    <thead>
                      <tr>
                        <th className="border-b border-r bg-muted/50 px-3 py-2 text-xs"></th>
                        {modelMetrics.labels?.map((l) => (
                          <th key={l} className="border-b bg-muted/50 px-3 py-2 text-xs font-medium">
                            Pred: {l}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {modelMetrics.confusion_matrix.map((row, i) => (
                        <tr key={i}>
                          <th className="border-r bg-muted/50 px-3 py-2 text-xs font-medium text-right">
                            True: {modelMetrics.labels?.[i] ?? i}
                          </th>
                          {row.map((cell, j) => (
                            <td key={j} className={`px-4 py-2 text-center text-sm ${i === j ? "bg-emerald-50 dark:bg-emerald-900/20 font-semibold" : "bg-rose-50 dark:bg-rose-900/20"}`}>
                              {cell}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </motion.div>

      {/* Image Quality Compare */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
        <Card className="border-0 shadow-md">
          <CardHeader>
            <CardTitle>Image Quality Metrics</CardTitle>
            <CardDescription>Upload two images to compute PSNR, SSIM and MSE</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <ImagePicker label="Original" file={file1} previewUrl={preview1} onFileSelect={handleFile1} />
              <ImagePicker label="Processed" file={file2} previewUrl={preview2} onFileSelect={handleFile2} />
            </div>

            <Button
              className="w-full bg-gradient-to-r from-teal-600 to-teal-500"
              onClick={compare}
              disabled={!file1 || !file2 || compareLoading}
            >
              {compareLoading ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Computing...</> : "Calculate Metrics"}
            </Button>

            {compareError && (
              <div className="p-3 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-600 dark:text-red-400 text-sm flex items-center gap-2">
                <AlertCircle className="w-4 h-4 shrink-0" /> {compareError}
              </div>
            )}

            {compareResult && (
              <div className="space-y-3">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  <div className="p-4 rounded-lg bg-muted/50 text-center">
                    <p className="text-xs text-muted-foreground">Peak Signal-to-Noise Ratio</p>
                    <p className="text-2xl font-bold text-foreground mt-1">{compareResult.psnr_db.toFixed(2)} dB</p>
                    <p className="text-xs text-muted-foreground mt-1">PSNR</p>
                  </div>
                  <div className="p-4 rounded-lg bg-muted/50 text-center">
                    <p className="text-xs text-muted-foreground">Structural Similarity</p>
                    <p className="text-2xl font-bold text-foreground mt-1">{compareResult.ssim?.toFixed(4) ?? "—"}</p>
                    <p className="text-xs text-muted-foreground mt-1">SSIM</p>
                  </div>
                  <div className="p-4 rounded-lg bg-muted/50 text-center">
                    <p className="text-xs text-muted-foreground">Mean Squared Error</p>
                    <p className="text-2xl font-bold text-foreground mt-1">{compareResult.mse.toFixed(4)}</p>
                    <p className="text-xs text-muted-foreground mt-1">MSE</p>
                  </div>
                </div>
                <div className={`p-3 rounded-lg flex items-center gap-2 text-sm ${
                  compareResult.psnr_db > 30
                    ? "bg-emerald-50 dark:bg-emerald-900/20 text-emerald-700 dark:text-emerald-400"
                    : compareResult.psnr_db > 20
                    ? "bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-400"
                    : "bg-amber-50 dark:bg-amber-900/20 text-amber-700 dark:text-amber-400"
                }`}>
                  <CheckCircle2 className="w-4 h-4 shrink-0" />
                  {compareResult.identical ? "Images are identical." :
                    compareResult.psnr_db > 30 ? "Excellent image quality (PSNR > 30 dB)." :
                    compareResult.psnr_db > 20 ? "Good image quality (PSNR > 20 dB)." :
                    "Significant differences detected (PSNR ≤ 20 dB)."}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}
