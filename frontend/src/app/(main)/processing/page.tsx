"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Image as ImageIcon, Upload, Wand2, Loader2, AlertCircle, CheckCircle2, X, Download } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { processingApi } from "@/lib/api";
import { extractApiError } from "@/lib/utils";
import type { DenoiseMethod, EnhanceMethod, SegmentMethod } from "@/lib/types";

type OperationId = "denoise" | "enhance" | "segment";

interface OperationResult {
  id: OperationId;
  method: string;
  imageUrl: string;
  loading: boolean;
}

const DENOISE_METHODS: { id: DenoiseMethod; label: string }[] = [
  { id: "nlm", label: "Non-Local Means" },
  { id: "bilateral", label: "Bilateral Filter" },
  { id: "gaussian", label: "Gaussian Blur" },
];

const ENHANCE_METHODS: { id: EnhanceMethod; label: string }[] = [
  { id: "clahe", label: "CLAHE" },
  { id: "detail", label: "Detail Enhance" },
  { id: "histogram", label: "Histogram Eq." },
];

const SEGMENT_METHODS: { id: SegmentMethod; label: string }[] = [
  { id: "threshold", label: "Threshold" },
  { id: "watershed", label: "Watershed" },
  { id: "kmeans", label: "K-Means" },
];

export default function ProcessingPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>("");
  const [results, setResults] = useState<OperationResult[]>([]);
  const [error, setError] = useState<string>("");
  const [denoiseMethod, setDenoiseMethod] = useState<DenoiseMethod>("nlm");
  const [enhanceMethod, setEnhanceMethod] = useState<EnhanceMethod>("clahe");
  const [segmentMethod, setSegmentMethod] = useState<SegmentMethod>("threshold");
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      results.forEach((r) => r.imageUrl && URL.revokeObjectURL(r.imageUrl));
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleFileSelect = useCallback((file: File) => {
    setError("");
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    results.forEach((r) => r.imageUrl && URL.revokeObjectURL(r.imageUrl));
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResults([]);
  }, [previewUrl, results]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) handleFileSelect(file);
  }, [handleFileSelect]);

  const runOperation = async (id: OperationId) => {
    if (!selectedFile) return;
    setError("");

    const method =
      id === "denoise" ? denoiseMethod :
      id === "enhance" ? enhanceMethod :
      segmentMethod;

    // Remove existing result for this op
    setResults((prev) => {
      const existing = prev.find((r) => r.id === id);
      if (existing?.imageUrl) URL.revokeObjectURL(existing.imageUrl);
      return prev.filter((r) => r.id !== id);
    });

    setResults((prev) => [...prev, { id, method, imageUrl: "", loading: true }]);

    try {
      let blob: Blob;
      if (id === "denoise") blob = await processingApi.denoise(selectedFile, denoiseMethod);
      else if (id === "enhance") blob = await processingApi.enhance(selectedFile, enhanceMethod);
      else blob = await processingApi.segment(selectedFile, segmentMethod);

      const url = URL.createObjectURL(blob);
      setResults((prev) => prev.map((r) => (r.id === id ? { ...r, imageUrl: url, loading: false } : r)));
    } catch (err) {
      setResults((prev) => prev.filter((r) => r.id !== id));
      setError(extractApiError(err, `Failed to ${id} image. Is the backend running?`));
    }
  };

  const removeResult = (id: OperationId) => {
    setResults((prev) => {
      const r = prev.find((x) => x.id === id);
      if (r?.imageUrl) URL.revokeObjectURL(r.imageUrl);
      return prev.filter((x) => x.id !== id);
    });
  };

  const downloadResult = (id: OperationId) => {
    const r = results.find((x) => x.id === id);
    if (!r?.imageUrl || !selectedFile) return;
    const a = document.createElement("a");
    a.href = r.imageUrl;
    a.download = `${id}_${r.method}_${selectedFile.name}`;
    a.click();
  };

  const clearImage = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    results.forEach((r) => r.imageUrl && URL.revokeObjectURL(r.imageUrl));
    setSelectedFile(null);
    setPreviewUrl("");
    setResults([]);
    setError("");
  };

  const operations: { id: OperationId; title: string; description: string; methods: { id: string; label: string }[]; selected: string; setSelected: (m: string) => void; color: string }[] = [
    {
      id: "denoise",
      title: "Denoise",
      description: "Reduce noise while preserving edges",
      methods: DENOISE_METHODS,
      selected: denoiseMethod,
      setSelected: (m) => setDenoiseMethod(m as DenoiseMethod),
      color: "from-blue-500 to-cyan-500",
    },
    {
      id: "enhance",
      title: "Enhance",
      description: "Improve contrast and detail",
      methods: ENHANCE_METHODS,
      selected: enhanceMethod,
      setSelected: (m) => setEnhanceMethod(m as EnhanceMethod),
      color: "from-amber-500 to-orange-500",
    },
    {
      id: "segment",
      title: "Segment",
      description: "Isolate regions of interest",
      methods: SEGMENT_METHODS,
      selected: segmentMethod,
      setSelected: (m) => setSegmentMethod(m as SegmentMethod),
      color: "from-violet-500 to-purple-500",
    },
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <div className="p-3 rounded-xl bg-primary/10">
          <ImageIcon className="w-6 h-6 text-primary" />
        </div>
        <div>
          <h1 className="text-2xl font-bold text-foreground">Image Processing</h1>
          <p className="text-muted-foreground">Denoise, enhance, and segment medical images</p>
        </div>
      </div>

      {/* Upload */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
        <Card className="border-0 shadow-md">
          <CardHeader>
            <CardTitle className="flex items-center gap-2"><Upload className="w-5 h-5" /> Upload Image</CardTitle>
            <CardDescription>Drag and drop or click to upload a medical image</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {!previewUrl ? (
              <div
                onDrop={handleDrop}
                onDragOver={(e) => e.preventDefault()}
                onClick={() => fileInputRef.current?.click()}
                className="border-2 border-dashed border-muted-foreground/25 rounded-xl p-12 text-center hover:border-primary/50 transition-colors cursor-pointer"
              >
                <Upload className="w-12 h-12 mx-auto text-muted-foreground/50 mb-4" />
                <p className="text-muted-foreground mb-2">Drop your medical image here or click to browse</p>
                <p className="text-sm text-muted-foreground/70">Supports: PNG, JPG, TIFF</p>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="relative rounded-xl overflow-hidden bg-black/5 flex items-center justify-center min-h-[200px]">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={previewUrl} alt="preview" className="max-h-[300px] object-contain" />
                </div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <ImageIcon className="w-4 h-4 text-muted-foreground" />
                    <span className="text-sm text-muted-foreground">{selectedFile?.name}</span>
                    <Badge variant="secondary">{((selectedFile?.size ?? 0) / 1024).toFixed(0)} KB</Badge>
                  </div>
                  <Button variant="outline" size="sm" onClick={clearImage}>Change Image</Button>
                </div>
              </div>
            )}
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) handleFileSelect(f);
              }}
            />
            {error && (
              <div className="p-3 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-600 dark:text-red-400 text-sm flex items-center gap-2">
                <AlertCircle className="w-4 h-4 shrink-0" />
                {error}
              </div>
            )}
          </CardContent>
        </Card>
      </motion.div>

      {/* Operations */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {operations.map((op, idx) => {
          const isRunning = results.some((r) => r.id === op.id && r.loading);
          const isDone = results.some((r) => r.id === op.id && !r.loading);
          return (
            <motion.div key={op.id} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 + idx * 0.1 }}>
              <Card className="border-0 shadow-md h-full">
                <CardHeader>
                  <div className="flex items-center gap-2">
                    <Wand2 className={`w-5 h-5 bg-gradient-to-r ${op.color} bg-clip-text text-transparent`} />
                    <CardTitle className="text-lg">{op.title}</CardTitle>
                    {isDone && <Badge variant="secondary" className="text-xs">Done</Badge>}
                  </div>
                  <CardDescription>{op.description}</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div>
                    <label className="text-xs text-muted-foreground mb-1.5 block">Method</label>
                    <div className="grid grid-cols-3 gap-2">
                      {op.methods.map((m) => (
                        <button
                          key={m.id}
                          onClick={() => op.setSelected(m.id)}
                          className={`px-2 py-1.5 rounded-md text-xs font-medium border-2 transition-all ${
                            op.selected === m.id
                              ? "border-primary bg-primary/5 text-primary"
                              : "border-transparent bg-muted/50 hover:bg-muted"
                          }`}
                        >
                          {m.label}
                        </button>
                      ))}
                    </div>
                  </div>
                  <Button
                    className={`w-full bg-gradient-to-r ${op.color} hover:opacity-90`}
                    onClick={() => runOperation(op.id)}
                    disabled={!selectedFile || isRunning}
                  >
                    {isRunning ? (
                      <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Processing...</>
                    ) : isDone ? (
                      <>Re-run</>
                    ) : (
                      <>Run {op.title}</>
                    )}
                  </Button>
                </CardContent>
              </Card>
            </motion.div>
          );
        })}
      </div>

      {/* Results */}
      <AnimatePresence>
        {results.length > 0 && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }}>
            <Card className="border-0 shadow-lg border-t-4 border-t-primary">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <CheckCircle2 className="w-5 h-5 text-primary" /> Processing Results
                </CardTitle>
                <CardDescription>Original image and processed outputs</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  <div className="space-y-2">
                    <Badge variant="outline" className="text-xs">Original</Badge>
                    <div className="relative rounded-xl overflow-hidden bg-black/5 flex items-center justify-center aspect-square">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={previewUrl} alt="original" className="w-full h-full object-contain" />
                    </div>
                  </div>
                  {results
                    .filter((r) => !r.loading && r.imageUrl)
                    .map((r) => (
                      <div key={r.id} className="space-y-2">
                        <div className="flex items-center justify-between">
                          <Badge className="text-xs capitalize">{r.id} · {r.method}</Badge>
                          <div className="flex items-center gap-1">
                            <Button variant="ghost" size="icon" className="h-6 w-6" onClick={() => downloadResult(r.id)}>
                              <Download className="h-3 w-3" />
                            </Button>
                            <Button variant="ghost" size="icon" className="h-6 w-6" onClick={() => removeResult(r.id)}>
                              <X className="h-3 w-3" />
                            </Button>
                          </div>
                        </div>
                        <div className="relative rounded-xl overflow-hidden bg-black/5 flex items-center justify-center aspect-square">
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img src={r.imageUrl} alt={r.id} className="w-full h-full object-contain" />
                        </div>
                      </div>
                    ))}
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
