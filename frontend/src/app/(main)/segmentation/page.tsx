"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Scissors, Upload, Layers, Loader2, AlertCircle, Download, CheckCircle2, ImageIcon } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { processingApi } from "@/lib/api";
import { extractApiError } from "@/lib/utils";
import type { SegmentMethod } from "@/lib/types";

const METHODS: { id: SegmentMethod; name: string; description: string }[] = [
  { id: "threshold", name: "Threshold", description: "Intensity-based segmentation using Otsu's method" },
  { id: "watershed", name: "Watershed", description: "Morphological region-growing segmentation" },
  { id: "kmeans", name: "K-Means", description: "Color/intensity clustering segmentation" },
];

export default function SegmentationPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>("");
  const [method, setMethod] = useState<SegmentMethod>("threshold");
  const [resultUrl, setResultUrl] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      if (resultUrl) URL.revokeObjectURL(resultUrl);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleFileSelect = useCallback((file: File) => {
    setError("");
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    if (resultUrl) URL.revokeObjectURL(resultUrl);
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResultUrl("");
  }, [previewUrl, resultUrl]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) handleFileSelect(file);
  }, [handleFileSelect]);

  const runSegmentation = async () => {
    if (!selectedFile) return;
    setError("");
    setLoading(true);
    if (resultUrl) URL.revokeObjectURL(resultUrl);
    setResultUrl("");
    try {
      const blob = await processingApi.segment(selectedFile, method);
      setResultUrl(URL.createObjectURL(blob));
    } catch (err) {
      setError(extractApiError(err, "Segmentation failed. Is the backend running?"));
    } finally {
      setLoading(false);
    }
  };

  const download = () => {
    if (!resultUrl || !selectedFile) return;
    const a = document.createElement("a");
    a.href = resultUrl;
    a.download = `segmented_${method}_${selectedFile.name}`;
    a.click();
  };

  const clearAll = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    if (resultUrl) URL.revokeObjectURL(resultUrl);
    setSelectedFile(null);
    setPreviewUrl("");
    setResultUrl("");
    setError("");
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <div className="p-3 rounded-xl bg-primary/10">
          <Scissors className="w-6 h-6 text-primary" />
        </div>
        <div>
          <h1 className="text-2xl font-bold text-foreground">Image Segmentation</h1>
          <p className="text-muted-foreground">Isolate regions of interest in medical images</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Upload */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
          <Card className="border-0 shadow-md h-full">
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><Upload className="w-5 h-5" /> Upload Image</CardTitle>
              <CardDescription>Upload a medical image for segmentation</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {!previewUrl ? (
                <div
                  onDrop={handleDrop}
                  onDragOver={(e) => e.preventDefault()}
                  onClick={() => fileInputRef.current?.click()}
                  className="border-2 border-dashed border-muted-foreground/25 rounded-xl p-12 text-center hover:border-primary/50 transition-colors cursor-pointer"
                >
                  <Layers className="w-12 h-12 mx-auto text-muted-foreground/50 mb-4" />
                  <p className="text-muted-foreground mb-2">Drop your medical image here or click to browse</p>
                  <p className="text-sm text-muted-foreground/70">Supports: PNG, JPG, TIFF</p>
                </div>
              ) : (
                <div className="space-y-3">
                  <div className="relative rounded-xl overflow-hidden bg-black/5 flex items-center justify-center min-h-[200px]">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={previewUrl} alt="preview" className="max-h-[300px] object-contain" />
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <ImageIcon className="w-4 h-4 text-muted-foreground" />
                      <span className="text-sm text-muted-foreground">{selectedFile?.name}</span>
                    </div>
                    <Button variant="outline" size="sm" onClick={clearAll}>Change Image</Button>
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

        {/* Method Selection */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
          <Card className="border-0 shadow-md h-full">
            <CardHeader>
              <CardTitle>Segmentation Method</CardTitle>
              <CardDescription>Choose the algorithm to apply</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {METHODS.map((m) => (
                <div
                  key={m.id}
                  onClick={() => setMethod(m.id)}
                  className={`p-4 rounded-lg cursor-pointer border-2 transition-all ${
                    method === m.id ? "border-primary bg-primary/5" : "border-transparent bg-muted/50 hover:bg-muted"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <p className="font-medium text-foreground">{m.name}</p>
                    {method === m.id && <CheckCircle2 className="w-4 h-4 text-primary" />}
                  </div>
                  <p className="text-sm text-muted-foreground mt-1">{m.description}</p>
                </div>
              ))}
              <Button
                className="w-full mt-2 bg-gradient-to-r from-teal-600 to-teal-500"
                onClick={runSegmentation}
                disabled={!selectedFile || loading}
              >
                {loading ? (
                  <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Segmenting...</>
                ) : (
                  <>Run Segmentation</>
                )}
              </Button>
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* Result */}
      <AnimatePresence>
        {resultUrl && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }}>
            <Card className="border-0 shadow-lg border-t-4 border-t-primary">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="flex items-center gap-2">
                      <CheckCircle2 className="w-5 h-5 text-primary" /> Segmentation Result
                    </CardTitle>
                    <CardDescription>Method: <span className="capitalize font-medium">{method}</span></CardDescription>
                  </div>
                  <Button variant="outline" size="sm" onClick={download}>
                    <Download className="w-4 h-4 mr-2" /> Download
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Badge variant="outline" className="text-xs">Original</Badge>
                    <div className="relative rounded-xl overflow-hidden bg-black/5 flex items-center justify-center aspect-square">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={previewUrl} alt="original" className="w-full h-full object-contain" />
                    </div>
                  </div>
                  <div className="space-y-2">
                    <Badge className="text-xs">Segmented</Badge>
                    <div className="relative rounded-xl overflow-hidden bg-black/5 flex items-center justify-center aspect-square">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={resultUrl} alt="segmented" className="w-full h-full object-contain" />
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
