"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Brain, Lightbulb, Network, BarChart3, Upload, Loader2, ImageIcon, CheckCircle2, AlertCircle, X } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { xaiApi } from "@/lib/api";

interface XaiResult {
  method: string;
  imageUrl: string;
  loading: boolean;
}

const XAI_METHODS = [
  {
    id: "gradcam",
    title: "Grad-CAM",
    description: "Gradient-weighted Class Activation Mapping - highlights important regions using gradients from the model's final convolutional layer",
    icon: Lightbulb,
    color: "from-amber-500 to-orange-500",
    bgColor: "from-amber-50 to-orange-50 dark:from-amber-900/20 dark:to-orange-900/20",
    borderColor: "border-amber-200 dark:border-amber-800",
  },
  {
    id: "lime",
    title: "LIME",
    description: "Local Interpretable Model-agnostic Explanations - identifies important superpixels by perturbing the input",
    icon: Network,
    color: "from-violet-500 to-purple-500",
    bgColor: "from-violet-50 to-purple-50 dark:from-violet-900/20 dark:to-purple-900/20",
    borderColor: "border-violet-200 dark:border-violet-800",
  },
  {
    id: "shap",
    title: "SHAP",
    description: "SHapley Additive exPlanations - game theory based feature attribution showing pixel-level importance",
    icon: BarChart3,
    color: "from-emerald-500 to-teal-500",
    bgColor: "from-emerald-50 to-teal-50 dark:from-emerald-900/20 dark:to-teal-900/20",
    borderColor: "border-emerald-200 dark:border-emerald-800",
  },
];

const MODELS = [
  { id: "resnet50_breast", name: "ResNet-50 (Breast)", description: "Breast histopathology" },
  { id: "resnet50_lung", name: "ResNet-50 (Lung)", description: "Lung cancer detection" },
  { id: "resnet50_brain", name: "ResNet-50 (Brain)", description: "Brain tumor MRI" },
  { id: "resnet50_skin", name: "ResNet-50 (Skin)", description: "Skin lesion classification" },
];

export default function XAIPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>("");
  const [selectedModel, setSelectedModel] = useState<string>("resnet50_breast");
  const [results, setResults] = useState<XaiResult[]>([]);
  const [error, setError] = useState<string>("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Cleanup object URLs on unmount
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      results.forEach((r) => {
        if (r.imageUrl) URL.revokeObjectURL(r.imageUrl);
      });
    };
  }, []);

  const handleFileSelect = useCallback((file: File) => {
    setSelectedFile(file);
    setError("");
    // Cleanup previous preview
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    // Clear previous results when new file is selected
    results.forEach((r) => {
      if (r.imageUrl) URL.revokeObjectURL(r.imageUrl);
    });
    setResults([]);
  }, [previewUrl, results]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      handleFileSelect(file);
    }
  }, [handleFileSelect]);

  const handleGenerate = async (methodId: string) => {
    if (!selectedFile) return;

    setError("");

    // Check if already generated
    const existingIndex = results.findIndex((r) => r.method === methodId);
    if (existingIndex !== -1) {
      // Remove existing result and regenerate
      const existing = results[existingIndex];
      if (existing.imageUrl) URL.revokeObjectURL(existing.imageUrl);
      setResults((prev) => prev.filter((r) => r.method !== methodId));
    }

    // Add loading state
    setResults((prev) => [...prev, { method: methodId, imageUrl: "", loading: true }]);

    try {
      let blob: Blob;
      switch (methodId) {
        case "gradcam":
          blob = await xaiApi.gradcam(selectedFile, selectedModel);
          break;
        case "lime":
          blob = await xaiApi.lime(selectedFile, selectedModel);
          break;
        case "shap":
          blob = await xaiApi.shap(selectedFile, selectedModel);
          break;
        default:
          throw new Error("Unknown method");
      }

      const imageUrl = URL.createObjectURL(blob);
      setResults((prev) =>
        prev.map((r) => (r.method === methodId ? { ...r, imageUrl, loading: false } : r))
      );
    } catch (err: unknown) {
      // Remove loading state on error
      setResults((prev) => prev.filter((r) => r.method !== methodId));
      const axiosErr = err as { response?: { data?: { detail?: string } }; message?: string };
      setError(axiosErr.response?.data?.detail || axiosErr.message || "Failed to generate explanation. Is the backend running?");
    }
  };

  const removeResult = (methodId: string) => {
    setResults((prev) => {
      const result = prev.find((r) => r.method === methodId);
      if (result?.imageUrl) URL.revokeObjectURL(result.imageUrl);
      return prev.filter((r) => r.method !== methodId);
    });
  };

  const clearImage = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    results.forEach((r) => {
      if (r.imageUrl) URL.revokeObjectURL(r.imageUrl);
    });
    setSelectedFile(null);
    setPreviewUrl("");
    setResults([]);
    setError("");
  };

  const isMethodGenerated = (methodId: string) => {
    return results.some((r) => r.method === methodId && !r.loading);
  };

  const isMethodLoading = (methodId: string) => {
    return results.some((r) => r.method === methodId && r.loading);
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center gap-4">
        <div className="p-3 rounded-xl bg-primary/10">
          <Brain className="w-6 h-6 text-primary" />
        </div>
        <div>
          <h1 className="text-2xl font-bold text-foreground">Explainable AI</h1>
          <p className="text-muted-foreground">Understand AI decisions with XAI techniques</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Upload Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="lg:col-span-2"
        >
          <Card className="border-0 shadow-md h-full">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Upload className="w-5 h-5" />
                Upload Image
              </CardTitle>
              <CardDescription>Upload a medical image to generate explanations</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {!previewUrl ? (
                <div
                  onDrop={handleDrop}
                  onDragOver={(e) => e.preventDefault()}
                  onClick={() => fileInputRef.current?.click()}
                  className="border-2 border-dashed border-muted-foreground/25 rounded-xl p-12 text-center hover:border-primary/50 transition-colors cursor-pointer"
                >
                  <ImageIcon className="w-12 h-12 mx-auto text-muted-foreground/50 mb-4" />
                  <p className="text-muted-foreground mb-2">Drop your medical image here or click to browse</p>
                  <p className="text-sm text-muted-foreground/70">Supports: Mammograms, X-rays, CT scans, Histopathology</p>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="relative rounded-xl overflow-hidden bg-black/5 flex items-center justify-center min-h-[200px]">
                    <img
                      src={previewUrl}
                      alt="Medical image preview"
                      className="max-h-[300px] object-contain"
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <ImageIcon className="w-4 h-4 text-muted-foreground" />
                      <span className="text-sm text-muted-foreground">{selectedFile?.name}</span>
                      <Badge variant="secondary">{(selectedFile?.size ? selectedFile.size / 1024 : 0).toFixed(0)} KB</Badge>
                    </div>
                    <Button variant="outline" size="sm" onClick={clearImage}>
                      Change Image
                    </Button>
                  </div>
                </div>
              )}

              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) handleFileSelect(file);
                }}
              />

              {/* Error */}
              {error && (
                <div className="p-3 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-600 dark:text-red-400 text-sm flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 shrink-0" />
                  {error}
                </div>
              )}
            </CardContent>
          </Card>
        </motion.div>

        {/* Model Selection */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <Card className="border-0 shadow-md h-full">
            <CardHeader>
              <CardTitle>Select Model</CardTitle>
              <CardDescription>Choose model for explanation</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {MODELS.map((model) => (
                <div
                  key={model.id}
                  onClick={() => setSelectedModel(model.id)}
                  className={`p-3 rounded-lg transition-all cursor-pointer border-2 ${
                    selectedModel === model.id
                      ? "border-primary bg-primary/5"
                      : "border-transparent bg-muted/50 hover:bg-muted"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-medium text-foreground text-sm">{model.name}</span>
                    {selectedModel === model.id && (
                      <CheckCircle2 className="w-4 h-4 text-primary" />
                    )}
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">{model.description}</p>
                </div>
              ))}
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* XAI Method Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {XAI_METHODS.map((method, index) => {
          const Icon = method.icon;
          const isLoading = isMethodLoading(method.id);
          const isGenerated = isMethodGenerated(method.id);

          return (
            <motion.div
              key={method.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 + index * 0.1 }}
            >
              <Card className={`border-0 shadow-md h-full ${isGenerated ? `border-t-4 ${method.borderColor}` : ""}`}>
                <CardHeader>
                  <div className="flex items-center gap-3">
                    <div className={`p-2 rounded-lg bg-gradient-to-br ${method.bgColor}`}>
                      <Icon className={`w-5 h-5 bg-gradient-to-br ${method.color} bg-clip-text`} style={{ color: method.id === 'gradcam' ? '#f59e0b' : method.id === 'lime' ? '#8b5cf6' : '#10b981' }} />
                    </div>
                    <div>
                      <CardTitle className="text-lg">{method.title}</CardTitle>
                      {isGenerated && (
                        <Badge variant="secondary" className="mt-1 text-xs">Generated</Badge>
                      )}
                    </div>
                  </div>
                  <CardDescription className="text-sm mt-2">{method.description}</CardDescription>
                </CardHeader>
                <CardContent>
                  <Button
                    className={`w-full bg-gradient-to-r ${method.color} hover:opacity-90 transition-opacity`}
                    onClick={() => handleGenerate(method.id)}
                    disabled={!selectedFile || isLoading}
                  >
                    {isLoading ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Generating...
                      </>
                    ) : isGenerated ? (
                      <>
                        <Lightbulb className="mr-2 h-4 w-4" />
                        Regenerate
                      </>
                    ) : (
                      <>
                        <Lightbulb className="mr-2 h-4 w-4" />
                        Generate
                      </>
                    )}
                  </Button>
                </CardContent>
              </Card>
            </motion.div>
          );
        })}
      </div>

      {/* Results Section */}
      <AnimatePresence>
        {results.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <Card className="border-0 shadow-lg border-t-4 border-t-primary">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <CheckCircle2 className="w-5 h-5 text-primary" />
                  Explanation Results
                </CardTitle>
                <CardDescription>
                  Original image compared with generated explanations
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                  {/* Original Image */}
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <Badge variant="outline" className="text-xs">Original</Badge>
                    </div>
                    <div className="relative rounded-xl overflow-hidden bg-black/5 flex items-center justify-center aspect-square">
                      <img
                        src={previewUrl}
                        alt="Original image"
                        className="w-full h-full object-contain"
                      />
                    </div>
                  </div>

                  {/* Generated Explanations */}
                  {results
                    .filter((r) => !r.loading && r.imageUrl)
                    .map((result) => {
                      const method = XAI_METHODS.find((m) => m.id === result.method);
                      if (!method) return null;

                      return (
                        <div key={result.method} className="space-y-2">
                          <div className="flex items-center justify-between">
                            <Badge className={`bg-gradient-to-r ${method.color} text-white text-xs`}>
                              {method.title}
                            </Badge>
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-6 w-6"
                              onClick={() => removeResult(result.method)}
                            >
                              <X className="h-3 w-3" />
                            </Button>
                          </div>
                          <div className="relative rounded-xl overflow-hidden bg-black/5 flex items-center justify-center aspect-square">
                            <img
                              src={result.imageUrl}
                              alt={`${method.title} explanation`}
                              className="w-full h-full object-contain"
                            />
                          </div>
                        </div>
                      );
                    })}
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
