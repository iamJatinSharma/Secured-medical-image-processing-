"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Stethoscope, Upload, Brain, Image as ImageIcon, Loader2, CheckCircle2, AlertCircle } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { api } from "@/lib/api";

interface ModelInfo {
  model_name: string;
  display_name: string;
  classes: string[];
  image_type: string;
  description: string;
}

interface PredictionResult {
  id: number;
  prediction_label: string;
  confidence: number;
  all_probabilities: Record<string, number>;
  processing_time_ms: number;
  model_name: string;
}

export default function DetectionPage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Fetch available models on mount
  useEffect(() => {
    api.get("/predict/models").then((res) => {
      setModels(res.data);
      if (res.data.length > 0) setSelectedModel(res.data[0].model_name);
    }).catch(() => {
      // Models endpoint not available yet
      setModels([
        { model_name: "resnet50_breast", display_name: "ResNet-50 (Breast Cancer)", classes: ["Normal", "Malignant"], image_type: "breast", description: "Breast histopathology" },
        { model_name: "resnet50_lung", display_name: "ResNet-50 (Lung Cancer)", classes: ["Normal", "Adenocarcinoma", "Squamous Cell"], image_type: "lung", description: "Lung cancer subtype" },
        { model_name: "resnet50_brain", display_name: "ResNet-50 (Brain Tumor)", classes: ["No Tumor", "Glioma", "Meningioma", "Pituitary"], image_type: "brain", description: "Brain tumor MRI" },
        { model_name: "resnet50_skin", display_name: "ResNet-50 (Skin Lesion)", classes: ["Benign", "Malignant"], image_type: "skin", description: "Skin lesion classification" },
      ]);
      setSelectedModel("resnet50_breast");
    });
  }, []);

  const handleFileSelect = useCallback((file: File) => {
    setSelectedFile(file);
    setResult(null);
    setError(null);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      handleFileSelect(file);
    }
  }, [handleFileSelect]);

  const handleAnalyze = async () => {
    if (!selectedFile || !selectedModel) return;
    setError(null);
    setResult(null);

    try {
      // Step 1: Upload image
      setIsUploading(true);
      const modelConfig = models.find((m) => m.model_name === selectedModel);
      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("image_type", modelConfig?.image_type || "breast");

      const uploadRes = await api.post("/images/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setIsUploading(false);

      // Step 2: Run prediction
      setIsPredicting(true);
      const predictRes = await api.post("/predict/detect", {
        image_id: uploadRes.data.id,
        model_name: selectedModel,
      });
      setIsPredicting(false);

      setResult(predictRes.data);
    } catch (err: unknown) {
      setIsUploading(false);
      setIsPredicting(false);
      const axiosErr = err as { response?: { data?: { detail?: string } } };
      setError(axiosErr.response?.data?.detail || "Analysis failed. Is the backend running?");
    }
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center gap-4">
        <div className="p-3 rounded-xl bg-primary/10">
          <Stethoscope className="w-6 h-6 text-primary" />
        </div>
        <div>
          <h1 className="text-2xl font-bold text-foreground">Disease Detection</h1>
          <p className="text-muted-foreground">AI-powered medical image analysis</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Upload + Preview */}
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
                Upload for Analysis
              </CardTitle>
              <CardDescription>Upload medical images for AI-powered disease detection</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {!previewUrl ? (
                <div
                  onDrop={handleDrop}
                  onDragOver={(e) => e.preventDefault()}
                  onClick={() => fileInputRef.current?.click()}
                  className="border-2 border-dashed border-muted-foreground/25 rounded-xl p-12 text-center hover:border-primary/50 transition-colors cursor-pointer"
                >
                  <Brain className="w-12 h-12 mx-auto text-muted-foreground/50 mb-4" />
                  <p className="text-muted-foreground mb-2">Drop your medical image here or click to browse</p>
                  <p className="text-sm text-muted-foreground/70">Supports: Mammograms, X-rays, CT scans, Histopathology</p>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="relative rounded-xl overflow-hidden bg-black/5 flex items-center justify-center min-h-[300px]">
                    <img
                      src={previewUrl}
                      alt="Medical image preview"
                      className="max-h-[400px] object-contain"
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <ImageIcon className="w-4 h-4 text-muted-foreground" />
                      <span className="text-sm text-muted-foreground">{selectedFile?.name}</span>
                      <Badge variant="secondary">{(selectedFile?.size ? selectedFile.size / 1024 : 0).toFixed(0)} KB</Badge>
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        setSelectedFile(null);
                        setPreviewUrl(null);
                        setResult(null);
                      }}
                    >
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

              {/* Analyze Button */}
              {selectedFile && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                  <Button
                    onClick={handleAnalyze}
                    disabled={isUploading || isPredicting}
                    className="w-full bg-gradient-to-r from-teal-600 to-teal-500 hover:from-teal-700 hover:to-teal-600 shadow-lg shadow-teal-500/25 h-12 text-base"
                  >
                    {isUploading ? (
                      <><Loader2 className="mr-2 h-5 w-5 animate-spin" /> Uploading...</>
                    ) : isPredicting ? (
                      <><Loader2 className="mr-2 h-5 w-5 animate-spin" /> Analyzing...</>
                    ) : (
                      <><Stethoscope className="mr-2 h-5 w-5" /> Analyze Image</>
                    )}
                  </Button>
                </motion.div>
              )}

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
              <CardTitle>Detection Models</CardTitle>
              <CardDescription>Select model for analysis</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {models.map((model) => (
                <div
                  key={model.model_name}
                  onClick={() => setSelectedModel(model.model_name)}
                  className={`p-3 rounded-lg transition-all cursor-pointer border-2 ${
                    selectedModel === model.model_name
                      ? "border-teal-500 bg-teal-50 dark:bg-teal-900/20"
                      : "border-transparent bg-muted/50 hover:bg-muted"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-medium text-foreground text-sm">{model.display_name}</span>
                    {selectedModel === model.model_name && (
                      <CheckCircle2 className="w-4 h-4 text-teal-600" />
                    )}
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">{model.description}</p>
                  <div className="flex gap-1 mt-2 flex-wrap">
                    {model.classes.map((cls) => (
                      <Badge key={cls} variant="secondary" className="text-xs">
                        {cls}
                      </Badge>
                    ))}
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* Results Section */}
      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <Card className="border-0 shadow-lg border-t-4 border-t-teal-500">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <CheckCircle2 className="w-5 h-5 text-teal-600" />
                  Analysis Results
                </CardTitle>
                <CardDescription>
                  Processed in {result.processing_time_ms}ms using {models.find(m => m.model_name === result.model_name)?.display_name}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Primary Result */}
                  <div className="space-y-4">
                    <div className="p-6 rounded-xl bg-gradient-to-br from-teal-50 to-emerald-50 dark:from-teal-900/20 dark:to-emerald-900/20 border border-teal-200 dark:border-teal-800">
                      <p className="text-sm text-muted-foreground mb-1">Prediction</p>
                      <p className="text-2xl font-bold text-foreground">{result.prediction_label}</p>
                      <div className="mt-3">
                        <p className="text-sm text-muted-foreground mb-1">Confidence</p>
                        <div className="flex items-center gap-3">
                          <div className="flex-1 h-3 rounded-full bg-muted overflow-hidden">
                            <motion.div
                              initial={{ width: 0 }}
                              animate={{ width: `${result.confidence * 100}%` }}
                              transition={{ duration: 0.8, ease: "easeOut" }}
                              className="h-full rounded-full bg-gradient-to-r from-teal-500 to-emerald-500"
                            />
                          </div>
                          <span className="font-bold text-teal-600">{(result.confidence * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* All Probabilities */}
                  <div className="space-y-3">
                    <p className="text-sm font-medium text-muted-foreground">Class Probabilities</p>
                    {Object.entries(result.all_probabilities)
                      .sort(([, a], [, b]) => b - a)
                      .map(([cls, prob]) => (
                        <div key={cls} className="space-y-1">
                          <div className="flex justify-between text-sm">
                            <span className="text-foreground">{cls}</span>
                            <span className="text-muted-foreground">{(prob * 100).toFixed(1)}%</span>
                          </div>
                          <div className="h-2 rounded-full bg-muted overflow-hidden">
                            <motion.div
                              initial={{ width: 0 }}
                              animate={{ width: `${prob * 100}%` }}
                              transition={{ duration: 0.6, ease: "easeOut" }}
                              className={`h-full rounded-full ${
                                cls === result.prediction_label
                                  ? "bg-gradient-to-r from-teal-500 to-emerald-500"
                                  : "bg-slate-300 dark:bg-slate-600"
                              }`}
                            />
                          </div>
                        </div>
                      ))}
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
