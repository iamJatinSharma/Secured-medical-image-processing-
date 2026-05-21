"use client";

import { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { Activity, Play, Loader2, AlertCircle, CheckCircle2, Database, Cpu } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { trainingApi } from "@/lib/api";
import { extractApiError } from "@/lib/utils";
import type { TrainingConfig, TrainingProgress, TrainingStatus, MedMnistDataset, TrainedModelInfo } from "@/lib/types";

const ARCHITECTURES = [
  { id: "resnet18", name: "ResNet-18", note: "Fastest, good accuracy" },
  { id: "smallcnn", name: "Small CNN", note: "Fastest on CPU" },
  { id: "resnet50", name: "ResNet-50", note: "Highest accuracy, slow on CPU" },
];

interface EpochPoint {
  epoch: number;
  train_loss: number;
  val_loss: number;
  train_acc: number;
  val_acc: number;
}

interface CompletionInfo {
  model_name?: string;
  best_val_acc?: number;
  accuracy?: number;
  f1_score?: number;
  total_time_seconds?: number;
  labels?: string[];
  architecture?: string;
}

function isProgress(p: TrainingStatus["progress"]): p is TrainingProgress {
  return typeof p === "object" && p !== null && "epoch" in p && "total_epochs" in p;
}

export default function TrainingPage() {
  const [config, setConfig] = useState<TrainingConfig>({
    model: "resnet18",
    dataset: "breastmnist",
    epochs: 5,
    batch_size: 32,
    learning_rate: 0.001,
    image_size: 64,
    max_images: null,
  });
  const [datasets, setDatasets] = useState<MedMnistDataset[]>([]);
  const [trained, setTrained] = useState<TrainedModelInfo[]>([]);
  const [status, setStatus] = useState<TrainingStatus["status"]>("idle");
  const [current, setCurrent] = useState<TrainingProgress | null>(null);
  const [history, setHistory] = useState<EpochPoint[]>([]);
  const [completionInfo, setCompletionInfo] = useState<CompletionInfo | null>(null);
  const [error, setError] = useState<string>("");
  const [starting, setStarting] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  const refreshTrained = () => {
    trainingApi.listTrained().then((r) => setTrained(r.models)).catch(() => {});
  };

  useEffect(() => {
    // Initial fetches
    trainingApi.listDatasets()
      .then((r) => setDatasets(r.datasets))
      .catch((err) => setError(extractApiError(err, "Failed to load datasets.")));
    refreshTrained();
    trainingApi.getStatus().then((s) => {
      setStatus(s.status);
      if (isProgress(s.progress)) setCurrent(s.progress);
    }).catch(() => {});

    // WebSocket
    const ws = trainingApi.connectWebSocket((data) => {
      const msg = data as Record<string, unknown>;
      if ("status" in msg && msg.status === "complete") {
        setStatus("complete");
        setCompletionInfo(msg as CompletionInfo);
        refreshTrained();
        return;
      }
      if ("status" in msg && msg.status === "error") {
        setStatus("idle");
        setError((msg.error as string) || "Training failed.");
        return;
      }
      if ("status" in msg && typeof msg.status === "string" && "progress" in msg) {
        setStatus(msg.status as TrainingStatus["status"]);
        if (isProgress(msg.progress as TrainingStatus["progress"])) {
          setCurrent(msg.progress as TrainingProgress);
        }
        return;
      }
      if ("epoch" in msg && "total_epochs" in msg) {
        const p = msg as unknown as TrainingProgress;
        setStatus("training");
        setCurrent(p);
        setHistory((prev) => {
          const filtered = prev.filter((x) => x.epoch !== p.epoch);
          return [...filtered, {
            epoch: p.epoch,
            train_loss: p.train_loss,
            val_loss: p.val_loss,
            train_acc: p.train_acc,
            val_acc: p.val_acc,
          }].sort((a, b) => a.epoch - b.epoch);
        });
      }
    });
    wsRef.current = ws;

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, []);

  const startTraining = async () => {
    setError("");
    setStarting(true);
    setHistory([]);
    setCompletionInfo(null);
    setCurrent(null);
    try {
      const result = await trainingApi.start(config);
      if (result.error) {
        setError(result.error);
      } else {
        setStatus("training");
      }
    } catch (err) {
      setError(extractApiError(err, "Failed to start training."));
    } finally {
      setStarting(false);
    }
  };

  const isTraining = status === "training";
  const selectedDataset = datasets.find((d) => d.id === config.dataset);

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <div className="p-3 rounded-xl bg-primary/10">
          <Activity className="w-6 h-6 text-primary" />
        </div>
        <div>
          <h1 className="text-2xl font-bold text-foreground">Training Monitor</h1>
          <p className="text-muted-foreground">Train real models on MedMNIST datasets with live progress</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Config */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
          <Card className="border-0 shadow-md h-full">
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><Cpu className="w-5 h-5" /> Configuration</CardTitle>
              <CardDescription>Pick architecture, dataset, hyperparameters</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Architecture */}
              <div className="space-y-2">
                <Label>Architecture</Label>
                <div className="space-y-1.5">
                  {ARCHITECTURES.map((a) => (
                    <button
                      key={a.id}
                      onClick={() => setConfig({ ...config, model: a.id })}
                      disabled={isTraining}
                      className={`w-full text-left p-2.5 rounded-md border-2 transition-all ${
                        config.model === a.id ? "border-primary bg-primary/5" : "border-transparent bg-muted/50 hover:bg-muted"
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium">{a.name}</span>
                        {config.model === a.id && <CheckCircle2 className="w-4 h-4 text-primary" />}
                      </div>
                      <p className="text-xs text-muted-foreground">{a.note}</p>
                    </button>
                  ))}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-2">
                  <Label htmlFor="epochs">Epochs</Label>
                  <Input
                    id="epochs"
                    type="number"
                    min={1}
                    max={50}
                    value={config.epochs}
                    onChange={(e) => setConfig({ ...config, epochs: parseInt(e.target.value) || 1 })}
                    disabled={isTraining}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="batch">Batch size</Label>
                  <Input
                    id="batch"
                    type="number"
                    min={1}
                    max={128}
                    value={config.batch_size}
                    onChange={(e) => setConfig({ ...config, batch_size: parseInt(e.target.value) || 1 })}
                    disabled={isTraining}
                  />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-2">
                  <Label htmlFor="lr">Learning rate</Label>
                  <Input
                    id="lr"
                    type="number"
                    step="0.0001"
                    min={0.00001}
                    max={1}
                    value={config.learning_rate}
                    onChange={(e) => setConfig({ ...config, learning_rate: parseFloat(e.target.value) || 0.001 })}
                    disabled={isTraining}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="imsize">Image size</Label>
                  <Input
                    id="imsize"
                    type="number"
                    min={28}
                    max={224}
                    step={8}
                    value={config.image_size}
                    onChange={(e) => setConfig({ ...config, image_size: parseInt(e.target.value) || 64 })}
                    disabled={isTraining}
                  />
                </div>
              </div>
              <div className="space-y-2">
                <Label htmlFor="maxim">Max training samples (0 = all)</Label>
                <Input
                  id="maxim"
                  type="number"
                  min={0}
                  value={config.max_images ?? 0}
                  onChange={(e) => {
                    const v = parseInt(e.target.value);
                    setConfig({ ...config, max_images: isNaN(v) || v === 0 ? null : v });
                  }}
                  disabled={isTraining}
                />
              </div>

              <Button
                className="w-full bg-gradient-to-r from-teal-600 to-teal-500"
                onClick={startTraining}
                disabled={isTraining || starting || !config.dataset}
              >
                {starting ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Starting...</> :
                 isTraining ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Training in progress</> :
                 <><Play className="mr-2 h-4 w-4" /> Start Training</>}
              </Button>
              {error && (
                <div className="p-3 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-600 dark:text-red-400 text-sm flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 shrink-0" /> {error}
                </div>
              )}
            </CardContent>
          </Card>
        </motion.div>

        {/* Dataset picker + live progress */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="lg:col-span-2 space-y-6">
          {/* Datasets */}
          <Card className="border-0 shadow-md">
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><Database className="w-5 h-5" /> Dataset</CardTitle>
              <CardDescription>MedMNIST datasets — auto-downloaded on first use</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
                {datasets.map((d) => (
                  <button
                    key={d.id}
                    onClick={() => setConfig({ ...config, dataset: d.id })}
                    disabled={isTraining}
                    className={`text-left p-3 rounded-lg border-2 transition-all ${
                      config.dataset === d.id ? "border-primary bg-primary/5" : "border-transparent bg-muted/50 hover:bg-muted"
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-medium text-sm">{d.name}</span>
                      {config.dataset === d.id && <CheckCircle2 className="w-4 h-4 text-primary" />}
                    </div>
                    <p className="text-xs text-muted-foreground">{d.modality}</p>
                    <p className="text-xs text-muted-foreground mt-0.5">
                      {d.task} · {d.size_train.toLocaleString()} train samples
                    </p>
                  </button>
                ))}
              </div>
              {selectedDataset && (
                <div className="mt-3 p-3 rounded-lg bg-muted/30 text-xs text-muted-foreground">
                  Classes: {selectedDataset.classes.join(", ")}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Live progress */}
          <Card className="border-0 shadow-md">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Live Progress</CardTitle>
                  <CardDescription>Streaming via WebSocket</CardDescription>
                </div>
                <Badge className={
                  status === "training" ? "bg-blue-500" :
                  status === "complete" ? "bg-emerald-500" :
                  "bg-muted-foreground"
                }>
                  {status}
                </Badge>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              {!current && status === "idle" && (
                <div className="text-center py-8 text-muted-foreground">
                  <Activity className="w-10 h-10 mx-auto mb-3 opacity-30" />
                  <p>No training run yet — pick a dataset and click Start Training.</p>
                </div>
              )}

              {current && (
                <>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="font-medium">Epoch {current.epoch} / {current.total_epochs}</span>
                      <span className="text-muted-foreground">{current.percent_complete.toFixed(1)}% · ETA {current.eta_seconds}s</span>
                    </div>
                    <div className="w-full bg-muted rounded-full h-2">
                      <div
                        className="bg-gradient-to-r from-teal-600 to-teal-500 h-2 rounded-full transition-all"
                        style={{ width: `${current.percent_complete}%` }}
                      />
                    </div>
                  </div>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    <Metric label="Train Loss" value={current.train_loss.toFixed(4)} />
                    <Metric label="Val Loss" value={current.val_loss.toFixed(4)} />
                    <Metric label="Train Acc" value={`${(current.train_acc * 100).toFixed(1)}%`} />
                    <Metric label="Val Acc" value={`${(current.val_acc * 100).toFixed(1)}%`} />
                  </div>
                </>
              )}

              {completionInfo && (
                <div className="p-3 rounded-lg bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800 text-emerald-700 dark:text-emerald-400 text-sm flex items-start gap-2">
                  <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5" />
                  <div>
                    <p className="font-medium">Training complete</p>
                    <p className="text-xs mt-0.5">
                      Model: <code className="font-mono">{completionInfo.model_name}</code> · Test accuracy: {((completionInfo.accuracy ?? 0) * 100).toFixed(1)}% · F1: {((completionInfo.f1_score ?? 0) * 100).toFixed(1)}% · Time: {completionInfo.total_time_seconds}s
                    </p>
                    <p className="text-xs mt-0.5">Now available in Disease Detection &amp; Evaluation pages.</p>
                  </div>
                </div>
              )}

              {history.length > 1 && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
                  <div>
                    <p className="text-xs font-medium text-muted-foreground mb-1">Loss</p>
                    <ResponsiveContainer width="100%" height={180}>
                      <LineChart data={history}>
                        <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                        <XAxis dataKey="epoch" fontSize={11} />
                        <YAxis fontSize={11} />
                        <Tooltip />
                        <Legend wrapperStyle={{ fontSize: 11 }} />
                        <Line type="monotone" dataKey="train_loss" stroke="#2563eb" strokeWidth={2} dot={false} name="Train" />
                        <Line type="monotone" dataKey="val_loss" stroke="#f59e0b" strokeWidth={2} dot={false} name="Val" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                  <div>
                    <p className="text-xs font-medium text-muted-foreground mb-1">Accuracy</p>
                    <ResponsiveContainer width="100%" height={180}>
                      <LineChart data={history}>
                        <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                        <XAxis dataKey="epoch" fontSize={11} />
                        <YAxis fontSize={11} domain={[0, 1]} />
                        <Tooltip />
                        <Legend wrapperStyle={{ fontSize: 11 }} />
                        <Line type="monotone" dataKey="train_acc" stroke="#10b981" strokeWidth={2} dot={false} name="Train" />
                        <Line type="monotone" dataKey="val_acc" stroke="#8b5cf6" strokeWidth={2} dot={false} name="Val" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Previously trained */}
          {trained.length > 0 && (
            <Card className="border-0 shadow-md">
              <CardHeader>
                <CardTitle>Trained Models</CardTitle>
                <CardDescription>Available for inference on the Disease Detection page</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {trained.map((m) => (
                    <div key={m.model_name} className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
                      <div>
                        <p className="font-medium text-sm">{m.model_name}</p>
                        <p className="text-xs text-muted-foreground">{m.dataset} · {m.architecture}</p>
                      </div>
                      <Badge variant="outline" className="text-xs">{(m.accuracy * 100).toFixed(1)}% accuracy</Badge>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </motion.div>
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="p-3 rounded-lg bg-muted/50">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="text-lg font-bold text-foreground">{value}</p>
    </div>
  );
}
