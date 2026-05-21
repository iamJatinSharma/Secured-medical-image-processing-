// --- User Types ---

export interface User {
  id: number;
  username: string;
  email: string;
  role: "admin" | "doctor" | "researcher" | "viewer";
  created_at: string;
  is_active: boolean;
}

export interface LoginRequest {
  username: string;
  password: string;
}

export interface RegisterRequest {
  username: string;
  email: string;
  password: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
}

// --- Image Types ---

export type ImageType = "breast" | "lung" | "brain" | "skin";

export interface MedicalImage {
  id: number;
  user_id: number;
  filename: string;
  original_path: string;
  processed_path: string | null;
  upload_time: string;
  image_type: ImageType;
  width: number | null;
  height: number | null;
  channels: number | null;
  file_size: number | null;
}

// --- Prediction Types ---

export interface Prediction {
  id: number;
  image_id: number;
  model_name: string;
  prediction_label: string;
  confidence: number;
  gradcam_path: string | null;
  lime_path: string | null;
  shap_path: string | null;
  created_at: string;
  processing_time_ms: number | null;
}

// --- Report Types ---

export interface Report {
  id: number;
  prediction_id: number;
  pdf_path: string;
  created_at: string;
}

// --- API Response Types ---

export interface ApiError {
  detail: string;
}

export interface HealthResponse {
  status: string;
  module: string;
}

// --- Dashboard Types ---

export interface DashboardStats {
  total_images: number;
  total_predictions: number;
  total_reports: number;
  recent_predictions: Prediction[];
}

// --- Security Types ---

export type EncryptionMethod = "arnold" | "logistic" | "aes";
export type WatermarkMethod = "dct" | "visible";

export interface EncryptionResult {
  encrypted_data: Blob;
  method: EncryptionMethod;
  original_width: number;
  original_height: number;
}

export interface WatermarkVerifyResponse {
  verified: boolean;
  expected_text: string;
}

export interface WatermarkExtractResponse {
  extracted_text: string;
  method: string;
}

export interface CompareResponse {
  psnr_db: number;
  mse: number;
  identical: boolean;
  ssim?: number;
}

// --- Processing Types ---

export type DenoiseMethod = "nlm" | "bilateral" | "gaussian";
export type EnhanceMethod = "clahe" | "detail" | "histogram";
export type SegmentMethod = "threshold" | "watershed" | "kmeans";

// --- Metrics Types ---

export interface ModelMetrics {
  model_name: string;
  accuracy: number;
  f1_score: number;
  auc?: number;
  confusion_matrix: number[][];
  labels: string[];
  classification_report?: Record<string, { precision: number; recall: number; "f1-score": number; support: number }>;
  roc_data?: { fpr: number[]; tpr: number[]; auc: number };
  training_history?: {
    epochs: number[];
    train_loss: number[];
    val_loss: number[];
    train_acc: number[];
    val_acc: number[];
  };
}

export interface ModelPerformanceResponse {
  accuracy: number;
  f1_score: number;
  confusion_matrix: number[][];
  classification_report: Record<string, { precision: number; recall: number; "f1-score": number; support: number }>;
  auc?: number;
  roc_data?: { fpr: number[]; tpr: number[]; auc: number };
}

// --- Training Types ---

export interface TrainingConfig {
  model: string;
  dataset?: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  image_size?: number;
  max_images?: number | null;
}

export interface MedMnistDataset {
  id: string;
  name: string;
  modality: string;
  task: string;
  classes: string[];
  size_train: number;
  size_val: number;
  size_test: number;
}

export interface TrainedModelInfo {
  model_name: string;
  dataset: string;
  architecture: string;
  accuracy: number;
  labels: string[];
}

export interface TrainingProgress {
  epoch: number;
  total_epochs: number;
  train_loss: number;
  val_loss: number;
  train_acc: number;
  val_acc: number;
  percent_complete: number;
  eta_seconds: number;
}

export interface TrainingStatus {
  status: "idle" | "training" | "complete";
  progress: TrainingProgress | { status?: string; best_val_acc?: number; total_time_seconds?: number } | Record<string, never>;
}

// --- Reports Types ---

export interface ReportSummary {
  id: number;
  prediction_id: number;
  pdf_path: string;
  created_at: string;
  prediction_label?: string;
  model_name?: string;
}

// --- Users (admin) Types ---

export interface UserListResponse {
  users: User[];
  total: number;
}

export interface UpdateUserRoleRequest {
  role: "admin" | "doctor" | "researcher" | "viewer";
}

// --- XAI Types ---

export type XaiMethod = "gradcam" | "lime" | "shap";

export interface XaiMethodInfo {
  id: string;
  name: string;
  description: string;
  supported_models: string[];
}

export interface XaiMethodsResponse {
  methods: XaiMethodInfo[];
}
