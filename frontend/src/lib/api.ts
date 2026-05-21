import axios, { AxiosError, InternalAxiosRequestConfig } from "axios";
import { useAuthStore } from "@/store/authStore";
import type {
  LoginRequest,
  RegisterRequest,
  TokenResponse,
  User,
  MedicalImage,
  Prediction,
  HealthResponse,
  EncryptionMethod,
  WatermarkMethod,
  WatermarkExtractResponse,
  WatermarkVerifyResponse,
  CompareResponse,
  XaiMethodsResponse,
  DenoiseMethod,
  EnhanceMethod,
  SegmentMethod,
  ModelMetrics,
  ModelPerformanceResponse,
  TrainingConfig,
  TrainingStatus,
  MedMnistDataset,
  TrainedModelInfo,
  ReportSummary,
  UserListResponse,
  UpdateUserRoleRequest,
} from "./types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

// Request interceptor to add JWT token
api.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    const token = useAuthStore.getState().token;
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error: AxiosError) => {
    return Promise.reject(error);
  }
);

// Response interceptor to handle 401 errors
api.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    if (error.response?.status === 401) {
      useAuthStore.getState().logout();
      if (typeof window !== "undefined") {
        window.location.href = "/auth/login";
      }
    }
    return Promise.reject(error);
  }
);

// --- Auth API ---

export const authApi = {
  login: async (data: LoginRequest): Promise<TokenResponse> => {
    const res = await api.post<TokenResponse>("/auth/login", data);
    return res.data;
  },

  register: async (data: RegisterRequest): Promise<User> => {
    const res = await api.post<User>("/auth/register", data);
    return res.data;
  },

  getMe: async (): Promise<User> => {
    const res = await api.get<User>("/auth/me");
    return res.data;
  },
};

// --- Images API ---

export const imagesApi = {
  upload: async (file: File, imageType: string): Promise<MedicalImage> => {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("image_type", imageType);
    const res = await api.post<MedicalImage>("/images/upload", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return res.data;
  },

  getAll: async (): Promise<MedicalImage[]> => {
    const res = await api.get<MedicalImage[]>("/images");
    return res.data;
  },
};

// --- Predict API ---

export const predictApi = {
  detect: async (imageId: number, modelName: string): Promise<Prediction> => {
    const res = await api.post<Prediction>("/predict/detect", {
      image_id: imageId,
      model_name: modelName,
    });
    return res.data;
  },

  list: async (): Promise<Prediction[]> => {
    const res = await api.get<Prediction[]>("/predict/predictions");
    return res.data;
  },
};

// --- Health API ---

export const healthApi = {
  check: async (): Promise<HealthResponse> => {
    const res = await api.get<HealthResponse>("/health");
    return res.data;
  },
};

// --- Security API ---

export const securityApi = {
  encrypt: async (file: File, method: EncryptionMethod, password?: string, iterations?: number): Promise<Blob> => {
    const formData = new FormData();
    formData.append("image", file);
    formData.append("method", method);
    if (password) formData.append("password", password);
    if (iterations) formData.append("iterations", String(iterations));
    const res = await api.post("/security/encrypt", formData, {
      headers: { "Content-Type": "multipart/form-data" },
      responseType: "blob",
    });
    return res.data;
  },

  decrypt: async (
    file: File | Blob,
    method: EncryptionMethod,
    password?: string,
    iterations?: number,
    width?: number,
    height?: number,
  ): Promise<Blob> => {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("method", method);
    if (password) formData.append("password", password);
    if (iterations) formData.append("iterations", String(iterations));
    if (width) formData.append("width", String(width));
    if (height) formData.append("height", String(height));
    const res = await api.post("/security/decrypt", formData, {
      headers: { "Content-Type": "multipart/form-data" },
      responseType: "blob",
    });
    return res.data;
  },

  embedWatermark: async (file: File, text: string, method: WatermarkMethod = "dct"): Promise<Blob> => {
    const formData = new FormData();
    formData.append("image", file);
    formData.append("text", text);
    formData.append("method", method);
    const res = await api.post("/security/watermark/embed", formData, {
      headers: { "Content-Type": "multipart/form-data" },
      responseType: "blob",
    });
    return res.data;
  },

  extractWatermark: async (file: File, method: WatermarkMethod = "dct"): Promise<WatermarkExtractResponse> => {
    const formData = new FormData();
    formData.append("image", file);
    formData.append("method", method);
    const res = await api.post<WatermarkExtractResponse>("/security/watermark/extract", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return res.data;
  },

  verifyWatermark: async (file: File, expectedText: string, method: WatermarkMethod = "dct"): Promise<WatermarkVerifyResponse> => {
    const formData = new FormData();
    formData.append("image", file);
    formData.append("expected_text", expectedText);
    formData.append("method", method);
    const res = await api.post<WatermarkVerifyResponse>("/security/watermark/verify", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return res.data;
  },

  compare: async (original: File, processed: File): Promise<CompareResponse> => {
    const formData = new FormData();
    formData.append("original", original);
    formData.append("processed", processed);
    const res = await api.post<CompareResponse>("/security/compare", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return res.data;
  },
};

// --- XAI API ---

export const xaiApi = {
  gradcam: async (file: File, modelName: string = "resnet50_breast", targetClass?: number): Promise<Blob> => {
    const formData = new FormData();
    formData.append("image", file);
    formData.append("model_name", modelName);
    if (targetClass !== undefined) formData.append("target_class", String(targetClass));
    const res = await api.post("/xai/gradcam", formData, {
      headers: { "Content-Type": "multipart/form-data" },
      responseType: "blob",
    });
    return res.data;
  },

  lime: async (file: File, modelName: string = "resnet50_breast", numSamples: number = 100): Promise<Blob> => {
    const formData = new FormData();
    formData.append("image", file);
    formData.append("model_name", modelName);
    formData.append("num_samples", String(numSamples));
    const res = await api.post("/xai/lime", formData, {
      headers: { "Content-Type": "multipart/form-data" },
      responseType: "blob",
    });
    return res.data;
  },

  shap: async (file: File, modelName: string = "resnet50_breast"): Promise<Blob> => {
    const formData = new FormData();
    formData.append("image", file);
    formData.append("model_name", modelName);
    const res = await api.post("/xai/shap", formData, {
      headers: { "Content-Type": "multipart/form-data" },
      responseType: "blob",
    });
    return res.data;
  },

  getMethods: async (): Promise<XaiMethodsResponse> => {
    const res = await api.get<XaiMethodsResponse>("/xai/methods");
    return res.data;
  },
};

// --- Processing API ---

export const processingApi = {
  denoise: async (file: File, method: DenoiseMethod = "nlm"): Promise<Blob> => {
    const formData = new FormData();
    formData.append("image", file);
    formData.append("method", method);
    const res = await api.post("/processing/denoise", formData, {
      headers: { "Content-Type": "multipart/form-data" },
      responseType: "blob",
    });
    return res.data;
  },

  enhance: async (file: File, method: EnhanceMethod = "clahe"): Promise<Blob> => {
    const formData = new FormData();
    formData.append("image", file);
    formData.append("method", method);
    const res = await api.post("/processing/enhance", formData, {
      headers: { "Content-Type": "multipart/form-data" },
      responseType: "blob",
    });
    return res.data;
  },

  segment: async (file: File, method: SegmentMethod = "threshold"): Promise<Blob> => {
    const formData = new FormData();
    formData.append("image", file);
    formData.append("method", method);
    const res = await api.post("/processing/segment", formData, {
      headers: { "Content-Type": "multipart/form-data" },
      responseType: "blob",
    });
    return res.data;
  },

  compare: async (image1: File, image2: File): Promise<CompareResponse> => {
    const formData = new FormData();
    formData.append("image1", image1);
    formData.append("image2", image2);
    const res = await api.post<CompareResponse>("/processing/compare", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return res.data;
  },
};

// --- Metrics API ---

export const metricsApi = {
  compare: async (image1: File, image2: File): Promise<CompareResponse> => {
    const formData = new FormData();
    formData.append("image1", image1);
    formData.append("image2", image2);
    const res = await api.post<CompareResponse>("/metrics/compare", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return res.data;
  },

  getModelMetrics: async (modelName: string): Promise<ModelMetrics> => {
    const res = await api.get<ModelMetrics>(`/metrics/model/${modelName}`);
    return res.data;
  },

  modelPerformance: async (
    yTrue: number[],
    yPred: number[],
    yProbs?: number[],
    labels?: string[],
  ): Promise<ModelPerformanceResponse> => {
    const formData = new FormData();
    formData.append("y_true", JSON.stringify(yTrue));
    formData.append("y_pred", JSON.stringify(yPred));
    if (yProbs) formData.append("y_probs", JSON.stringify(yProbs));
    if (labels) formData.append("labels", JSON.stringify(labels));
    const res = await api.post<ModelPerformanceResponse>("/metrics/model-performance", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return res.data;
  },
};

// --- Training API ---

export const trainingApi = {
  start: async (config: TrainingConfig): Promise<{ message?: string; error?: string; config?: TrainingConfig }> => {
    const res = await api.post("/training/start", config);
    return res.data;
  },

  getStatus: async (): Promise<TrainingStatus> => {
    const res = await api.get<TrainingStatus>("/training/status");
    return res.data;
  },

  listDatasets: async (): Promise<{ datasets: MedMnistDataset[] }> => {
    const res = await api.get<{ datasets: MedMnistDataset[] }>("/training/datasets");
    return res.data;
  },

  listTrained: async (): Promise<{ models: TrainedModelInfo[] }> => {
    const res = await api.get<{ models: TrainedModelInfo[] }>("/training/trained");
    return res.data;
  },

  connectWebSocket: (onMessage: (data: unknown) => void): WebSocket => {
    const wsBase = API_BASE_URL.replace(/^http/, "ws").replace(/\/api$/, "/api");
    const ws = new WebSocket(`${wsBase}/training/ws`);
    ws.onmessage = (event) => {
      try {
        onMessage(JSON.parse(event.data));
      } catch {
        // ignore non-JSON
      }
    };
    // Keep-alive: server expects receive_text to detect disconnect
    const interval = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send("ping");
      } else {
        clearInterval(interval);
      }
    }, 15000);
    return ws;
  },
};

// --- Reports API ---

export const reportsApi = {
  list: async (): Promise<ReportSummary[]> => {
    const res = await api.get<ReportSummary[]>("/reports");
    return res.data;
  },

  generate: async (predictionId: number): Promise<ReportSummary> => {
    const res = await api.post<ReportSummary>("/reports/generate", { prediction_id: predictionId });
    return res.data;
  },

  download: async (reportId: number): Promise<Blob> => {
    const res = await api.get(`/reports/${reportId}/download`, { responseType: "blob" });
    return res.data;
  },

  delete: async (reportId: number): Promise<void> => {
    await api.delete(`/reports/${reportId}`);
  },
};

// --- Users (admin) API ---

export const usersApi = {
  list: async (): Promise<UserListResponse> => {
    const res = await api.get<UserListResponse>("/auth/users");
    return res.data;
  },

  updateRole: async (userId: number, role: UpdateUserRoleRequest["role"]): Promise<User> => {
    const res = await api.patch<User>(`/auth/users/${userId}/role`, { role });
    return res.data;
  },

  setActive: async (userId: number, isActive: boolean): Promise<User> => {
    const res = await api.patch<User>(`/auth/users/${userId}/active`, { is_active: isActive });
    return res.data;
  },

  delete: async (userId: number): Promise<void> => {
    await api.delete(`/auth/users/${userId}`);
  },
};

export default api;
