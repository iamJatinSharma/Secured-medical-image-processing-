"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Shield,
  Lock,
  Key,
  FileCheck,
  Upload,
  Download,
  Eye,
  EyeOff,
  Loader2,
  CheckCircle2,
  AlertCircle,
  Image as ImageIcon,
  Fingerprint,
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { securityApi } from "@/lib/api";
import type { EncryptionMethod, WatermarkMethod, WatermarkExtractResponse, WatermarkVerifyResponse } from "@/lib/types";

type MainTab = "encryption" | "watermarking";
type WatermarkSubTab = "embed" | "extract" | "verify";

interface ImageDimensions {
  width: number;
  height: number;
}

export default function SecurityPage() {
  // Main tab state
  const [mainTab, setMainTab] = useState<MainTab>("encryption");
  
  // Encryption state
  const [encryptionFile, setEncryptionFile] = useState<File | null>(null);
  const [encryptionPreview, setEncryptionPreview] = useState<string | null>(null);
  const [encryptionMethod, setEncryptionMethod] = useState<EncryptionMethod>("arnold");
  const [encryptionPassword, setEncryptionPassword] = useState("");
  const [encryptionIterations, setEncryptionIterations] = useState(20);
  const [showPassword, setShowPassword] = useState(false);
  const [encryptedResult, setEncryptedResult] = useState<Blob | null>(null);
  const [encryptedPreview, setEncryptedPreview] = useState<string | null>(null);
  const [decryptedResult, setDecryptedResult] = useState<Blob | null>(null);
  const [decryptedPreview, setDecryptedPreview] = useState<string | null>(null);
  const [originalDimensions, setOriginalDimensions] = useState<ImageDimensions | null>(null);
  const [isEncrypting, setIsEncrypting] = useState(false);
  const [isDecrypting, setIsDecrypting] = useState(false);
  const [encryptionError, setEncryptionError] = useState<string | null>(null);
  
  // Watermarking state
  const [watermarkSubTab, setWatermarkSubTab] = useState<WatermarkSubTab>("embed");
  const [watermarkFile, setWatermarkFile] = useState<File | null>(null);
  const [watermarkPreview, setWatermarkPreview] = useState<string | null>(null);
  const [watermarkMethod, setWatermarkMethod] = useState<WatermarkMethod>("dct");
  const [watermarkText, setWatermarkText] = useState("");
  const [expectedText, setExpectedText] = useState("");
  const [watermarkedResult, setWatermarkedResult] = useState<Blob | null>(null);
  const [watermarkedPreview, setWatermarkedPreview] = useState<string | null>(null);
  const [extractedResult, setExtractedResult] = useState<WatermarkExtractResponse | null>(null);
  const [verifyResult, setVerifyResult] = useState<WatermarkVerifyResponse | null>(null);
  const [psnrValue, setPsnrValue] = useState<number | null>(null);
  const [isWatermarking, setIsWatermarking] = useState(false);
  const [watermarkError, setWatermarkError] = useState<string | null>(null);
  
  const encryptionFileRef = useRef<HTMLInputElement>(null);
  const watermarkFileRef = useRef<HTMLInputElement>(null);

  // Cleanup object URLs on unmount - using refs to track URLs without dependency issues
  const urlCleanupRef = useRef<string[]>([]);
  
  const registerUrl = useCallback((url: string) => {
    urlCleanupRef.current.push(url);
  }, []);
  
  useEffect(() => {
    return () => {
      urlCleanupRef.current.forEach((url) => URL.revokeObjectURL(url));
    };
  }, []);

  // Handle encryption file selection
  const handleEncryptionFileSelect = useCallback((file: File) => {
    setEncryptionFile(file);
    setEncryptionError(null);
    setEncryptedResult(null);
    setEncryptedPreview(null);
    setDecryptedResult(null);
    setDecryptedPreview(null);
    
    const url = URL.createObjectURL(file);
    registerUrl(url);
    setEncryptionPreview(url);
    
    // Get image dimensions
    const img = new window.Image();
    img.onload = () => {
      setOriginalDimensions({ width: img.width, height: img.height });
    };
    img.src = url;
  }, [registerUrl]);

  // Handle watermark file selection
  const handleWatermarkFileSelect = useCallback((file: File) => {
    setWatermarkFile(file);
    setWatermarkError(null);
    setWatermarkedResult(null);
    setWatermarkedPreview(null);
    setExtractedResult(null);
    setVerifyResult(null);
    setPsnrValue(null);
    
    const url = URL.createObjectURL(file);
    registerUrl(url);
    setWatermarkPreview(url);
  }, [registerUrl]);

  // Drag and drop handlers
  const handleEncryptionDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      handleEncryptionFileSelect(file);
    }
  }, [handleEncryptionFileSelect]);

  const handleWatermarkDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      handleWatermarkFileSelect(file);
    }
  }, [handleWatermarkFileSelect]);

  // Encrypt handler
  const handleEncrypt = async () => {
    if (!encryptionFile) return;
    
    setIsEncrypting(true);
    setEncryptionError(null);
    setEncryptedResult(null);
    setEncryptedPreview(null);
    setDecryptedResult(null);
    setDecryptedPreview(null);
    
    try {
      const result = await securityApi.encrypt(
        encryptionFile,
        encryptionMethod,
        encryptionMethod === "aes" ? encryptionPassword : undefined,
        (encryptionMethod === "arnold" || encryptionMethod === "logistic") ? encryptionIterations : undefined
      );
      
      setEncryptedResult(result);
      
      // For Arnold/Logistic, the result is a viewable image
      if (encryptionMethod !== "aes") {
        const url = URL.createObjectURL(result);
        registerUrl(url);
        setEncryptedPreview(url);
      }
    } catch (err: unknown) {
      const axiosErr = err as { response?: { data?: { detail?: string } } };
      setEncryptionError(axiosErr.response?.data?.detail || "Encryption failed. Is the backend running?");
    } finally {
      setIsEncrypting(false);
    }
  };

  // Decrypt handler
  const handleDecrypt = async () => {
    if (!encryptedResult || !originalDimensions) return;
    
    setIsDecrypting(true);
    setEncryptionError(null);
    setDecryptedResult(null);
    setDecryptedPreview(null);
    
    try {
      const result = await securityApi.decrypt(
        encryptedResult,
        encryptionMethod,
        encryptionMethod === "aes" ? encryptionPassword : undefined,
        (encryptionMethod === "arnold" || encryptionMethod === "logistic") ? encryptionIterations : undefined,
        originalDimensions.width,
        originalDimensions.height
      );
      
      setDecryptedResult(result);
      const url = URL.createObjectURL(result);
      registerUrl(url);
      setDecryptedPreview(url);
    } catch (err: unknown) {
      const axiosErr = err as { response?: { data?: { detail?: string } } };
      setEncryptionError(axiosErr.response?.data?.detail || "Decryption failed. Check your parameters.");
    } finally {
      setIsDecrypting(false);
    }
  };

  // Embed watermark handler
  const handleEmbedWatermark = async () => {
    if (!watermarkFile || !watermarkText) return;
    
    setIsWatermarking(true);
    setWatermarkError(null);
    setWatermarkedResult(null);
    setWatermarkedPreview(null);
    setPsnrValue(null);
    
    try {
      const result = await securityApi.embedWatermark(watermarkFile, watermarkText, watermarkMethod);
      setWatermarkedResult(result);
      
      const url = URL.createObjectURL(result);
      registerUrl(url);
      setWatermarkedPreview(url);
      
      // Calculate PSNR by comparing original and watermarked
      if (watermarkMethod === "dct") {
        // Convert Blob to File for the compare API
        const watermarkedFile = new File([result], watermarkFile.name, { type: result.type || "image/png" });
        const compareResult = await securityApi.compare(watermarkFile, watermarkedFile);
        setPsnrValue(compareResult.psnr_db);
      }
    } catch (err: unknown) {
      const axiosErr = err as { response?: { data?: { detail?: string } } };
      setWatermarkError(axiosErr.response?.data?.detail || "Watermark embedding failed.");
    } finally {
      setIsWatermarking(false);
    }
  };

  // Extract watermark handler
  const handleExtractWatermark = async () => {
    if (!watermarkFile) return;
    
    setIsWatermarking(true);
    setWatermarkError(null);
    setExtractedResult(null);
    
    try {
      const result = await securityApi.extractWatermark(watermarkFile, watermarkMethod);
      setExtractedResult(result);
    } catch (err: unknown) {
      const axiosErr = err as { response?: { data?: { detail?: string } } };
      setWatermarkError(axiosErr.response?.data?.detail || "Watermark extraction failed.");
    } finally {
      setIsWatermarking(false);
    }
  };

  // Verify watermark handler
  const handleVerifyWatermark = async () => {
    if (!watermarkFile || !expectedText) return;
    
    setIsWatermarking(true);
    setWatermarkError(null);
    setVerifyResult(null);
    
    try {
      const result = await securityApi.verifyWatermark(watermarkFile, expectedText, watermarkMethod);
      setVerifyResult(result);
    } catch (err: unknown) {
      const axiosErr = err as { response?: { data?: { detail?: string } } };
      setWatermarkError(axiosErr.response?.data?.detail || "Watermark verification failed.");
    } finally {
      setIsWatermarking(false);
    }
  };

  // Download handler
  const handleDownload = (blob: Blob, filename: string) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center gap-4">
        <div className="p-3 rounded-xl bg-primary/10">
          <Shield className="w-6 h-6 text-primary" />
        </div>
        <div>
          <h1 className="text-2xl font-bold text-foreground">Security</h1>
          <p className="text-muted-foreground">Image encryption and watermarking tools</p>
        </div>
      </div>

      {/* Main Tabs */}
      <div className="flex gap-2 p-1 bg-muted/50 rounded-lg w-fit">
        <button
          onClick={() => setMainTab("encryption")}
          className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-all ${
            mainTab === "encryption"
              ? "bg-card text-foreground shadow-sm"
              : "text-muted-foreground hover:text-foreground"
          }`}
        >
          <Lock className="w-4 h-4" />
          Encryption
        </button>
        <button
          onClick={() => setMainTab("watermarking")}
          className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-all ${
            mainTab === "watermarking"
              ? "bg-card text-foreground shadow-sm"
              : "text-muted-foreground hover:text-foreground"
          }`}
        >
          <Fingerprint className="w-4 h-4" />
          Watermarking
        </button>
      </div>

      {/* Encryption Tab */}
      <AnimatePresence mode="wait">
        {mainTab === "encryption" && (
          <motion.div
            key="encryption"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.2 }}
            className="space-y-6"
          >
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Upload & Settings */}
              <Card className="border-0 shadow-md">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Upload className="w-5 h-5" />
                    Upload Image
                  </CardTitle>
                  <CardDescription>Select an image to encrypt</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {/* File Upload Area */}
                  {!encryptionPreview ? (
                    <div
                      onDrop={handleEncryptionDrop}
                      onDragOver={(e) => e.preventDefault()}
                      onClick={() => encryptionFileRef.current?.click()}
                      className="border-2 border-dashed border-muted-foreground/25 rounded-xl p-8 text-center hover:border-primary/50 transition-colors cursor-pointer"
                    >
                      <ImageIcon className="w-10 h-10 mx-auto text-muted-foreground/50 mb-3" />
                      <p className="text-muted-foreground mb-1">Drop your image here or click to browse</p>
                      <p className="text-sm text-muted-foreground/70">Supports PNG, JPG, BMP</p>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      <div className="relative rounded-xl overflow-hidden bg-black/5 flex items-center justify-center min-h-[200px]">
                        <img
                          src={encryptionPreview}
                          alt="Original image"
                          className="max-h-[250px] object-contain"
                        />
                      </div>
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <ImageIcon className="w-4 h-4 text-muted-foreground" />
                          <span className="text-sm text-muted-foreground truncate max-w-[150px]">
                            {encryptionFile?.name}
                          </span>
                          {originalDimensions && (
                            <Badge variant="secondary" className="text-xs">
                              {originalDimensions.width}x{originalDimensions.height}
                            </Badge>
                          )}
                        </div>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => {
                            setEncryptionFile(null);
                            setEncryptionPreview(null);
                            setEncryptedResult(null);
                            setEncryptedPreview(null);
                            setDecryptedResult(null);
                            setDecryptedPreview(null);
                          }}
                        >
                          Change
                        </Button>
                      </div>
                    </div>
                  )}
                  
                  <input
                    ref={encryptionFileRef}
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) handleEncryptionFileSelect(file);
                    }}
                  />

                  {/* Method Selection */}
                  <div className="space-y-3">
                    <Label>Encryption Method</Label>
                    <div className="grid grid-cols-3 gap-2">
                      {[
                        { id: "arnold", label: "Arnold Cat Map", icon: Lock },
                        { id: "logistic", label: "Logistic Map", icon: Lock },
                        { id: "aes", label: "AES-256-GCM", icon: Key },
                      ].map((method) => {
                        const Icon = method.icon;
                        return (
                          <button
                            key={method.id}
                            onClick={() => setEncryptionMethod(method.id as EncryptionMethod)}
                            className={`p-3 rounded-lg border-2 transition-all text-center ${
                              encryptionMethod === method.id
                                ? "border-primary bg-primary/5"
                                : "border-transparent bg-muted/50 hover:bg-muted"
                            }`}
                          >
                            <Icon className={`w-5 h-5 mx-auto mb-1 ${encryptionMethod === method.id ? "text-primary" : "text-muted-foreground"}`} />
                            <span className={`text-xs font-medium ${encryptionMethod === method.id ? "text-primary" : "text-muted-foreground"}`}>
                              {method.label}
                            </span>
                          </button>
                        );
                      })}
                    </div>
                  </div>

                  {/* Method-specific settings */}
                  {encryptionMethod === "aes" && (
                    <div className="space-y-2">
                      <Label htmlFor="password">Password</Label>
                      <div className="relative">
                        <Input
                          id="password"
                          type={showPassword ? "text" : "password"}
                          value={encryptionPassword}
                          onChange={(e) => setEncryptionPassword(e.target.value)}
                          placeholder="Enter encryption password"
                          className="pr-10"
                        />
                        <button
                          type="button"
                          onClick={() => setShowPassword(!showPassword)}
                          className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                        >
                          {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                        </button>
                      </div>
                    </div>
                  )}

                  {(encryptionMethod === "arnold" || encryptionMethod === "logistic") && (
                    <div className="space-y-2">
                      <Label htmlFor="iterations">Iterations: {encryptionIterations}</Label>
                      <input
                        id="iterations"
                        type="range"
                        min="1"
                        max="50"
                        value={encryptionIterations}
                        onChange={(e) => setEncryptionIterations(Number(e.target.value))}
                        className="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer accent-primary"
                      />
                      <div className="flex justify-between text-xs text-muted-foreground">
                        <span>1</span>
                        <span>50</span>
                      </div>
                    </div>
                  )}

                  {/* Encrypt Button */}
                  {encryptionFile && (
                    <Button
                      onClick={handleEncrypt}
                      disabled={isEncrypting || (encryptionMethod === "aes" && !encryptionPassword)}
                      className="w-full bg-gradient-to-r from-violet-600 to-violet-500 hover:from-violet-700 hover:to-violet-600 shadow-lg shadow-violet-500/25"
                    >
                      {isEncrypting ? (
                        <>
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          Encrypting...
                        </>
                      ) : (
                        <>
                          <Lock className="mr-2 h-4 w-4" />
                          Encrypt Image
                        </>
                      )}
                    </Button>
                  )}

                  {/* Error */}
                  {encryptionError && (
                    <div className="p-3 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-600 dark:text-red-400 text-sm flex items-center gap-2">
                      <AlertCircle className="w-4 h-4 shrink-0" />
                      {encryptionError}
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Results */}
              <Card className="border-0 shadow-md">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <FileCheck className="w-5 h-5" />
                    Results
                  </CardTitle>
                  <CardDescription>Encrypted and decrypted outputs</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {!encryptedResult ? (
                    <div className="flex flex-col items-center justify-center py-12 text-center">
                      <Lock className="w-12 h-12 text-muted-foreground/30 mb-3" />
                      <p className="text-muted-foreground">Encrypt an image to see results</p>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      {/* Encrypted Result */}
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-sm font-medium text-foreground">Encrypted</span>
                          <div className="flex items-center gap-2">
                            {encryptionMethod === "aes" ? (
                              <Badge variant="secondary" className="bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400">
                                Binary Data
                              </Badge>
                            ) : (
                              <Badge variant="secondary" className="bg-violet-100 text-violet-700 dark:bg-violet-900/30 dark:text-violet-400">
                                Scrambled Image
                              </Badge>
                            )}
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => handleDownload(encryptedResult, `encrypted_${encryptionFile?.name || "image"}`)}
                            >
                              <Download className="w-3 h-3 mr-1" />
                              Download
                            </Button>
                          </div>
                        </div>
                        
                        {encryptionMethod !== "aes" && encryptedPreview ? (
                          <div className="relative rounded-xl overflow-hidden bg-black/5 flex items-center justify-center min-h-[150px]">
                            <img
                              src={encryptedPreview}
                              alt="Encrypted image"
                              className="max-h-[200px] object-contain"
                            />
                          </div>
                        ) : (
                          <div className="rounded-xl bg-muted/50 p-6 text-center">
                            <Key className="w-8 h-8 mx-auto text-muted-foreground/50 mb-2" />
                            <p className="text-sm text-muted-foreground">
                              AES encryption produces binary data, not a viewable image
                            </p>
                          </div>
                        )}
                      </div>

                      {/* Decrypt Button */}
                      <Button
                        onClick={handleDecrypt}
                        disabled={isDecrypting}
                        variant="outline"
                        className="w-full"
                      >
                        {isDecrypting ? (
                          <>
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                            Decrypting...
                          </>
                        ) : (
                          <>
                            <Eye className="mr-2 h-4 w-4" />
                            Decrypt to Verify
                          </>
                        )}
                      </Button>

                      {/* Decrypted Result */}
                      {decryptedResult && decryptedPreview && (
                        <div className="space-y-2">
                          <div className="flex items-center justify-between">
                            <span className="text-sm font-medium text-foreground">Decrypted</span>
                            <Badge className="bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400">
                              <CheckCircle2 className="w-3 h-3 mr-1" />
                              Verified
                            </Badge>
                          </div>
                          <div className="relative rounded-xl overflow-hidden bg-black/5 flex items-center justify-center min-h-[150px]">
                            <img
                              src={decryptedPreview}
                              alt="Decrypted image"
                              className="max-h-[200px] object-contain"
                            />
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Watermarking Tab */}
      <AnimatePresence mode="wait">
        {mainTab === "watermarking" && (
          <motion.div
            key="watermarking"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.2 }}
            className="space-y-6"
          >
            {/* Sub-tabs */}
            <div className="flex gap-2">
              {[
                { id: "embed", label: "Embed", icon: FileCheck },
                { id: "extract", label: "Extract", icon: Eye },
                { id: "verify", label: "Verify", icon: CheckCircle2 },
              ].map((tab) => {
                const Icon = tab.icon;
                return (
                  <button
                    key={tab.id}
                    onClick={() => {
                      setWatermarkSubTab(tab.id as WatermarkSubTab);
                      setWatermarkedResult(null);
                      setWatermarkedPreview(null);
                      setExtractedResult(null);
                      setVerifyResult(null);
                      setPsnrValue(null);
                      setWatermarkError(null);
                    }}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                      watermarkSubTab === tab.id
                        ? "bg-primary text-primary-foreground"
                        : "bg-muted/50 text-muted-foreground hover:bg-muted hover:text-foreground"
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                    {tab.label}
                  </button>
                );
              })}
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Upload & Settings */}
              <Card className="border-0 shadow-md">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Upload className="w-5 h-5" />
                    {watermarkSubTab === "embed" && "Embed Watermark"}
                    {watermarkSubTab === "extract" && "Extract Watermark"}
                    {watermarkSubTab === "verify" && "Verify Watermark"}
                  </CardTitle>
                  <CardDescription>
                    {watermarkSubTab === "embed" && "Add a hidden or visible watermark to your image"}
                    {watermarkSubTab === "extract" && "Extract embedded watermark text from an image"}
                    {watermarkSubTab === "verify" && "Verify if an image contains expected watermark text"}
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {/* File Upload Area */}
                  {!watermarkPreview ? (
                    <div
                      onDrop={handleWatermarkDrop}
                      onDragOver={(e) => e.preventDefault()}
                      onClick={() => watermarkFileRef.current?.click()}
                      className="border-2 border-dashed border-muted-foreground/25 rounded-xl p-8 text-center hover:border-primary/50 transition-colors cursor-pointer"
                    >
                      <ImageIcon className="w-10 h-10 mx-auto text-muted-foreground/50 mb-3" />
                      <p className="text-muted-foreground mb-1">Drop your image here or click to browse</p>
                      <p className="text-sm text-muted-foreground/70">Supports PNG, JPG, BMP</p>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      <div className="relative rounded-xl overflow-hidden bg-black/5 flex items-center justify-center min-h-[200px]">
                        <img
                          src={watermarkPreview}
                          alt="Watermark image"
                          className="max-h-[250px] object-contain"
                        />
                      </div>
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <ImageIcon className="w-4 h-4 text-muted-foreground" />
                          <span className="text-sm text-muted-foreground truncate max-w-[200px]">
                            {watermarkFile?.name}
                          </span>
                        </div>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => {
                            setWatermarkFile(null);
                            setWatermarkPreview(null);
                            setWatermarkedResult(null);
                            setWatermarkedPreview(null);
                            setExtractedResult(null);
                            setVerifyResult(null);
                          }}
                        >
                          Change
                        </Button>
                      </div>
                    </div>
                  )}

                  <input
                    ref={watermarkFileRef}
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) handleWatermarkFileSelect(file);
                    }}
                  />

                  {/* Method Selection */}
                  <div className="space-y-2">
                    <Label>Watermark Method</Label>
                    <div className="grid grid-cols-2 gap-2">
                      <button
                        onClick={() => setWatermarkMethod("dct")}
                        className={`p-3 rounded-lg border-2 transition-all text-center ${
                          watermarkMethod === "dct"
                            ? "border-primary bg-primary/5"
                            : "border-transparent bg-muted/50 hover:bg-muted"
                        }`}
                      >
                        <EyeOff className={`w-5 h-5 mx-auto mb-1 ${watermarkMethod === "dct" ? "text-primary" : "text-muted-foreground"}`} />
                        <span className={`text-xs font-medium ${watermarkMethod === "dct" ? "text-primary" : "text-muted-foreground"}`}>
                          DCT Invisible
                        </span>
                      </button>
                      <button
                        onClick={() => setWatermarkMethod("visible")}
                        className={`p-3 rounded-lg border-2 transition-all text-center ${
                          watermarkMethod === "visible"
                            ? "border-primary bg-primary/5"
                            : "border-transparent bg-muted/50 hover:bg-muted"
                        }`}
                      >
                        <Eye className={`w-5 h-5 mx-auto mb-1 ${watermarkMethod === "visible" ? "text-primary" : "text-muted-foreground"}`} />
                        <span className={`text-xs font-medium ${watermarkMethod === "visible" ? "text-primary" : "text-muted-foreground"}`}>
                          Visible
                        </span>
                      </button>
                    </div>
                  </div>

                  {/* Embed: Text Input */}
                  {watermarkSubTab === "embed" && (
                    <div className="space-y-2">
                      <Label htmlFor="watermarkText">Watermark Text</Label>
                      <Input
                        id="watermarkText"
                        value={watermarkText}
                        onChange={(e) => setWatermarkText(e.target.value)}
                        placeholder="Enter text to embed"
                      />
                    </div>
                  )}

                  {/* Verify: Expected Text Input */}
                  {watermarkSubTab === "verify" && (
                    <div className="space-y-2">
                      <Label htmlFor="expectedText">Expected Watermark Text</Label>
                      <Input
                        id="expectedText"
                        value={expectedText}
                        onChange={(e) => setExpectedText(e.target.value)}
                        placeholder="Enter expected watermark text"
                      />
                    </div>
                  )}

                  {/* Action Buttons */}
                  {watermarkFile && (
                    <>
                      {watermarkSubTab === "embed" && (
                        <Button
                          onClick={handleEmbedWatermark}
                          disabled={isWatermarking || !watermarkText}
                          className="w-full bg-gradient-to-r from-cyan-600 to-cyan-500 hover:from-cyan-700 hover:to-cyan-600 shadow-lg shadow-cyan-500/25"
                        >
                          {isWatermarking ? (
                            <>
                              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                              Embedding...
                            </>
                          ) : (
                            <>
                              <FileCheck className="mr-2 h-4 w-4" />
                              Embed Watermark
                            </>
                          )}
                        </Button>
                      )}

                      {watermarkSubTab === "extract" && (
                        <Button
                          onClick={handleExtractWatermark}
                          disabled={isWatermarking}
                          className="w-full bg-gradient-to-r from-cyan-600 to-cyan-500 hover:from-cyan-700 hover:to-cyan-600 shadow-lg shadow-cyan-500/25"
                        >
                          {isWatermarking ? (
                            <>
                              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                              Extracting...
                            </>
                          ) : (
                            <>
                              <Eye className="mr-2 h-4 w-4" />
                              Extract Watermark
                            </>
                          )}
                        </Button>
                      )}

                      {watermarkSubTab === "verify" && (
                        <Button
                          onClick={handleVerifyWatermark}
                          disabled={isWatermarking || !expectedText}
                          className="w-full bg-gradient-to-r from-cyan-600 to-cyan-500 hover:from-cyan-700 hover:to-cyan-600 shadow-lg shadow-cyan-500/25"
                        >
                          {isWatermarking ? (
                            <>
                              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                              Verifying...
                            </>
                          ) : (
                            <>
                              <CheckCircle2 className="mr-2 h-4 w-4" />
                              Verify Watermark
                            </>
                          )}
                        </Button>
                      )}
                    </>
                  )}

                  {/* Error */}
                  {watermarkError && (
                    <div className="p-3 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-600 dark:text-red-400 text-sm flex items-center gap-2">
                      <AlertCircle className="w-4 h-4 shrink-0" />
                      {watermarkError}
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Results */}
              <Card className="border-0 shadow-md">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <FileCheck className="w-5 h-5" />
                    Results
                  </CardTitle>
                  <CardDescription>Watermarking output</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {/* Embed Results */}
                  {watermarkSubTab === "embed" && !watermarkedResult && (
                    <div className="flex flex-col items-center justify-center py-12 text-center">
                      <Fingerprint className="w-12 h-12 text-muted-foreground/30 mb-3" />
                      <p className="text-muted-foreground">Embed a watermark to see results</p>
                    </div>
                  )}

                  {watermarkSubTab === "embed" && watermarkedResult && watermarkedPreview && (
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium text-foreground">Watermarked Image</span>
                        <div className="flex items-center gap-2">
                          {psnrValue !== null && (
                            <Badge variant="secondary" className="bg-cyan-100 text-cyan-700 dark:bg-cyan-900/30 dark:text-cyan-400">
                              PSNR: {psnrValue.toFixed(2)} dB
                            </Badge>
                          )}
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => handleDownload(watermarkedResult, `watermarked_${watermarkFile?.name || "image"}`)}
                          >
                            <Download className="w-3 h-3 mr-1" />
                            Download
                          </Button>
                        </div>
                      </div>
                      <div className="relative rounded-xl overflow-hidden bg-black/5 flex items-center justify-center min-h-[200px]">
                        <img
                          src={watermarkedPreview}
                          alt="Watermarked image"
                          className="max-h-[250px] object-contain"
                        />
                      </div>
                      <div className="grid grid-cols-2 gap-4">
                        <div className="p-3 rounded-lg bg-muted/50">
                          <p className="text-xs text-muted-foreground mb-1">Original</p>
                          <img
                            src={watermarkPreview || ""}
                            alt="Original"
                            className="w-full h-24 object-contain rounded"
                          />
                        </div>
                        <div className="p-3 rounded-lg bg-muted/50">
                          <p className="text-xs text-muted-foreground mb-1">Watermarked</p>
                          <img
                            src={watermarkedPreview}
                            alt="Watermarked"
                            className="w-full h-24 object-contain rounded"
                          />
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Extract Results */}
                  {watermarkSubTab === "extract" && !extractedResult && (
                    <div className="flex flex-col items-center justify-center py-12 text-center">
                      <Eye className="w-12 h-12 text-muted-foreground/30 mb-3" />
                      <p className="text-muted-foreground">Extract a watermark to see results</p>
                    </div>
                  )}

                  {watermarkSubTab === "extract" && extractedResult && (
                    <div className="space-y-4">
                      <div className="p-6 rounded-xl bg-gradient-to-br from-cyan-50 to-teal-50 dark:from-cyan-900/20 dark:to-teal-900/20 border border-cyan-200 dark:border-cyan-800">
                        <p className="text-sm text-muted-foreground mb-2">Extracted Text</p>
                        <p className="text-xl font-bold text-foreground break-all">{extractedResult.extracted_text || "(empty)"}</p>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge variant="secondary">Method: {extractedResult.method}</Badge>
                      </div>
                    </div>
                  )}

                  {/* Verify Results */}
                  {watermarkSubTab === "verify" && !verifyResult && (
                    <div className="flex flex-col items-center justify-center py-12 text-center">
                      <CheckCircle2 className="w-12 h-12 text-muted-foreground/30 mb-3" />
                      <p className="text-muted-foreground">Verify a watermark to see results</p>
                    </div>
                  )}

                  {watermarkSubTab === "verify" && verifyResult && (
                    <div className="space-y-4">
                      <div className={`p-6 rounded-xl border ${
                        verifyResult.verified
                          ? "bg-gradient-to-br from-emerald-50 to-green-50 dark:from-emerald-900/20 dark:to-green-900/20 border-emerald-200 dark:border-emerald-800"
                          : "bg-gradient-to-br from-red-50 to-rose-50 dark:from-red-900/20 dark:to-rose-900/20 border-red-200 dark:border-red-800"
                      }`}>
                        <div className="flex items-center gap-3 mb-3">
                          {verifyResult.verified ? (
                            <CheckCircle2 className="w-8 h-8 text-emerald-600" />
                          ) : (
                            <AlertCircle className="w-8 h-8 text-red-600" />
                          )}
                          <p className={`text-xl font-bold ${verifyResult.verified ? "text-emerald-700 dark:text-emerald-400" : "text-red-700 dark:text-red-400"}`}>
                            {verifyResult.verified ? "Verified" : "Not Verified"}
                          </p>
                        </div>
                        <p className="text-sm text-muted-foreground">
                          Expected: <span className="font-medium text-foreground">{verifyResult.expected_text}</span>
                        </p>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
