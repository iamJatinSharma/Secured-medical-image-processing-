"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuthStore } from "@/store/authStore";
import { Loader2 } from "lucide-react";

export default function RootPage() {
  const router = useRouter();
  const { token, user } = useAuthStore();

  useEffect(() => {
    if (token && user) {
      router.replace("/dashboard");
    } else {
      router.replace("/auth/login");
    }
  }, [token, user, router]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-background">
      <Loader2 className="w-8 h-8 animate-spin text-teal-500" />
    </div>
  );
}
