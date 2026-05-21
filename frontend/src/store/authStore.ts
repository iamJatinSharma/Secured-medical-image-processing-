import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { User } from "@/lib/types";

interface AuthState {
  token: string | null;
  user: User | null;
  isLoading: boolean;
  error: string | null;
  setAuth: (token: string, user: User) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  logout: () => void;
  isAuthenticated: () => boolean;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      token: null,
      user: null,
      isLoading: false,
      error: null,
      setAuth: (token, user) => set({ token, user, error: null }),
      setLoading: (isLoading) => set({ isLoading }),
      setError: (error) => set({ error, isLoading: false }),
      logout: () => set({ token: null, user: null, error: null }),
      isAuthenticated: () => !!get().token && !!get().user,
    }),
    {
      name: "auth-storage",
      partialize: (state) => ({ token: state.token, user: state.user }),
    }
  )
);
