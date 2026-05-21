import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

type PydanticError = { msg?: string; loc?: (string | number)[] };

export function extractApiError(err: unknown, fallback: string): string {
  const detail = (err as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return (detail as PydanticError[])
      .map((e) => {
        const field = Array.isArray(e.loc) ? e.loc.filter((p) => p !== "body").join(".") : "";
        return field ? `${field}: ${e.msg ?? ""}` : (e.msg ?? "");
      })
      .filter(Boolean)
      .join("; ") || fallback;
  }
  return fallback;
}
