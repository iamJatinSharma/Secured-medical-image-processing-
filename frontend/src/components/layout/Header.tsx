"use client";

import { useRouter } from "next/navigation";
import { LogOut, Bell, Settings } from "lucide-react";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useAuthStore } from "@/store/authStore";

export function Header() {
  const router = useRouter();
  const { user, logout } = useAuthStore();

  const handleLogout = () => {
    logout();
    router.push("/auth/login");
  };

  return (
    <header className="sticky top-0 z-30 h-16 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 border-b border-border">
      <div className="flex items-center justify-between h-full px-6">
        <div className="flex items-center gap-4">
          <h1 className="text-lg font-semibold text-foreground">
            Medical Image Processing
          </h1>
        </div>

        <div className="flex items-center gap-3">
          <Button
            variant="ghost"
            size="icon"
            className="relative text-muted-foreground hover:text-foreground"
            title="Notifications"
          >
            <Bell className="w-5 h-5" />
            <Badge
              variant="destructive"
              className="absolute -top-1 -right-1 h-5 w-5 flex items-center justify-center p-0 text-xs"
            >
              3
            </Badge>
          </Button>

          <Button
            variant="ghost"
            size="icon"
            className="text-muted-foreground hover:text-foreground"
            title="Settings"
          >
            <Settings className="w-5 h-5" />
          </Button>

          <div className="flex items-center gap-3 pl-3 border-l border-border">
            <div className="flex flex-col items-end">
              <span className="text-sm font-medium text-foreground">
                {user?.username || "Dr. Smith"}
              </span>
              <span className="text-xs text-muted-foreground">
                {user?.role || "Radiologist"}
              </span>
            </div>
            <Avatar className="w-9 h-9">
              <AvatarImage src="/avatar.png" alt="User" />
              <AvatarFallback className="bg-primary text-primary-foreground">
                {user?.username?.charAt(0).toUpperCase() || "DS"}
              </AvatarFallback>
            </Avatar>
          </div>

          <Button
            variant="ghost"
            size="icon"
            onClick={handleLogout}
            className="text-muted-foreground hover:text-destructive"
            title="Logout"
          >
            <LogOut className="w-5 h-5" />
          </Button>
        </div>
      </div>
    </header>
  );
}
