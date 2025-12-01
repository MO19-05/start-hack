import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Users, AlertCircle, CheckCircle2, CircleDotDashed, Send, CheckCircle, Video } from "lucide-react";
import { toast } from "sonner";
import { sendAlert, BASE } from "@/lib/api";
import kitchenImage from "@/assets/kitchen-dining-area.png";
import collaborationHubImage from "@/assets/collaboration-hub.png";
import focusZoneImage from "@/assets/focus-zone.png";

interface RoomDetailDialogProps {
  room: {
    id: string;
    name: string;
    availableSeats: number;
    totalSeats: number;
    cleaningStatus: "clean" | "needs_cleaning" | "in_progress";
  } | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onStatusUpdate: (roomId: string, newStatus: "clean" | "needs_cleaning" | "in_progress") => void;
}

export function RoomDetailDialog({ room, open, onOpenChange, onStatusUpdate }: RoomDetailDialogProps) {
  const [message, setMessage] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [showRoomView, setShowRoomView] = useState(false);
  const [frameUrl, setFrameUrl] = useState<string | null>(null);

  if (!room) return null;

  const occupancyRate = ((room.totalSeats - room.availableSeats) / room.totalSeats) * 100;

  const getStatusConfig = () => {
    switch (room.cleaningStatus) {
      case "clean":
        return {
          className: "status-clean",
          label: "Clean",
          icon: CheckCircle2
        };
      case "in_progress":
        return {
          className: "status-in-progress",
          label: "In Progress",
          icon: CircleDotDashed
        };
      case "needs_cleaning":
        return {
          className: "status-needs-cleaning",
          label: "Needs Cleaning",
          icon: AlertCircle
        };
    }
  };

  const statusConfig = getStatusConfig();
  const StatusIcon = statusConfig.icon;

  const handleSendAlert = async () => {
    if (!message.trim()) {
      toast.error("Please enter a message");
      return;
    }

    setIsSending(true);
    try {
      const ok = await sendAlert(room.id, message.trim());
      if (!ok) throw new Error("send failed");
      toast.success("Cleaning alert sent to staff!");
      onStatusUpdate(room.id, "in_progress");
      setMessage("");
    } catch (error) {
      console.error("Error sending alert:", error);
      toast.error("Failed to send alert. Please try again.");
    } finally {
      setIsSending(false);
    }
  };

  useEffect(() => {
    // Use the backend MJPEG streaming endpoint for a live-like view.
    // When `showRoomView` is true we set the img src to the streaming URL; when false we clear it.
    if (!room) return;

    if (showRoomView) {
      // construct stream URL from BASE
      const streamUrl = `${BASE}/rooms/${room.id}/stream`;
      setFrameUrl(streamUrl);
    } else {
      // If we were showing a blob URL previously, revoke it; otherwise just clear
      if (frameUrl && frameUrl.startsWith("blob:")) {
        try { URL.revokeObjectURL(frameUrl); } catch (e) {}
      }
      setFrameUrl(null);
    }

    return () => {
      if (frameUrl && frameUrl.startsWith("blob:")) {
        try { URL.revokeObjectURL(frameUrl); } catch (e) {}
      }
    };
  }, [showRoomView, room]);

  const handleMarkAsClean = () => {
    onStatusUpdate(room.id, "clean");
    toast.success("Room marked as clean!");
  };

  const getRoomImage = () => {
    switch (room.name) {
      case "Collaboration Hub":
        return collaborationHubImage;
      case "Focus Zone":
        return focusZoneImage;
      case "Kitchen and Dining Area":
        return kitchenImage;
      default:
        return kitchenImage;
    }
  };

  return (
    <Dialog key={room.id} open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg bg-card border-border max-h-[90vh] flex flex-col">
        <DialogHeader>
          <DialogTitle className="text-3xl font-light tracking-tight text-foreground">
            {room.name}
          </DialogTitle>
          <DialogDescription className="font-light text-muted-foreground">
            Room {room.id} - Detailed Status
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6 py-4 overflow-y-auto">
          <div className="flex items-center justify-between">
            <motion.div
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-light ${statusConfig.className}`}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.3, ease: "easeOut" as const }}
            >
              <StatusIcon className="w-4 h-4" strokeWidth={2} />
              {statusConfig.label}
            </motion.div>
          </div>

          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <Users className="w-6 h-6 text-muted-foreground" strokeWidth={1.5} />
              <div className="flex-1">
                <div className="flex items-baseline gap-2 mb-2">
                  <span className="text-4xl font-light text-foreground tracking-tight">
                    {room.availableSeats}
                  </span>
                  <span className="text-sm text-muted-foreground font-light">
                    / {room.totalSeats} seats available
                  </span>
                </div>
                <div className="w-full bg-muted rounded-full h-2.5 overflow-hidden">
                  <motion.div 
                    className="h-full bg-gradient-to-r from-primary to-primary/80 rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${occupancyRate}%` }}
                    transition={{ duration: 1, ease: "easeOut" as const }}
                  />
                </div>
                <p className="text-xs text-muted-foreground mt-1 font-light">
                  {occupancyRate.toFixed(0)}% occupied
                </p>
              </div>
            </div>
          </div>

          {room.cleaningStatus === "needs_cleaning" && (
            <div className="space-y-3 p-4 bg-destructive/5 rounded-lg border border-destructive/20">
              <h4 className="font-light text-base text-destructive tracking-tight">
                Send Cleaning Alert
              </h4>
              
              <Button
                onClick={() => setShowRoomView(!showRoomView)}
                variant="outline"
                className="w-full"
                size="lg"
              >
                <Video className="w-4 h-4 mr-2" />
                {showRoomView ? "Hide Room View" : "View Room Now"}
              </Button>

              {showRoomView && (
                <div className="relative w-full aspect-video bg-secondary rounded-lg overflow-hidden border border-border">
                  {frameUrl ? (
                    <img src={frameUrl} alt="Room view" className="w-full h-full object-cover" />
                  ) : (
                    <img src={getRoomImage()} alt="Room view" className="w-full h-full object-cover" />
                  )}
                </div>
              )}

              <Textarea
                placeholder="Add a message for the cleaning staff (e.g., priority areas, special instructions...)"
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                className="min-h-[100px] resize-none"
              />
              <Button
                onClick={handleSendAlert}
                disabled={isSending}
                className="w-full"
                size="lg"
              >
                <Send className="w-4 h-4 mr-2" />
                {isSending ? "Sending..." : "Send Alert to Cleaning Staff"}
              </Button>
            </div>
          )}

          {room.cleaningStatus === "in_progress" && (
            <div className="space-y-3 p-4 bg-warning/5 rounded-lg border border-warning/20">
              <h4 className="font-light text-base text-warning flex items-center gap-2 tracking-tight">
                <CircleDotDashed className="w-4 h-4" />
                Cleaning in Progress
              </h4>
              <p className="text-sm text-muted-foreground font-light">
                The cleaning staff has been notified and is working on this room.
              </p>
              <Button
                onClick={handleMarkAsClean}
                variant="outline"
                className="w-full"
                size="lg"
              >
                <CheckCircle className="w-4 h-4 mr-2" />
                Mark as Clean
              </Button>
            </div>
          )}

          {room.cleaningStatus === "clean" && (
            <div className="space-y-3 p-4 bg-success/5 rounded-lg border border-success/20">
              <p className="text-sm text-success flex items-center gap-2 font-light">
                <CheckCircle className="w-4 h-4" />
                This room is clean and ready for use
              </p>
              
              <Button
                onClick={() => setShowRoomView(!showRoomView)}
                variant="outline"
                className="w-full"
                size="lg"
              >
                <Video className="w-4 h-4 mr-2" />
                {showRoomView ? "Hide Room View" : "View Room Now"}
              </Button>

              {showRoomView && (
                <div className="relative w-full aspect-video bg-secondary rounded-lg overflow-hidden border border-border">
                  {frameUrl ? (
                    <img src={frameUrl} alt="Room view" className="w-full h-full object-cover" />
                  ) : (
                    <img src={getRoomImage()} alt="Room view" className="w-full h-full object-cover" />
                  )}
                </div>
              )}
            </div>
          )}

          <div className="pt-4 border-t border-border">
            <p className="text-xs text-muted-foreground font-light">
              Last updated: {new Date().toLocaleTimeString()}
            </p>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
