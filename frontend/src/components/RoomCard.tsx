import { motion } from "framer-motion";
import { Users, CheckCircle2, AlertCircle, CircleDotDashed } from "lucide-react";

interface RoomCardProps {
  room: {
    id: string;
    name: string;
    availableSeats: number;
    totalSeats: number;
    cleaningStatus: "clean" | "needs_cleaning" | "in_progress";
  };
  onClick: () => void;
}

export function RoomCard({ room, onClick }: RoomCardProps) {
  const occupancyRate = ((room.totalSeats - room.availableSeats) / room.totalSeats) * 100;
  
  const getStatusConfig = () => {
    switch (room.cleaningStatus) {
      case "clean":
        return {
          label: "Clean",
          icon: CheckCircle2,
          className: "status-clean"
        };
      case "in_progress":
        return {
          label: "In Progress",
          icon: CircleDotDashed,
          className: "status-in-progress"
        };
      case "needs_cleaning":
        return {
          label: "Needs Cleaning",
          icon: AlertCircle,
          className: "status-needs-cleaning"
        };
    }
  };

  const statusConfig = getStatusConfig();
  const StatusIcon = statusConfig.icon;
  
  return (
    <div 
      className="relative bg-card border border-border rounded-xl p-6 overflow-hidden group hover:border-border/60 transition-colors duration-300 cursor-pointer"
      onClick={onClick}
    >
      <div className="absolute inset-0 bg-gradient-to-br from-white/[0.02] to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />

      <div className="relative">
        <div className="flex items-start justify-between mb-4">
          <div>
            <h3 className="text-lg font-light text-foreground mb-1">
              {room.name}
            </h3>
            <p className="text-sm text-muted-foreground font-light">Room {room.id}</p>
          </div>
          <motion.div
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-light ${statusConfig.className}`}
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3, ease: "easeOut" as const }}
          >
            <StatusIcon className="w-3.5 h-3.5" strokeWidth={2} />
            <span>{statusConfig.label}</span>
          </motion.div>
        </div>

        <div className="mb-4">
          <div className="flex items-center gap-3 mb-3">
            <Users className="w-4 h-4 text-muted-foreground" strokeWidth={1.5} />
            <div className="flex items-baseline gap-1">
              <span className="text-3xl font-light text-foreground">
                {room.availableSeats}
              </span>
              <span className="text-muted-foreground text-sm font-light">
                / {room.totalSeats} seats free
              </span>
            </div>
          </div>

          <div className="relative h-1 bg-muted rounded-full overflow-hidden">
            <motion.div
              className="absolute inset-y-0 left-0 bg-gradient-to-r from-primary to-primary/80 rounded-full"
              initial={{ width: 0 }}
              animate={{ width: `${occupancyRate}%` }}
              transition={{ duration: 1, delay: 0.3, ease: "easeOut" as const }}
            />
          </div>
        </div>

        <div className="text-xs text-muted-foreground font-light">
          Last updated: {new Date().toLocaleTimeString()}
        </div>
      </div>
    </div>
  );
}

