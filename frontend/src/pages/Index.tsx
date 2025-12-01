import { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { RoomCard } from "@/components/RoomCard";
import { RoomDetailDialog } from "@/components/RoomDetailDialog";
import { Building2, Users, AlertCircle } from "lucide-react";

type Room = {
  id: string;
  name: string;
  availableSeats: number;
  totalSeats: number;
  cleaningStatus: "clean" | "needs_cleaning" | "in_progress";
};

const initialRooms: Room[] = [
  { id: "102", name: "Collaboration Hub", availableSeats: 3, totalSeats: 16, cleaningStatus: "needs_cleaning" },
  { id: "101", name: "Focus Zone", availableSeats: 8, totalSeats: 12, cleaningStatus: "clean" },
  { id: "103", name: "Quiet Room", availableSeats: 5, totalSeats: 8, cleaningStatus: "clean" },
  { id: "104", name: "Innovation Lab", availableSeats: 12, totalSeats: 20, cleaningStatus: "clean" },
  { id: "105", name: "Meeting Pods", availableSeats: 0, totalSeats: 6, cleaningStatus: "needs_cleaning" },
  { id: "106", name: "Kitchen and Dining Area", availableSeats: 15, totalSeats: 24, cleaningStatus: "clean" },
];

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.08 }
  }
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.5, ease: "easeOut" as const }
  }
};

const Index = () => {
  const [rooms, setRooms] = useState<Room[]>(initialRooms);
  const [selectedRoom, setSelectedRoom] = useState<Room | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let mounted = true;
    import("@/lib/api").then(({ fetchRooms, createNotificationsSocket }) => {
      fetchRooms().then((data) => {
        if (!mounted) return;
        // map fields from backend to local shape if needed
        const mapped = data.map((r: any) => ({
          id: r.room_id,
          name: `Room ${r.room_id}`,
          availableSeats: Math.max(0, r.capacity - (r.num_people ?? 0)),
          totalSeats: r.capacity,
          cleaningStatus: r.cleanliness ?? "clean",
        }));
        setRooms(mapped);
      }).catch(() => {
        // keep initialRooms if backend not available
      });

      const ws = createNotificationsSocket((data: any) => {
        if (data.type === "warning") {
          // refresh rooms quickly
          fetchRooms().then((d) => {
            const mapped = d.map((r: any) => ({
              id: r.room_id,
              name: `Room ${r.room_id}`,
              availableSeats: Math.max(0, r.capacity - (r.num_people ?? 0)),
              totalSeats: r.capacity,
              cleaningStatus: r.cleanliness ?? "clean",
            }));
            setRooms(mapped);
          }).catch(() => {});
        }
      });
      wsRef.current = ws;
    });

    return () => {
      mounted = false;
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  const handleRoomClick = (room: Room) => {
    setSelectedRoom(room);
    setDialogOpen(true);
  };

  const handleStatusUpdate = (roomId: string, newStatus: "clean" | "needs_cleaning" | "in_progress") => {
    setRooms(prevRooms => 
      prevRooms.map(room => 
        room.id === roomId ? { ...room, cleaningStatus: newStatus } : room
      )
    );
    if (selectedRoom?.id === roomId) {
      setSelectedRoom(prev => prev ? { ...prev, cleaningStatus: newStatus } : null);
    }
  };

  const totalSeats = rooms.reduce((sum, room) => sum + room.totalSeats, 0);
  const availableSeats = rooms.reduce((sum, room) => sum + room.availableSeats, 0);
  const roomsNeedingCleaning = rooms.filter(room => room.cleaningStatus === "needs_cleaning").length;

  return (
    <div className="min-h-screen w-full bg-background">
      <div className="max-w-[1400px] mx-auto px-6 py-8">
          <motion.div
            className="mb-12"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: "easeOut" as const }}
          >
          <div className="flex items-center gap-3 mb-2">
            <Building2 className="w-8 h-8 text-primary" strokeWidth={1.5} />
            <h1 className="text-4xl font-light tracking-tight text-foreground">
              Coworking Dashboard
            </h1>
          </div>
          <p className="text-muted-foreground text-sm font-light ml-11">
            Real-time space monitoring
          </p>
        </motion.div>

        <motion.div
          className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8"
          variants={containerVariants}
          initial="hidden"
          animate="visible"
        >
          <motion.div variants={itemVariants}>
            <div className="relative bg-card border border-border rounded-xl p-6 overflow-hidden group">
              <div className="absolute inset-0 bg-gradient-to-br from-white/[0.02] to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
              <div className="relative">
                <div className="flex items-center justify-between mb-4">
                  <span className="text-muted-foreground text-sm font-light">Total Rooms</span>
                  <Building2 className="w-5 h-5 text-muted-foreground" strokeWidth={1.5} />
                </div>
                  <motion.span
                    className="text-5xl font-light tracking-tight text-foreground"
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.5, delay: 0.2, ease: "easeOut" as const }}
                  >
                  {rooms.length}
                </motion.span>
              </div>
            </div>
          </motion.div>

          <motion.div variants={itemVariants}>
            <div className="relative bg-card border border-border rounded-xl p-6 overflow-hidden group">
              <div className="absolute inset-0 bg-gradient-to-br from-white/[0.02] to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
              <div className="relative">
                <div className="flex items-center justify-between mb-4">
                  <span className="text-muted-foreground text-sm font-light">Available Seats</span>
                  <Users className="w-5 h-5 text-muted-foreground" strokeWidth={1.5} />
                </div>
                <div className="flex items-baseline gap-1">
                  <motion.span
                    className="text-5xl font-light tracking-tight text-success"
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.5, delay: 0.2, ease: "easeOut" as const }}
                  >
                    {availableSeats}
                  </motion.span>
                  <span className="text-muted-foreground text-xl font-light">/ {totalSeats}</span>
                </div>
              </div>
            </div>
          </motion.div>

          <motion.div variants={itemVariants}>
            <div className="relative bg-card border border-border rounded-xl p-6 overflow-hidden group">
              <div className="absolute inset-0 bg-gradient-to-br from-white/[0.02] to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
              <div className="relative">
                <div className="flex items-center justify-between mb-4">
                  <span className="text-muted-foreground text-sm font-light">Needs Cleaning</span>
                  <AlertCircle className="w-5 h-5 text-muted-foreground" strokeWidth={1.5} />
                </div>
                  <motion.span
                    className="text-5xl font-light tracking-tight text-destructive"
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.5, delay: 0.2, ease: "easeOut" as const }}
                  >
                  {roomsNeedingCleaning}
                </motion.span>
              </div>
            </div>
          </motion.div>
        </motion.div>

        <motion.div
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4"
          variants={containerVariants}
          initial="hidden"
          animate="visible"
        >
          {rooms.map((room) => (
            <motion.div key={room.id} variants={itemVariants}>
              <RoomCard room={room} onClick={() => handleRoomClick(room)} />
            </motion.div>
          ))}
        </motion.div>
      </div>

      <RoomDetailDialog
        key={selectedRoom ? selectedRoom.id : 'room-dialog-none'}
        room={selectedRoom}
        open={dialogOpen}
        onOpenChange={setDialogOpen}
        onStatusUpdate={handleStatusUpdate}
      />
    </div>
  );
};

export default Index;
