"use client";

import { useStore } from "@/lib/store";
import { useEffect } from "react";
import AircraftModel from "@/components/3d/AircraftModel";
import { Filter } from "lucide-react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Environment } from "@react-three/drei";

export default function Dashboard() {
  const { aircraft } = useStore();

  // Determine system status based on aircraft state
  const hasHydraulicsFault = aircraft.hydraulics.status !== "Healthy";
  const hasEngineFault = aircraft.engine.status !== "Healthy";
  const hasEcsFault = aircraft.ecs.status !== "Healthy";
  const engineRUL = aircraft.engine.metrics.rulCycles ?? 98;
  const faults = aircraft.crossDomainAlerts || [];

  // Generate dynamic event logs based on current subsystem statuses and alerts
  const logs: any[] = [];
  
  if (hasEngineFault) {
    logs.push({
      id: "eng-1",
      timestamp: new Date().toISOString(),
      level: aircraft.engine.status === "Critical" ? "CRITICAL" : "WARNING",
      message: `Engine health at ${aircraft.engine.score}% - RUL: ${engineRUL} cycles`,
      subsystem: "engine"
    });
  }
  if (hasHydraulicsFault) {
    logs.push({
      id: "hyd-1",
      timestamp: new Date().toISOString(),
      level: aircraft.hydraulics.status === "Critical" ? "CRITICAL" : "WARNING",
      message: `Hydraulics pressure anomaly detected`,
      subsystem: "hydraulics"
    });
  }
  if (hasEcsFault) {
    logs.push({
      id: "ecs-1",
      timestamp: new Date().toISOString(),
      level: aircraft.ecs.status === "Critical" ? "CRITICAL" : "WARNING",
      message: `ECS fouling detected: ${aircraft.ecs.metrics.foulingPct ?? 0}%`,
      subsystem: "ecs"
    });
  }
  faults.forEach((alert, i) => {
    logs.push({
      id: `alert-${i}`,
      timestamp: new Date().toISOString(),
      level: "CRITICAL",
      message: alert,
      subsystem: "fusion"
    });
  });

  return (
    <main className="flex-grow relative flex flex-col items-center justify-center min-h-[calc(100vh-140px)] w-full">
      {/* Background Layer */}
      <div className="absolute inset-0 z-0">
        <img 
          alt="" 
          className="w-full h-full object-cover opacity-30 mix-blend-screen pointer-events-none" 
          src="https://lh3.googleusercontent.com/aida-public/AB6AXuBynMnKqD519f0vFCofhAhce83aySb9sIBNur9qYgRMZrlXX59EwDjng4mSCjvtx6YBnWwes7Ur6EaGuA3N7o5vxKnVLj5BGC5SbUk1fQaMB2nN3eCg9m0kXUJkrL2NmZ8fwoQvM7v568VjL8sA7CIYatJYE7yEWTfDYS28TfKhBngrualBAReGEFOs5cF6w0xyXC5xx062gB7E6YKQqa-g3WY9P14K8xBApwBa3ezZatIjFjjIr_niPBfN3G0kNW9POMauvFtNV-E" 
        />
        <div className="absolute inset-0 grid-bg pointer-events-none"></div>
        <div className="absolute inset-0 bg-gradient-to-b from-obsidian-base via-transparent to-obsidian-base pointer-events-none"></div>
      </div>

      <div className="relative z-10 w-full max-w-[1440px] mx-auto px-8 lg:px-12 py-8 flex-grow flex flex-col md:flex-row gap-8">
        
        {/* Left Panel: Subsystem Status */}
        <aside className="w-full md:w-1/4 flex flex-col gap-1">
          <div className="bg-panel-surface border border-secondary/20 rounded p-4 flex flex-col gap-4">
            <h2 className="text-[11px] font-bold uppercase tracking-widest text-bronze-oxide border-b border-secondary/20 pb-2 mb-2">
              Subsystem Health
            </h2>
            
            {/* Engine */}
            <div className={`flex flex-col gap-1 border-t-2 pt-2 ${hasEngineFault ? "border-status-critical tactical-glow" : "border-tactical-amber"}`}>
              {hasEngineFault && (
                <div className="flex gap-2 items-start border-l-2 border-status-critical pl-2 bg-surface-container-low/50 p-2 mb-2">
                  <span className="text-on-surface-variant/70 min-w-[60px] text-xs">WARNING</span>
                  <div className="flex flex-col">
                    <span className="text-status-critical font-bold text-xs">ENGINE L/R (BiLSTM)</span>
                    <span className="text-on-surface text-xs">RUL degradation detected.</span>
                  </div>
                </div>
              )}
              <div className="flex justify-between items-center text-[11px] font-bold uppercase tracking-widest text-on-surface">
                <span>ENGINE L/R (BiLSTM)</span>
                <span className={hasEngineFault ? "text-status-critical critical-blink" : "text-tactical-amber"}>
                  {engineRUL}% RUL
                </span>
              </div>
              <div className="w-full bg-surface-dim border border-secondary/10 h-2 rounded-full overflow-hidden">
                <div className={`${hasEngineFault ? "bg-status-critical" : "bg-tactical-amber"} h-full`} style={{ width: `${engineRUL}%` }}></div>
              </div>
            </div>

            {/* Hydraulics */}
            <div className={`flex flex-col gap-1 border-t-2 pt-2 ${hasHydraulicsFault ? "border-status-critical tactical-glow" : "border-status-optimal"}`}>
              <div className="flex justify-between items-center text-[11px] font-bold uppercase tracking-widest text-on-surface">
                <span>HYDRAULICS (Autoencoder)</span>
                {hasHydraulicsFault ? (
                  <span className="text-status-critical critical-blink">ANOMALY</span>
                ) : (
                  <span className="text-status-optimal">NOMINAL</span>
                )}
              </div>
              <div className="w-full bg-surface-dim border border-secondary/10 h-2 rounded-full overflow-hidden">
                <div className={`${hasHydraulicsFault ? "bg-status-critical" : "bg-status-optimal"} h-full`} style={{ width: hasHydraulicsFault ? "45%" : "98%" }}></div>
              </div>
            </div>

            {/* Landing Gear */}
            <div className="flex flex-col gap-1 border-t-2 border-status-optimal pt-2">
              <div className="flex justify-between items-center text-[11px] font-bold uppercase tracking-widest text-on-surface">
                <span>LANDING GEAR (XGBoost)</span>
                <span className="text-status-optimal">NOMINAL</span>
              </div>
              <div className="w-full bg-surface-dim border border-secondary/10 h-2 rounded-full overflow-hidden">
                <div className="bg-status-optimal h-full" style={{ width: "98%" }}></div>
              </div>
            </div>

            {/* ECS */}
            <div className={`flex flex-col gap-1 border-t-2 pt-2 ${hasEcsFault ? "border-status-critical tactical-glow" : "border-status-optimal"}`}>
              <div className="flex justify-between items-center text-[11px] font-bold uppercase tracking-widest text-on-surface">
                <span>ECS</span>
                {hasEcsFault ? (
                  <span className="text-status-critical critical-blink">FAULT</span>
                ) : (
                  <span className="text-status-optimal">NOMINAL</span>
                )}
              </div>
              <div className="w-full bg-surface-dim border border-secondary/10 h-2 rounded-full overflow-hidden">
                <div className={`${hasEcsFault ? "bg-status-critical" : "bg-status-optimal"} h-full`} style={{ width: hasEcsFault ? "20%" : "95%" }}></div>
              </div>
            </div>
          </div>

          {/* Model Confidence */}
          <div className="bg-panel-surface border border-secondary/20 rounded p-4 mt-auto">
            <h3 className="text-[11px] font-bold uppercase tracking-widest text-bronze-oxide mb-2 border-b border-secondary/20 pb-1">
              Model Inference
            </h3>
            {hasEcsFault ? (
              <div className="flex gap-2 items-start border-l-2 border-status-critical pl-2 bg-surface-container-low/50 p-2 mb-2">
                <span className="text-on-surface-variant/70 min-w-[60px] text-xs">LIVE</span>
                <div className="flex flex-col">
                  <span className="text-status-critical font-bold text-xs">ECS DEGRADATION</span>
                  <span className="text-on-surface text-xs">Valve unresponsive in cooling loop.</span>
                </div>
              </div>
            ) : (
               <div className="flex gap-2 items-start border-l-2 border-tactical-amber pl-2 bg-surface-container-low/50 p-2 mb-2">
                <span className="text-on-surface-variant/70 min-w-[60px] text-xs">SYS</span>
                <div className="flex flex-col">
                  <span className="text-tactical-amber font-bold text-xs">NOMINAL OPS</span>
                  <span className="text-on-surface text-xs">All inference targets within bounds.</span>
                </div>
              </div>
            )}
            <div className="flex justify-between items-center text-[18px] tracking-wide font-medium text-on-surface">
              <span>CONFIDENCE:</span>
              <span className={hasEcsFault ? "text-status-critical" : "text-tactical-amber"}>
                {hasEcsFault ? "98.7%" : "94.2%"}
              </span>
            </div>
            <div className="text-[11px] font-bold uppercase tracking-widest text-on-surface-variant/70 mt-2">
              DATASET: NASA C-MAPSS, UCI HYD, SYNTHETIC
            </div>
          </div>
        </aside>

        {/* Center Panel: 3D Viz */}
        <section className="w-full md:w-1/2 flex flex-col relative min-h-[400px] border border-secondary/20 rounded bg-surface-dim/80 overflow-hidden backdrop-blur-sm">
          <div className="absolute top-4 left-4 z-20 text-[11px] font-bold uppercase tracking-widest text-bronze-oxide">
            ASSET: A/C-77X <span className="text-on-surface-variant">/</span> MAIN FUSELAGE
          </div>
          
          <div className="absolute top-4 right-4 z-20 flex flex-col gap-2">
            {faults.map((alert, idx) => (
              <span key={idx} className="bg-status-critical text-on-error text-[11px] font-bold uppercase tracking-widest px-2 py-1 rounded critical-blink shadow-[0_0_8px_var(--color-status-critical)]">
                {alert}
              </span>
            ))}
          </div>

          {/* 3D Model */}
          <div className="flex-grow w-full h-full relative">
             <Canvas camera={{ position: [0, 5, 15], fov: 50 }}>
               <ambientLight intensity={0.5} />
               <directionalLight position={[10, 10, 5]} intensity={1} />
               <Environment preset="city" />
               <AircraftModel />
               <OrbitControls enablePan={false} maxPolarAngle={Math.PI / 1.5} minDistance={5} maxDistance={20} />
             </Canvas>
          </div>

          <div className="absolute bottom-4 left-0 w-full px-4 flex justify-between items-end z-20 pointer-events-none">
            <div className="text-[18px] tracking-wide font-medium text-on-surface bg-surface-dim/50 p-2 rounded backdrop-blur">
              M. 0.82 <br /> <span className="text-bronze-oxide text-sm">FL350</span>
            </div>
            <div className="flex gap-4 bg-surface-dim/50 p-2 rounded backdrop-blur">
              <div className="text-center">
                <div className="text-[11px] font-bold uppercase tracking-widest text-bronze-oxide">PITCH</div>
                <div className="text-[18px] tracking-wide font-medium text-on-surface">+2.4°</div>
              </div>
              <div className="text-center">
                <div className="text-[11px] font-bold uppercase tracking-widest text-bronze-oxide">ROLL</div>
                <div className="text-[18px] tracking-wide font-medium text-on-surface">0.0°</div>
              </div>
            </div>
          </div>
        </section>

        {/* Right Panel: Event Log */}
        <aside className="w-full md:w-1/4 flex flex-col">
          <div className="bg-panel-surface border border-secondary/20 rounded h-full flex flex-col">
            <div className="p-4 border-b border-secondary/20 flex justify-between items-center">
              <h2 className="text-[11px] font-bold uppercase tracking-widest text-bronze-oxide">Event Log</h2>
              <Filter className="w-4 h-4 text-bronze-oxide" />
            </div>
            <div className="flex-grow overflow-y-auto p-4 flex flex-col gap-2 text-[11px] font-bold uppercase tracking-widest">
              
              {logs.map(log => {
                let borderClass = "border-secondary/30";
                let textClass = "text-bronze-oxide";
                
                if (log.level === "CRITICAL") {
                  borderClass = "border-status-critical";
                  textClass = "text-status-critical";
                } else if (log.level === "WARNING") {
                  borderClass = "border-tactical-amber";
                  textClass = "text-tactical-amber";
                }

                return (
                  <div key={log.id} className={`flex gap-2 items-start border-l-2 ${borderClass} pl-2 bg-surface-dim/50 p-2`}>
                    <span className="text-on-surface-variant/70 min-w-[60px]">{log.timestamp.split('T')[1].substring(0, 8)}</span>
                    <div className="flex flex-col">
                      <span className={`${textClass} font-bold`}>{log.level}: {log.message}</span>
                      <span className="text-on-surface/70 mt-1 capitalize">{log.subsystem} Subsystem</span>
                    </div>
                  </div>
                )
              })}

              {logs.length === 0 && (
                <div className="flex gap-2 items-start border-l-2 border-status-optimal pl-2 p-2">
                  <span className="text-on-surface-variant/70 min-w-[60px]">--:--:--</span>
                  <div className="flex flex-col">
                    <span className="text-status-optimal font-bold">SYSTEM READY</span>
                    <span className="text-on-surface">Awaiting telemetry...</span>
                  </div>
                </div>
              )}
            </div>
            <div className="p-4 border-t border-secondary/20">
              <button 
                onClick={() => {
                  const logText = logs.length > 0 
                    ? logs.map(l => `[${l.timestamp}] ${l.level}: ${l.message} (${l.subsystem} Subsystem)`).join("\\n")
                    : "SYSTEM READY - Awaiting telemetry...";
                  const blob = new Blob([logText], { type: "text/plain" });
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement("a");
                  a.href = url;
                  a.download = `aerosentinal-logs-${new Date().toISOString().replace(/:/g, '-')}.txt`;
                  document.body.appendChild(a);
                  a.click();
                  document.body.removeChild(a);
                  URL.revokeObjectURL(url);
                }}
                className="w-full bg-tactical-amber text-on-primary-fixed text-[11px] font-bold uppercase tracking-widest py-2 rounded hover:bg-primary-fixed-dim transition-colors active:scale-95"
              >
                EXPORT LOGS
              </button>
            </div>
          </div>
        </aside>
      </div>
    </main>
  );
}
