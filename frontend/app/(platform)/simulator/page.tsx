"use client";

import { useStore } from "@/lib/store";
import { useEffect } from "react";
import AircraftModel from "@/components/3d/AircraftModel";
import { Settings, Bell, User, Video, Power, Droplets, Wind, FileText, AlertTriangle, Play } from "lucide-react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Environment } from "@react-three/drei";

export default function Simulator() {
  const { aircraft, setPhysicsParam, runSimulation, attribution } = useStore();
  const engineRUL = Number(aircraft.engine.metrics.rulCycles ?? 98);
  const faults = aircraft.crossDomainAlerts || [];
  
  const logs: any[] = attribution ? [{
    id: '1',
    timestamp: new Date().toISOString(),
    level: 'WARNING',
    message: attribution
  }] : [];

  const handleInjectFault = (subsystem: "engine" | "hydraulics" | "ecs", message: string) => {
    if (subsystem === "ecs") {
      setPhysicsParam("ecs", "foulingPct", 85);
    } else if (subsystem === "engine") {
      setPhysicsParam("engine", "degradationCycles", 100);
    } else if (subsystem === "hydraulics") {
      setPhysicsParam("hydraulics", "leakSeverity", 0.7);
    }
    runSimulation();
  };

  const handleReset = () => {
    setPhysicsParam("ecs", "foulingPct", 0);
    setPhysicsParam("engine", "degradationCycles", 0);
    setPhysicsParam("hydraulics", "leakSeverity", 0);
    setPhysicsParam("landingGear", "brakeWearPct", 0);
    runSimulation();
  };

  return (
    <div className="flex flex-1 overflow-hidden font-sans min-h-[calc(100vh-140px)] text-sm">
      {/* SideNavBar */}
      <aside className="bg-panel-surface border-r border-secondary/20 w-64 flex flex-col z-40 hidden md:flex">
        <div className="p-6 border-b border-secondary/20">
          <div className="text-2xl font-semibold text-tactical-amber mb-2">Simulation</div>
          <div className="text-[11px] font-bold uppercase tracking-widest text-on-surface-variant mb-6">Active Session: X-DF-01</div>
          
          <button 
            onClick={handleReset}
            className="w-full bg-tactical-amber text-[#2a1800] text-[11px] font-bold uppercase tracking-widest py-2 px-4 rounded hover:bg-primary transition-colors flex items-center justify-center gap-2 active:scale-95 mb-3"
          >
            <Play className="w-4 h-4 fill-current" />
            RESET SIMULATION
          </button>
        </div>
        
        <div className="flex-1 overflow-y-auto py-4">
          <div className="text-on-surface-variant text-[11px] font-bold uppercase tracking-widest px-4 mb-2 opacity-50">Fault Injection Controls</div>
          
          <button 
            onClick={() => handleInjectFault("engine", "Compressor fouling detected. RUL accelerating.")}
            className="w-full text-left text-on-surface-variant p-4 flex items-center gap-4 hover:bg-surface-dim hover:text-on-surface transition-all active:scale-[0.98]"
          >
            <Power className="w-5 h-5 text-tactical-amber" />
            <div className="flex flex-col">
              <span className="text-[11px] font-bold uppercase tracking-widest text-on-surface">Engines</span>
              <span className="text-[10px] text-on-surface-variant/70">Inject Compressor Foul</span>
            </div>
          </button>
          
          <button 
            onClick={() => handleInjectFault("hydraulics", "Hydraulic leak injected at Actuator 4 (Port Wing).")}
            className="w-full text-left text-on-surface-variant p-4 flex items-center gap-4 hover:bg-surface-dim hover:text-on-surface transition-all active:scale-[0.98]"
          >
            <Droplets className="w-5 h-5 text-status-critical" />
            <div className="flex flex-col">
              <span className="text-[11px] font-bold uppercase tracking-widest text-on-surface">Hydraulics</span>
              <span className="text-[10px] text-on-surface-variant/70">Inject Port Wing Leak</span>
            </div>
          </button>
          
          <button 
             onClick={() => handleInjectFault("ecs", "ECS load compensating. Valve unresponsive.")}
            className="w-full text-left text-on-surface-variant p-4 flex items-center gap-4 hover:bg-surface-dim hover:text-on-surface transition-all active:scale-[0.98]"
          >
            <Wind className="w-5 h-5 text-tactical-amber" />
            <div className="flex flex-col">
              <span className="text-[11px] font-bold uppercase tracking-widest text-on-surface">ECS</span>
              <span className="text-[10px] text-on-surface-variant/70">Inject Valve Failure</span>
            </div>
          </button>
        </div>
      </aside>

      {/* Center Canvas & Right Telemetry */}
      <div className="flex-1 flex flex-col bg-obsidian-base relative overflow-hidden">
        {/* 3D Visualization Area */}
        <div className="flex-1 relative grid-bg flex items-center justify-center p-8 border-b border-secondary/20">
          <div className="absolute inset-0 z-0 pointer-events-auto">
             <Canvas camera={{ position: [0, 5, 15], fov: 50 }}>
               <ambientLight intensity={0.5} />
               <directionalLight position={[10, 10, 5]} intensity={1} />
               <Environment preset="city" />
               <AircraftModel />
               <OrbitControls enablePan={false} maxPolarAngle={Math.PI / 1.5} minDistance={5} maxDistance={20} />
             </Canvas>
          </div>
          
          {/* AR Overlays */}
          <div className="z-10 relative w-full max-w-4xl h-full flex flex-col justify-between py-12 pointer-events-none">
            <div className="flex justify-between items-start w-full">
              <div className="bg-panel-surface/80 border border-secondary/50 p-3 rounded backdrop-blur-sm">
                <div className="text-[11px] font-bold uppercase tracking-widest text-tactical-amber mb-1">NODE: ENG-L</div>
                <div className="text-[18px] tracking-wide font-medium text-white">TEMP: {faults.some(f => typeof f === 'string' && f.toLowerCase().includes('engine')) ? '942°C' : '842°C'}</div>
              </div>
              
              {faults.length > 0 && (
                <div className="bg-panel-surface/80 border border-status-critical/50 p-3 rounded backdrop-blur-sm critical-blink tactical-glow">
                  <div className="text-[11px] font-bold uppercase tracking-widest text-status-critical mb-1">NODE: ALERT</div>
                  <div className="text-[18px] tracking-wide font-medium text-white">FAULT ACTIVE</div>
                </div>
              )}
            </div>
          </div>
        </div>
        
        {/* Bottom Diagnostic Log */}
        <div className="h-48 bg-panel-surface flex flex-col z-20">
          <div className="px-4 py-2 border-b border-secondary/20 flex justify-between items-center bg-surface-dim">
            <span className="text-[11px] font-bold uppercase tracking-widest text-on-surface-variant">Diagnostic Event Log</span>
            <span className="text-[11px] font-bold uppercase tracking-widest text-tactical-amber">Live Tracking</span>
          </div>
          <div className="flex-1 overflow-y-auto p-4 text-[14px] leading-tight space-y-2 font-mono">
             {logs.map((log) => (
                <div key={log.id} className={`flex gap-4 ${log.level === 'CRITICAL' ? 'bg-status-critical/20 p-1 -mx-1 text-on-error-container' : 'opacity-80'}`}>
                  <span className="text-bronze-oxide/70">{log.timestamp.split('T')[1].substring(0, 8)}</span>
                  <span className={log.level === 'CRITICAL' ? 'text-status-critical font-bold' : log.level === 'WARNING' ? 'text-tactical-amber' : 'text-status-optimal'}>
                    [{log.level}]
                  </span>
                  <span className="text-on-surface">{log.message}</span>
                </div>
             ))}
             {logs.length === 0 && (
                <div className="flex gap-4 opacity-70">
                  <span className="text-bronze-oxide/70">--:--:--</span>
                  <span className="text-tactical-amber">[INFO]</span>
                  <span className="text-on-surface">Awaiting telemetry...</span>
                </div>
             )}
          </div>
        </div>
      </div>

      {/* Right Sidebar - Telemetry */}
      <aside className="w-80 bg-panel-surface border-l border-secondary/20 flex flex-col z-30 hidden lg:flex">
        <div className="p-4 border-b border-secondary/20 bg-surface-dim">
          <h3 className="text-[11px] font-bold uppercase tracking-widest text-tactical-amber">Real-Time Telemetry</h3>
        </div>
        
        <div className="flex-1 overflow-y-auto p-4 space-y-6">
          {/* Health Overview */}
          <div className="space-y-2">
            <div className="flex justify-between items-end">
              <span className="text-[11px] font-bold uppercase tracking-widest text-on-surface-variant">System Health</span>
              <span className={`text-[18px] tracking-wide font-medium ${faults.length > 0 ? 'text-status-critical' : 'text-tactical-amber'}`}>
                {faults.length > 0 ? '42%' : '98%'}
              </span>
            </div>
            <div className="h-2 w-full bg-surface-dim rounded-full overflow-hidden">
              <div className={`h-full ${faults.length > 0 ? 'bg-status-critical' : 'bg-tactical-amber'}`} style={{ width: faults.length > 0 ? '42%' : '98%' }}></div>
            </div>
          </div>
          
          {/* Engine RUL */}
          <div className="border border-secondary/20 rounded p-4 bg-surface-dim border-t-2 border-t-tactical-amber">
            <div className="text-[11px] font-bold uppercase tracking-widest text-on-surface-variant mb-4 flex justify-between">
              <span>Engine RUL</span>
              <Power className="w-4 h-4" />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="text-[10px] text-bronze-oxide/70 mb-1">ENG-L (CYCLES)</div>
                <div className="text-[20px] tracking-wide font-medium text-on-surface">{Math.floor(engineRUL * 14.5)}</div>
              </div>
              <div>
                <div className="text-[10px] text-bronze-oxide/70 mb-1">ENG-R (CYCLES)</div>
                <div className="text-[20px] tracking-wide font-medium text-on-surface">1,388</div>
              </div>
            </div>
          </div>
          
          {/* Attitude */}
          <div className="border border-secondary/20 rounded p-4 bg-surface-dim">
            <div className="text-[11px] font-bold uppercase tracking-widest text-on-surface-variant mb-4 flex justify-between">
              <span>Attitude Dynamics</span>
            </div>
            <div className="space-y-3 text-[14px] font-mono tracking-wide">
              <div className="flex justify-between border-b border-secondary/20 pb-1">
                <span className="text-bronze-oxide/70">PITCH</span>
                <span className={faults.length > 0 ? "text-status-critical" : "text-tactical-amber"}>
                    {faults.length > 0 ? "+4.2°" : "+2.4°"}
                </span>
              </div>
              <div className="flex justify-between border-b border-secondary/20 pb-1">
                <span className="text-bronze-oxide/70">ROLL</span>
                <span className="text-on-surface">{faults.length > 0 ? "-1.5°" : "0.0°"}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-bronze-oxide/70">YAW</span>
                <span className="text-on-surface">-0.1°</span>
              </div>
            </div>
          </div>
        </div>
      </aside>
    </div>
  );
}
