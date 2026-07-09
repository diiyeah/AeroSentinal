"use client";

import { useEffect, useState } from "react";
import { Cpu, AlertTriangle, PlaneLanding, Power, Activity, Download, Timer } from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

export default function MLModels() {
  const [modelStatuses, setModelStatuses] = useState<Record<string, string>>({
    engine: "loading",
    hydraulics: "loading",
    landing_gear: "loading",
    apu: "loading",
    ecs: "physics_model"
  });

  useEffect(() => {
    const fetchStatuses = async () => {
      try {
        const res = await fetch(`${API_BASE}/health`);
        if (res.ok) {
          const data = await res.json();
          if (data.subsystems) {
            setModelStatuses(data.subsystems);
          }
        } else {
          throw new Error();
        }
      } catch {
        setModelStatuses({
          engine: "offline",
          hydraulics: "offline",
          landing_gear: "offline",
          apu: "offline",
          ecs: "offline"
        });
      }
    };

    fetchStatuses();
    const interval = setInterval(fetchStatuses, 5000);
    return () => clearInterval(interval);
  }, []);

  const getStatusDisplay = (subsystem: string) => {
    const status = modelStatuses[subsystem];
    if (status === "model_loaded") {
      return <span className="text-status-optimal tracking-wide font-medium">LIVE (ONNX)</span>;
    } else if (status === "simulation") {
      return <span className="text-tactical-amber tracking-wide font-medium">SIMULATION</span>;
    } else if (status === "physics_model") {
      return <span className="text-status-optimal tracking-wide font-medium">LIVE (PHYSICS)</span>;
    } else if (status === "offline") {
      return <span className="text-status-critical tracking-wide font-medium">OFFLINE</span>;
    }
    return <span className="text-on-surface-variant tracking-wide font-medium">LOADING...</span>;
  };

  const getCardBorder = (subsystem: string, defaultClass: string = "") => {
    const status = modelStatuses[subsystem];
    if (status === "offline") return "border-t-status-critical border-secondary/20";
    if (status === "simulation") return "border-t-tactical-amber border-secondary/20";
    return defaultClass;
  };

  return (
    <main className="flex-grow px-8 lg:px-12 py-8 max-w-[1440px] mx-auto w-full font-sans">
      <div className="flex items-center gap-2 mb-8 text-secondary/60 text-[11px] font-bold uppercase tracking-widest">
        <span>AEROSENTINAL</span>
        <span>/</span>
        <span className="text-tactical-amber">ML MODELS</span>
      </div>

      <header className="mb-12">
        <h1 className="text-3xl font-bold text-on-surface mb-2 tracking-tight">Machine Learning Subsystems</h1>
        <p className="text-on-surface-variant max-w-2xl leading-relaxed">
          Telemetry anomaly detection and remaining useful life (RUL) prediction models. All models are calibrated for high-density, real-time flight instrumentation data.
        </p>
      </header>

      {/* Model Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mb-12">
        
        {/* Turbofan Engine Card */}
        <div className={`bg-panel-surface border rounded-lg p-6 relative flex flex-col border-t-2 ${getCardBorder("engine", "border-t-status-optimal border-secondary/20")}`}>
          <div className="flex justify-between items-start mb-6">
            <div>
              <h2 className="text-xl font-semibold text-on-surface mb-1">Turbofan Engine</h2>
              <span className="text-[11px] font-bold uppercase tracking-widest text-secondary/70">BiLSTM + Attention</span>
            </div>
            <Cpu className={`${modelStatuses.engine === "simulation" ? "text-tactical-amber" : "text-status-optimal"} w-6 h-6`} />
          </div>
          
          <div className="space-y-4 font-mono text-sm mb-6 flex-grow">
            <div className="flex justify-between border-b border-secondary/10 pb-2">
              <span className="text-on-surface-variant">Status</span>
              {getStatusDisplay("engine")}
            </div>
            <div className="flex justify-between border-b border-secondary/10 pb-2">
              <span className="text-on-surface-variant">Dataset</span>
              <span className="text-on-surface">C-MAPSS</span>
            </div>
            <div className="flex justify-between border-b border-secondary/10 pb-2">
              <span className="text-on-surface-variant">Accuracy (RMSE)</span>
              <span className="text-on-surface">12.42</span>
            </div>
            <div className="flex justify-between border-b border-secondary/10 pb-2">
              <span className="text-on-surface-variant">Latency</span>
              <span className="text-on-surface">{modelStatuses.engine === "model_loaded" ? "42ms" : "N/A"}</span>
            </div>
          </div>
          
          <div className="mt-auto">
            <button className="w-full bg-tactical-amber text-on-primary-fixed text-[11px] font-bold uppercase tracking-widest py-3 rounded hover:bg-primary transition-colors active:scale-95">
              VIEW METRICS
            </button>
          </div>
        </div>

        {/* Hydraulics Card */}
        <div className={`bg-panel-surface border rounded-lg p-6 relative flex flex-col border-t-2 ${getCardBorder("hydraulics", "border-t-status-optimal border-secondary/20")}`}>
          <div className="flex justify-between items-start mb-6">
            <div>
              <h2 className="text-xl font-semibold text-on-surface mb-1">Hydraulics</h2>
              <span className="text-[11px] font-bold uppercase tracking-widest text-secondary/70">1D Conv Autoencoder</span>
            </div>
            <AlertTriangle className={`${modelStatuses.hydraulics === "simulation" ? "text-tactical-amber" : "text-status-optimal"} w-6 h-6`} />
          </div>
          
          <div className="space-y-4 font-mono text-sm mb-6 flex-grow">
            <div className="flex justify-between border-b border-secondary/10 pb-2">
              <span className="text-on-surface-variant">Status</span>
              {getStatusDisplay("hydraulics")}
            </div>
            <div className="flex justify-between border-b border-secondary/10 pb-2">
              <span className="text-on-surface-variant">Dataset</span>
              <span className="text-on-surface">UCI Hydraulics</span>
            </div>
            <div className="flex justify-between border-b border-secondary/10 pb-2">
              <span className="text-on-surface-variant">Anomaly Score</span>
              <span className="text-on-surface">{modelStatuses.hydraulics === "model_loaded" ? "Computed" : "Simulated"}</span>
            </div>
            <div className="flex justify-between border-b border-secondary/10 pb-2">
              <span className="text-on-surface-variant">Latency</span>
              <span className="text-on-surface">{modelStatuses.hydraulics === "model_loaded" ? "18ms" : "N/A"}</span>
            </div>
          </div>
          
          <div className="mt-auto">
            <button className="w-full border border-tactical-amber text-tactical-amber text-[11px] font-bold uppercase tracking-widest py-3 rounded hover:bg-tactical-amber/10 transition-colors active:scale-95">
              INVESTIGATE FAULT
            </button>
          </div>
        </div>

        {/* Landing Gear Card */}
        <div className={`bg-panel-surface border rounded-lg p-6 relative flex flex-col border-t-2 ${getCardBorder("landing_gear", "border-t-status-optimal border-secondary/20")}`}>
          <div className="flex justify-between items-start mb-6">
            <div>
              <h2 className="text-xl font-semibold text-on-surface mb-1">Landing Gear</h2>
              <span className="text-[11px] font-bold uppercase tracking-widest text-secondary/70">XGBoost</span>
            </div>
            <PlaneLanding className={`${modelStatuses.landing_gear === "simulation" ? "text-tactical-amber" : "text-status-optimal"} w-6 h-6`} />
          </div>
          
          <div className="space-y-4 font-mono text-sm mb-6 flex-grow">
            <div className="flex justify-between border-b border-secondary/10 pb-2">
              <span className="text-on-surface-variant">Status</span>
              {getStatusDisplay("landing_gear")}
            </div>
            <div className="flex justify-between border-b border-secondary/10 pb-2">
              <span className="text-on-surface-variant">Dataset</span>
              <span className="text-on-surface">Custom Synth</span>
            </div>
            <div className="flex justify-between border-b border-secondary/10 pb-2">
              <span className="text-on-surface-variant">F1 Score</span>
              <span className="text-on-surface">0.94</span>
            </div>
            <div className="flex justify-between border-b border-secondary/10 pb-2">
              <span className="text-on-surface-variant">Latency</span>
              <span className="text-on-surface">{modelStatuses.landing_gear === "model_loaded" ? "5ms" : "N/A"}</span>
            </div>
          </div>
          
          <div className="mt-auto">
            <button className="w-full bg-tactical-amber text-on-primary-fixed text-[11px] font-bold uppercase tracking-widest py-3 rounded hover:bg-primary transition-colors active:scale-95">
              DEPLOY UPDATE
            </button>
          </div>
        </div>

        {/* APU Health Card */}
        <div className={`bg-panel-surface border rounded-lg p-6 relative flex flex-col border-t-2 ${getCardBorder("apu", "border-t-status-optimal border-secondary/20")}`}>
          <div className="flex justify-between items-start mb-6">
            <div>
              <h2 className="text-xl font-semibold text-on-surface mb-1">APU Health</h2>
              <span className="text-[11px] font-bold uppercase tracking-widest text-secondary/70">Random Forest</span>
            </div>
            <Power className={`${modelStatuses.apu === "simulation" ? "text-tactical-amber" : "text-status-optimal"} w-6 h-6`} />
          </div>
          
          <div className="space-y-4 font-mono text-sm mb-6 flex-grow">
            <div className="flex justify-between border-b border-secondary/10 pb-2">
              <span className="text-on-surface-variant">Status</span>
              {getStatusDisplay("apu")}
            </div>
            <div className="flex justify-between border-b border-secondary/10 pb-2">
              <span className="text-on-surface-variant">Dataset</span>
              <span className="text-on-surface">Proprietary</span>
            </div>
            <div className="flex justify-between border-b border-secondary/10 pb-2">
              <span className="text-on-surface-variant">Accuracy</span>
              <span className="text-on-surface">98.2%</span>
            </div>
            <div className="flex justify-between border-b border-secondary/10 pb-2">
              <span className="text-on-surface-variant">Latency</span>
              <span className="text-on-surface">{modelStatuses.apu === "model_loaded" ? "12ms" : "N/A"}</span>
            </div>
          </div>
          
          <div className="mt-auto">
            <button className="w-full bg-tactical-amber text-on-primary-fixed text-[11px] font-bold uppercase tracking-widest py-3 rounded hover:bg-primary transition-colors active:scale-95">
              VIEW METRICS
            </button>
          </div>
        </div>

        {/* ECS Cross-Domain Card */}
        <div className="bg-panel-surface border border-secondary/20 rounded-lg p-6 relative flex flex-col border-t-2 border-t-status-optimal md:col-span-2 lg:col-span-2">
          <div className="flex justify-between items-start mb-6">
            <div>
              <h2 className="text-xl font-semibold text-on-surface mb-1">ECS Cross-Domain</h2>
              <span className="text-[11px] font-bold uppercase tracking-widest text-secondary/70">Brayton-cycle Simulator Surrogate</span>
            </div>
            <Activity className="text-status-optimal w-6 h-6" />
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-6 flex-grow">
            <div className="space-y-4 font-mono text-sm">
              <div className="flex justify-between border-b border-secondary/10 pb-2">
                <span className="text-on-surface-variant">Status</span>
                {getStatusDisplay("ecs")}
              </div>
              <div className="flex justify-between border-b border-secondary/10 pb-2">
                <span className="text-on-surface-variant">Dataset</span>
                <span className="text-on-surface">Simulated Physics</span>
              </div>
            </div>
            <div className="space-y-4 font-mono text-sm">
              <div className="flex justify-between border-b border-secondary/10 pb-2">
                <span className="text-on-surface-variant">Error Bound</span>
                <span className="text-on-surface">± 1.2%</span>
              </div>
              <div className="flex justify-between border-b border-secondary/10 pb-2">
                <span className="text-on-surface-variant">Latency</span>
                <span className="text-on-surface">85ms</span>
              </div>
            </div>
          </div>
          
          <div className="bg-surface-dim p-4 rounded mb-6 font-mono text-sm text-secondary/80">
            <span className="block mb-2 text-tactical-amber">&gt; SYSTEM OUTPUT LOG</span>
            [08:42:11] ECS Model sync initialized... OK<br/>
            [08:42:15] Flow rate variance detected... COMPENSATED<br/>
            [08:42:18] Thermal efficiency... NOMINAL
          </div>
          
          <div className="mt-auto flex gap-4">
            <button className="flex-1 bg-tactical-amber text-on-primary-fixed text-[11px] font-bold uppercase tracking-widest py-3 rounded hover:bg-primary transition-colors active:scale-95">
              VIEW SIMULATION
            </button>
            <button className="flex-1 border border-tactical-amber text-tactical-amber text-[11px] font-bold uppercase tracking-widest py-3 rounded hover:bg-tactical-amber/10 transition-colors active:scale-95">
              CALIBRATE
            </button>
          </div>
        </div>
      </div>

      {/* Export & Benchmarks Section */}
      <section className="bg-panel-surface border border-secondary/20 rounded-lg p-8">
        <h3 className="text-2xl font-semibold text-on-surface mb-6 border-b border-secondary/20 pb-4">Model Export & Benchmarks</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="md:col-span-2">
            <p className="text-on-surface-variant mb-6 leading-relaxed">
              All models are optimized for edge deployment on flight control systems. The standard export format is ONNX (Open Neural Network Exchange), ensuring cross-platform compatibility and rigorous performance benchmarking prior to live integration. Hardware acceleration via TensorRT is supported for compatible APU units.
            </p>
            
            <div className="bg-surface-dim rounded p-4 border border-secondary/10">
              <div className="flex justify-between items-center mb-4">
                <span className="text-[11px] font-bold uppercase tracking-widest text-secondary">LATEST BUILD PIPELINE</span>
                <span className="font-mono text-sm text-status-optimal tracking-wide font-medium">SUCCESS</span>
              </div>
              <div className="space-y-2 font-mono text-sm text-on-surface-variant">
                <div className="flex justify-between"><span className="text-tactical-amber">Turbofan_BiLSTM_v2.1.onnx</span> <span>14.2 MB</span></div>
                <div className="flex justify-between"><span className="text-tactical-amber">Hydraulics_ConvAE_v1.8.onnx</span> <span>3.8 MB</span></div>
                <div className="flex justify-between"><span className="text-tactical-amber">ECS_Surrogate_v3.0.onnx</span> <span>22.5 MB</span></div>
              </div>
            </div>
          </div>
          
          <div className="flex flex-col gap-4 justify-center">
            <button className="w-full bg-surface-dim text-on-surface border border-secondary/20 text-[11px] font-bold uppercase tracking-widest py-4 rounded hover:bg-secondary/10 transition-colors flex items-center justify-center gap-2 active:scale-[0.98]">
              <Download className="w-4 h-4" />
              DOWNLOAD ALL ONNX
            </button>
            <button className="w-full bg-surface-dim text-on-surface border border-secondary/20 text-[11px] font-bold uppercase tracking-widest py-4 rounded hover:bg-secondary/10 transition-colors flex items-center justify-center gap-2 active:scale-[0.98]">
              <Timer className="w-4 h-4" />
              RUN BENCHMARKS
            </button>
          </div>
        </div>
      </section>
    </main>
  );
}
