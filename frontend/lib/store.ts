import { create } from 'zustand';
import { useShallow } from 'zustand/react/shallow';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

export type SubsystemHealth = {
  score: number;
  status: 'Healthy' | 'Warning' | 'Critical';
  metrics: Record<string, number | string>;
  is_synthetic_data: boolean;
};

export type AircraftState = {
  aircraftId: string;
  globalHealthScore: number;
  globalStatus: string;
  engine: SubsystemHealth;
  landingGear: SubsystemHealth;
  apu: SubsystemHealth;
  hydraulics: SubsystemHealth;
  ecs: SubsystemHealth;
  crossDomainAlerts: string[];
  acarsMessage: string;
  aogRisk: {
    probability: number;
    totalRiskUsd: number;
    riskLevel: string;
    recommendation: string;
  };
};

export type PhysicsSimInputs = {
  ecs: { foulingPct: number; flightPhase: 'ground'|'climb'|'cruise'|'descent'; ambientTempOffsetK: number };
  apu: { foulingPct: number; egtOffset: number };
  landingGear: { aircraftMassKg: number; wheelVelocityKts: number; brakePressurePsi: number; runwayFriction: number; brakeWearPct: number };
  engine: { degradationCycles: number };
  hydraulics: { leakSeverity: number };
};

export type DatasetTrajectoryInputs = {
  engine: { unitId: string; cycle: number };
  hydraulics: { unitId: string; cycle: number };
};

interface AppState {
  aircraft: AircraftState;
  loading: boolean;
  backendOnline: boolean;
  error: string | null;

  // Granular Inputs
  physicsInputs: PhysicsSimInputs;
  datasetInputs: DatasetTrajectoryInputs;

  // Dissection UI
  dissectAmount: number;
  setDissectAmount: (val: number) => void;

  // Target Zone
  focusedZone: string | null;
  setFocusedZone: (zone: string | null) => void;

  // Simulator comparison results
  naiveEngineScore: number | null;
  fusionEngineScore: number | null;
  attribution: string;

  // History
  history: Array<{ time: number; engineScore: number; ecsFouling: number }>;

  // Actions
  setPhysicsParam: <T extends keyof PhysicsSimInputs, K extends keyof PhysicsSimInputs[T]>(subsystem: T, key: K, val: PhysicsSimInputs[T][K]) => void;
  setDatasetParam: <T extends keyof DatasetTrajectoryInputs, K extends keyof DatasetTrajectoryInputs[T]>(subsystem: T, key: K, val: DatasetTrajectoryInputs[T][K]) => void;
  fetchHealth: () => Promise<void>;
  runSimulation: () => Promise<void>;
  checkBackend: () => Promise<void>;
}

const initialAircraft: AircraftState = {
  aircraftId: "N1234A",
  globalHealthScore: 92,
  globalStatus: "Healthy",
  engine: { score: 95, status: 'Healthy', metrics: { rulCycles: 142 }, is_synthetic_data: true },
  landingGear: { score: 88, status: 'Healthy', metrics: { brakeWearPct: 12 }, is_synthetic_data: true },
  apu: { score: 98, status: 'Healthy', metrics: { egtMargin: 45 }, is_synthetic_data: true },
  hydraulics: { score: 100, status: 'Healthy', metrics: { pressureAnomaly: 0.01 }, is_synthetic_data: true },
  ecs: { score: 100, status: 'Healthy', metrics: { foulingPct: 0 }, is_synthetic_data: true },
  crossDomainAlerts: [],
  acarsMessage: "",
  aogRisk: { probability: 0, totalRiskUsd: 0, riskLevel: "LOW", recommendation: "" },
};

const initialHistory = Array.from({ length: 20 }, (_, i) => ({
  time: i,
  engineScore: 95 + Math.random() * 2 - 1,
  ecsFouling: 0,
}));

function parseFusionResponse(data: any): Partial<AircraftState> {
  const findSub = (name: string) => data.subsystems?.find((s: any) => s.name === name);
  const engine = findSub("engine");
  const hydraulics = findSub("hydraulics");
  const landing_gear = findSub("landing_gear");
  const apu = findSub("apu");
  const ecs = findSub("ecs");

  return {
    aircraftId: data.aircraft_id,
    globalHealthScore: data.global_health_score,
    globalStatus: data.global_status,
    engine: engine ? { score: engine.health_score, status: engine.status, metrics: { rulCycles: engine.rul_estimate ?? 0 }, is_synthetic_data: engine.is_synthetic_data } : initialAircraft.engine,
    hydraulics: hydraulics ? { score: hydraulics.health_score, status: hydraulics.status, metrics: { pressureAnomaly: 0 }, is_synthetic_data: hydraulics.is_synthetic_data } : initialAircraft.hydraulics,
    landingGear: landing_gear ? { score: landing_gear.health_score, status: landing_gear.status, metrics: { brakeWearPct: 0, remainingLandings: landing_gear.rul_estimate ?? 0 }, is_synthetic_data: landing_gear.is_synthetic_data } : initialAircraft.landingGear,
    apu: apu ? { score: apu.health_score, status: apu.status, metrics: { egtMargin: 0 }, is_synthetic_data: apu.is_synthetic_data } : initialAircraft.apu,
    ecs: ecs ? { score: ecs.health_score, status: ecs.status, metrics: { foulingPct: 0 }, is_synthetic_data: ecs.is_synthetic_data } : initialAircraft.ecs,
    crossDomainAlerts: data.cross_domain_alerts || [],
    acarsMessage: data.acars_message || "",
    aogRisk: data.aog_risk ? { probability: data.aog_risk.probability_of_failure, totalRiskUsd: data.aog_risk.total_risk_usd, riskLevel: data.aog_risk.risk_level, recommendation: data.aog_risk.recommendation } : initialAircraft.aogRisk,
  };
}

export const useStore = create<AppState>((set, get) => ({
  aircraft: initialAircraft,
  loading: false,
  backendOnline: false,
  error: null,

  physicsInputs: {
    ecs: { foulingPct: 0, flightPhase: 'cruise', ambientTempOffsetK: 0 },
    apu: { foulingPct: 0, egtOffset: 0 },
    landingGear: { aircraftMassKg: 70000, wheelVelocityKts: 65, brakePressurePsi: 2500, runwayFriction: 0.8, brakeWearPct: 0 },
    engine: { degradationCycles: 0 },
    hydraulics: { leakSeverity: 0 },
  },
  datasetInputs: {
    engine: { unitId: "1", cycle: 100 },
    hydraulics: { unitId: "1", cycle: 100 },
  },

  dissectAmount: 0,
  setDissectAmount: (val) => set({ dissectAmount: val }),

  focusedZone: null,
  setFocusedZone: (zone) => set({ focusedZone: zone }),

  naiveEngineScore: null,
  fusionEngineScore: null,
  attribution: "",
  history: initialHistory,

  checkBackend: async () => {
    try {
      const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
      set({ backendOnline: res.ok });
    } catch {
      set({ backendOnline: false });
    }
  },

  setPhysicsParam: (subsystem, key, val) => set((state) => ({
    physicsInputs: { ...state.physicsInputs, [subsystem]: { ...state.physicsInputs[subsystem], [key]: val } }
  })),

  setDatasetParam: (subsystem, key, val) => set((state) => ({
    datasetInputs: { ...state.datasetInputs, [subsystem]: { ...state.datasetInputs[subsystem], [key]: val } }
  })),

  fetchHealth: async () => {
    const state = get();
    set({ loading: true, error: null });

    try {
      const params = new URLSearchParams({
        aircraft_id: state.aircraft.aircraftId,
        engine_cycle: String(state.datasetInputs.engine.cycle),
        brake_wear_pct: String(state.physicsInputs.landingGear.brakeWearPct),
        apu_fouling: String(state.physicsInputs.apu.foulingPct),
        ecs_fouling: String(state.physicsInputs.ecs.foulingPct),
      });

      const res = await fetch(`${API_BASE}/fusion/health?${params}`);
      if (!res.ok) throw new Error(`API returned ${res.status}`);

      const data = await res.json();
      const parsed = parseFusionResponse(data);

      const newEntry = {
        time: state.history[state.history.length - 1].time + 1,
        engineScore: parsed.engine?.score ?? state.aircraft.engine.score,
        ecsFouling: state.physicsInputs.ecs.foulingPct,
      };

      set({
        aircraft: { ...state.aircraft, ...parsed } as AircraftState,
        history: [...state.history.slice(1), newEntry],
        loading: false,
        backendOnline: true,
      });
    } catch (err: any) {
      // Offline mock fallback
      const val = state.physicsInputs.ecs.foulingPct;
      const baseEngineScore = 95;
      const couplingEffect = (val / 100) * 15;
      const newEngineScore = Math.max(0, baseEngineScore - couplingEffect);
      const newEcsScore = Math.max(0, 100 - val);

      set({
        aircraft: {
          ...state.aircraft,
          engine: { ...state.aircraft.engine, score: Math.floor(newEngineScore), status: newEngineScore < 50 ? 'Critical' : newEngineScore < 75 ? 'Warning' : 'Healthy' },
          ecs: { ...state.aircraft.ecs, score: Math.floor(newEcsScore), status: newEcsScore < 50 ? 'Critical' : newEcsScore < 75 ? 'Warning' : 'Healthy', metrics: { foulingPct: val } },
        },
        loading: false,
        backendOnline: false,
      });
    }
  },

  runSimulation: async () => {
    const state = get();
    set({ loading: true, error: null });

    try {
      const res = await fetch(`${API_BASE}/simulate/what-if`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ecs_fouling_pct: state.physicsInputs.ecs.foulingPct,
          engine_degradation_cycles: state.physicsInputs.engine.degradationCycles,
          hydraulic_leak_severity: state.physicsInputs.hydraulics.leakSeverity,
          brake_wear_pct: state.physicsInputs.landingGear.brakeWearPct,
          apu_fouling_factor: state.physicsInputs.apu.foulingPct / 100,
        }),
      });

      if (!res.ok) throw new Error(`API returned ${res.status}`);
      const data = await res.json();

      const naiveEngine = data.naive_assessment?.subsystems?.find((s: any) => s.name === "engine");
      const fusionEngine = data.fusion_assessment?.subsystems?.find((s: any) => s.name === "engine");
      const parsed = parseFusionResponse(data.fusion_assessment);

      set({
        aircraft: { ...state.aircraft, ...parsed } as AircraftState,
        naiveEngineScore: naiveEngine?.health_score ?? null,
        fusionEngineScore: fusionEngine?.health_score ?? null,
        attribution: data.attribution_explanation || "",
        loading: false,
        backendOnline: true,
      });
    } catch (err: any) {
      // Offline fallback
      const ecsFouling = state.physicsInputs.ecs.foulingPct;
      const naiveScore = Math.max(0, 95 - (ecsFouling / 100) * 15);
      const fusionScore = Math.min(100, naiveScore + (ecsFouling / 100) * 15);
      set({
        naiveEngineScore: Math.floor(naiveScore),
        fusionEngineScore: Math.floor(fusionScore),
        attribution: ecsFouling > 10 ? `[Offline] ECS fouling at ${ecsFouling}% would cause false engine depression.` : "[Offline] All subsystems nominal.",
        loading: false,
        backendOnline: false,
      });
    }
  },
}));

// Selector hooks for components to subscribe to specific zone colors.
// IMPORTANT: useShallow is required here because the selector returns an object.
// Without it, Zustand's useSyncExternalStore sees a NEW object reference on every
// render and triggers an infinite update loop.
export const useZoneStatusColor = (zone: string) => {
  const mapping: Record<string, keyof AircraftState> = {
    zone_engine: 'engine',
    zone_ecs: 'ecs',
    zone_apu: 'apu',
    zone_landing_gear: 'landingGear',
    zone_hydraulics: 'hydraulics',
    engine: 'engine',
    ecs: 'ecs',
    apu: 'apu',
    landingGear: 'landingGear',
    hydraulics: 'hydraulics',
  };

  return useStore(
    useShallow((state) => {
      const key = mapping[zone];
      const subsystem = key ? (state.aircraft[key] as SubsystemHealth | undefined) : undefined;

      if (!subsystem) return { color: '#10b981', intensity: 0, isWarning: false };

      const colors: Record<string, string> = {
        Healthy: '#10b981',
        Warning: '#f59e0b',
        Critical: '#f43f5e',
      };

      return {
        color: colors[subsystem.status] ?? '#10b981',
        intensity: Math.max(0, (100 - subsystem.score) / 100),
        isWarning: subsystem.status !== 'Healthy',
      };
    })
  );
};
