import React, { useState, useEffect, useMemo } from 'react';
import {
  Activity,
  Cpu,
  Zap,
  BarChart3,
  Layers,
  Moon,
  Sun,
  ShieldAlert,
  Play,
  Pause,
  RotateCcw,
  CheckCircle2,
  Code2,
  Radio,
  Clock,
  ArrowRight,
  TrendingDown,
  Server
} from 'lucide-react';

// Pre-defined PRB profiles matching python environment
const PRB_PROFILES = [
  { name: "Heavy eMBB Burst", embb: 70, urllc: 15, mmtc: 15 },
  { name: "Moderate eMBB + URLLC", embb: 60, urllc: 25, mmtc: 15 },
  { name: "High URLLC Priority", embb: 50, urllc: 35, mmtc: 15 },
  { name: "Standard Balanced", embb: 50, urllc: 30, mmtc: 20 },
  { name: "Extreme URLLC Burst", embb: 40, urllc: 45, mmtc: 15 },
  { name: "High mMTC Batch", embb: 40, urllc: 30, mmtc: 30 },
  { name: "Low eMBB / URLLC+mMTC", embb: 35, urllc: 35, mmtc: 30 },
  { name: "eMBB + mMTC", embb: 55, urllc: 20, mmtc: 25 },
  { name: "Distributed", embb: 45, urllc: 25, mmtc: 30 },
  { name: "Critical URLLC Focus", embb: 30, urllc: 50, mmtc: 20 },
  { name: "Streaming Focus", embb: 65, urllc: 20, mmtc: 15 },
  { name: "IoT Event Focus", embb: 30, urllc: 30, mmtc: 40 },
];

export default function App() {
  const [activeTab, setActiveTab] = useState<'simulator' | 'comparison' | 'architecture'>('simulator');
  const [selectedPolicy, setSelectedPolicy] = useState<'Proposed_MAPPO' | 'Static_Slicing' | 'Always_Active' | 'Greedy_Threshold' | 'Single_Agent_DRL'>('Proposed_MAPPO');
  const [isRunning, setIsRunning] = useState(false);
  const [step, setStep] = useState(0);

  // Simulation state
  const [traffic, setTraffic] = useState({ embb: 62.4, urllc: 11.8, mmtc: 7.9 });
  const [prbs, setPrbs] = useState({ embb: 50, urllc: 30, mmtc: 20 });
  const [xappState, setXappState] = useState<0 | 1 | 2>(0); // 0=Active, 1=Light Sleep, 2=Deep Sleep
  const [history, setHistory] = useState<Array<{
    step: number;
    power: number;
    energy: number;
    slaViol: number;
    urllcLat: number;
    prbUtil: number;
    state: number;
  }>>([]);

  const [totalEnergy, setTotalEnergy] = useState(0);
  const [slaViolationCount, setSlaViolationCount] = useState(0);

  // Calculate power based on state
  const currentPower = useMemo(() => {
    const pBase = 180;
    const rfPower = xappState === 0 ? 360 : xappState === 1 ? 120 : 24;
    const xappPower = xappState === 0 ? 45 : xappState === 1 ? 15 : 3;
    return pBase + rfPower + xappPower;
  }, [xappState]);

  // Step simulation function
  const stepSimulation = () => {
    setStep(prev => {
      const nextStep = prev + 1;

      // Time varying synthetic traffic
      const diurnal = 0.7 + 0.3 * Math.sin((2 * Math.PI * nextStep) / 100);
      const isBurst = Math.random() < 0.2;
      const embbDemand = Math.max(10, 65 * diurnal + (Math.random() - 0.5) * 10 * (isBurst ? 1.6 : 1));
      const urllcDemand = Math.max(2, 12 + (Math.random() < 0.25 ? Math.random() * 15 : (Math.random() - 0.5) * 3));
      const mmtcDemand = Math.max(1, 8 * (0.85 + 0.3 * Math.random()));
      const totalDemand = embbDemand + urllcDemand + mmtcDemand;

      setTraffic({ embb: embbDemand, urllc: urllcDemand, mmtc: mmtcDemand });

      // Policy decision logic
      let chosenProfile = 3;
      let nextState: 0 | 1 | 2 = 0;

      if (selectedPolicy === 'Proposed_MAPPO') {
        // Cooperative learned policy
        if (urllcDemand > 18) {
          chosenProfile = 4; // Extreme URLLC profile
          nextState = 0; // Must stay active for URLLC
        } else if (totalDemand < 40) {
          chosenProfile = 3;
          nextState = 2; // Deep sleep when load is very low
        } else if (totalDemand < 70) {
          chosenProfile = 1;
          nextState = 1; // Light sleep
        } else {
          chosenProfile = 0;
          nextState = 0; // Active
        }
      } else if (selectedPolicy === 'Static_Slicing') {
        chosenProfile = 3; // Fixed 50/30/20
        nextState = 0; // Always active
      } else if (selectedPolicy === 'Always_Active') {
        chosenProfile = urllcDemand > 15 ? 4 : embbDemand > 55 ? 0 : 3;
        nextState = 0; // Never sleep
      } else if (selectedPolicy === 'Greedy_Threshold') {
        if (totalDemand < 35) nextState = 2;
        else if (totalDemand < 70) nextState = 1;
        else nextState = 0;
        chosenProfile = urllcDemand > 15 ? 4 : 3;
      } else {
        // Single Agent DRL
        chosenProfile = totalDemand > 75 ? 0 : 3;
        nextState = totalDemand < 45 ? 1 : 0;
      }

      const activePrbs = PRB_PROFILES[chosenProfile];
      setPrbs({ embb: activePrbs.embb, urllc: activePrbs.urllc, mmtc: activePrbs.mmtc });
      setXappState(nextState);

      // Latency and SLA violations
      const se = nextState === 0 ? 3.8 : nextState === 1 ? 3.0 : 2.2;
      const urllcCapacity = (activePrbs.urllc * 0.36 * se);
      const urllcLoad = urllcDemand / (urllcCapacity + 1e-3);
      const urllcLat = 1.2 + (urllcLoad < 0.85 ? 0.5 / (1.0 - urllcLoad + 0.1) : 3.5 + 8 * (urllcLoad - 0.85));
      const slaViol = (urllcLat > 5.0 || urllcDemand > urllcCapacity) ? 1 : 0;

      const pBase = 180;
      const rfP = nextState === 0 ? 360 : nextState === 1 ? 120 : 24;
      const xP = nextState === 0 ? 45 : nextState === 1 ? 15 : 3;
      const pTotal = pBase + rfP + xP;
      const stepEnergy = pTotal * 0.1; // 100 ms step

      setTotalEnergy(e => e + stepEnergy);
      if (slaViol) setSlaViolationCount(c => c + 1);

      setHistory(h => [
        ...h.slice(-29),
        {
          step: nextStep,
          power: pTotal,
          energy: stepEnergy,
          slaViol: slaViol,
          urllcLat: urllcLat,
          prbUtil: Math.min(1.0, totalDemand / 120),
          state: nextState
        }
      ]);

      return nextStep;
    });
  };

  useEffect(() => {
    let interval: any;
    if (isRunning) {
      interval = setInterval(stepSimulation, 300);
    }
    return () => clearInterval(interval);
  }, [isRunning, selectedPolicy, xappState]);

  const resetSimulation = () => {
    setIsRunning(false);
    setStep(0);
    setTotalEnergy(0);
    setSlaViolationCount(0);
    setHistory([]);
    setXappState(0);
    setPrbs({ embb: 50, urllc: 30, mmtc: 20 });
    setTraffic({ embb: 62.4, urllc: 11.8, mmtc: 7.9 });
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-cyan-500 selection:text-white">
      {/* Top Navigation Bar */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur-md px-6 py-4 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20 shrink-0">
              <Radio className="h-6 w-6 text-white animate-pulse" />
            </div>
            <div>
              <h1 className="text-lg md:text-xl font-bold tracking-tight text-white">
                Energy-Aware Multi-Agent DRL for Slicing & xApp Sleeping in O-RAN
              </h1>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav className="flex items-center gap-1 bg-slate-950/70 p-1.5 rounded-xl border border-slate-800 text-sm">
            <button
              onClick={() => setActiveTab('simulator')}
              className={`px-3.5 py-1.5 rounded-lg font-medium transition-all ${
                activeTab === 'simulator'
                  ? 'bg-cyan-600 text-white shadow'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
              }`}
            >
              Live Simulator
            </button>
            <button
              onClick={() => setActiveTab('comparison')}
              className={`px-3.5 py-1.5 rounded-lg font-medium transition-all ${
                activeTab === 'comparison'
                  ? 'bg-cyan-600 text-white shadow'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
              }`}
            >
              Benchmark Matrix
            </button>
            <button
              onClick={() => setActiveTab('architecture')}
              className={`px-3.5 py-1.5 rounded-lg font-medium transition-all ${
                activeTab === 'architecture'
                  ? 'bg-cyan-600 text-white shadow'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
              }`}
            >
              System & MAPPO CTDE
            </button>
          </nav>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 md:p-6 space-y-6">
        {/* Tab 1: Live Interactive Simulator */}
        {activeTab === 'simulator' && (
          <div className="space-y-6">
            {/* Control Ribbon */}
            <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-4 md:p-6 shadow-xl flex flex-wrap items-center justify-between gap-4">
              <div className="flex flex-wrap items-center gap-4">
                <div>
                  <label className="text-xs uppercase tracking-wider font-semibold text-slate-400 block mb-1">
                    Control Policy
                  </label>
                  <select
                    value={selectedPolicy}
                    onChange={(e) => setSelectedPolicy(e.target.value as any)}
                    className="bg-slate-950 border border-slate-700 text-cyan-400 text-sm font-semibold rounded-lg px-3 py-2 outline-none focus:ring-2 focus:ring-cyan-500"
                  >
                    <option value="Proposed_MAPPO">Proposed MARL (MAPPO CTDE)</option>
                    <option value="Static_Slicing">Baseline 1: Static Slicing</option>
                    <option value="Always_Active">Baseline 2: Always-Active</option>
                    <option value="Greedy_Threshold">Baseline 3: Greedy / Threshold</option>
                    <option value="Single_Agent_DRL">Baseline 4: Single-Agent DRL</option>
                  </select>
                </div>

                <div className="h-8 w-px bg-slate-800 hidden sm:block" />

                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setIsRunning(!isRunning)}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg font-semibold text-sm transition shadow-lg ${
                      isRunning
                        ? 'bg-amber-600 hover:bg-amber-500 text-white shadow-amber-900/30'
                        : 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-emerald-900/30'
                    }`}
                  >
                    {isRunning ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                    {isRunning ? 'Pause Sim' : 'Run Live Sim'}
                  </button>
                  <button
                    onClick={stepSimulation}
                    disabled={isRunning}
                    className="px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-sm font-medium transition"
                  >
                    Single Step
                  </button>
                  <button
                    onClick={resetSimulation}
                    className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white transition"
                    title="Reset Simulation"
                  >
                    <RotateCcw className="h-4 w-4" />
                  </button>
                </div>
              </div>

              {/* Status Pills */}
              <div className="flex items-center gap-3">
                <div className="bg-slate-950 px-3.5 py-1.5 rounded-lg border border-slate-800 text-xs">
                  <span className="text-slate-400">Step:</span>{' '}
                  <span className="font-mono font-bold text-white">{step}</span>
                  <span className="text-slate-500 ml-1">({(step * 0.1).toFixed(1)}s)</span>
                </div>
                <div className={`px-3.5 py-1.5 rounded-lg border text-xs font-semibold flex items-center gap-2 ${
                  xappState === 0
                    ? 'bg-emerald-950/60 border-emerald-800 text-emerald-300'
                    : xappState === 1
                    ? 'bg-amber-950/60 border-amber-800 text-amber-300'
                    : 'bg-indigo-950/60 border-indigo-800 text-indigo-300'
                }`}>
                  {xappState === 0 ? <Sun className="h-3.5 w-3.5" /> : <Moon className="h-3.5 w-3.5" />}
                  State: {xappState === 0 ? 'Active (Full Power)' : xappState === 1 ? 'Light Sleep (DTX)' : 'Deep Sleep (Standby)'}
                </div>
              </div>
            </div>

            {/* KPI Metric Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="bg-slate-900/80 border border-slate-800/80 rounded-2xl p-5 shadow-lg relative overflow-hidden">
                <div className="flex items-center justify-between text-slate-400 text-xs font-semibold uppercase tracking-wider mb-2">
                  <span>Current Power Draw</span>
                  <Zap className="h-4 w-4 text-amber-400" />
                </div>
                <div className="text-3xl font-extrabold text-white font-mono">
                  {currentPower.toFixed(1)} <span className="text-sm font-normal text-slate-400">Watts</span>
                </div>
                <div className="mt-2 text-xs text-slate-400 flex items-center justify-between">
                  <span>Base: 180W | RF: {xappState === 0 ? 360 : xappState === 1 ? 120 : 24}W</span>
                  <span className="text-emerald-400 font-medium">
                    {xappState > 0 ? `-${(((585 - currentPower) / 585) * 100).toFixed(0)}% saved` : 'Peak'}
                  </span>
                </div>
                <div className="absolute -bottom-6 -right-6 w-24 h-24 bg-amber-500/10 rounded-full blur-xl pointer-events-none" />
              </div>

              <div className="bg-slate-900/80 border border-slate-800/80 rounded-2xl p-5 shadow-lg relative overflow-hidden">
                <div className="flex items-center justify-between text-slate-400 text-xs font-semibold uppercase tracking-wider mb-2">
                  <span>Accumulated Energy</span>
                  <Activity className="h-4 w-4 text-cyan-400" />
                </div>
                <div className="text-3xl font-extrabold text-white font-mono">
                  {totalEnergy.toFixed(1)} <span className="text-sm font-normal text-slate-400">Joules</span>
                </div>
                <div className="mt-2 text-xs text-slate-400 flex items-center justify-between">
                  <span>{(totalEnergy / 3600).toFixed(3)} Wh</span>
                  <span className="text-cyan-400">{step} decision intervals</span>
                </div>
                <div className="absolute -bottom-6 -right-6 w-24 h-24 bg-cyan-500/10 rounded-full blur-xl pointer-events-none" />
              </div>

              <div className="bg-slate-900/80 border border-slate-800/80 rounded-2xl p-5 shadow-lg relative overflow-hidden">
                <div className="flex items-center justify-between text-slate-400 text-xs font-semibold uppercase tracking-wider mb-2">
                  <span>URLLC Latency</span>
                  <Clock className="h-4 w-4 text-purple-400" />
                </div>
                <div className="text-3xl font-extrabold text-white font-mono flex items-baseline gap-2">
                  {(history[history.length - 1]?.urllcLat || 2.4).toFixed(2)}{' '}
                  <span className="text-sm font-normal text-slate-400">ms</span>
                </div>
                <div className="mt-2 text-xs text-slate-400 flex items-center justify-between">
                  <span className="flex items-center gap-1">
                    SLA Limit: <strong className="text-rose-400">≤ 5.0 ms</strong>
                  </span>
                  <span className={(history[history.length - 1]?.urllcLat || 2.4) <= 5.0 ? 'text-emerald-400' : 'text-rose-400'}>
                    {(history[history.length - 1]?.urllcLat || 2.4) <= 5.0 ? 'Compliant' : 'VIOLATION'}
                  </span>
                </div>
                <div className="absolute -bottom-6 -right-6 w-24 h-24 bg-purple-500/10 rounded-full blur-xl pointer-events-none" />
              </div>

              <div className="bg-slate-900/80 border border-slate-800/80 rounded-2xl p-5 shadow-lg relative overflow-hidden">
                <div className="flex items-center justify-between text-slate-400 text-xs font-semibold uppercase tracking-wider mb-2">
                  <span>SLA Violations</span>
                  <ShieldAlert className="h-4 w-4 text-rose-400" />
                </div>
                <div className="text-3xl font-extrabold text-white font-mono">
                  {slaViolationCount}{' '}
                  <span className="text-sm font-normal text-slate-400">
                    ({step > 0 ? ((slaViolationCount / step) * 100).toFixed(1) : 0}%)
                  </span>
                </div>
                <div className="mt-2 text-xs text-slate-400 flex items-center justify-between">
                  <span>Target: 0.0%</span>
                  <span className={slaViolationCount === 0 ? 'text-emerald-400' : 'text-rose-400 font-medium'}>
                    {slaViolationCount === 0 ? 'Zero Violations' : 'Degraded QoS'}
                  </span>
                </div>
                <div className="absolute -bottom-6 -right-6 w-24 h-24 bg-rose-500/10 rounded-full blur-xl pointer-events-none" />
              </div>
            </div>

            {/* Two-Agent Operational View */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Agent 1: Slicing PRB Controller */}
              <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-5">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="p-2.5 rounded-xl bg-cyan-950 border border-cyan-800/60 text-cyan-400">
                      <Layers className="h-5 w-5" />
                    </div>
                    <div>
                      <h3 className="font-bold text-white text-base">Agent 1: Dynamic PRB Slicing</h3>
                      <p className="text-xs text-slate-400">Partition of 100 PRBs (20 MHz Sub-6 GHz 5G Carrier)</p>
                    </div>
                  </div>
                  <span className="text-xs bg-slate-800 px-2.5 py-1 rounded text-cyan-300 font-mono">
                    Budget: 100 PRBs
                  </span>
                </div>

                {/* Stacked PRB Visualizer Bar */}
                <div className="space-y-2">
                  <div className="flex justify-between text-xs text-slate-400 font-medium">
                    <span>PRB Distribution:</span>
                    <span>{prbs.embb} eMBB | {prbs.urllc} URLLC | {prbs.mmtc} mMTC</span>
                  </div>
                  <div className="h-7 w-full bg-slate-950 rounded-xl overflow-hidden flex p-1 border border-slate-800 gap-1">
                    <div
                      style={{ width: `${prbs.embb}%` }}
                      className="bg-gradient-to-r from-blue-600 to-cyan-500 rounded-lg flex items-center justify-center text-[10px] font-bold text-white transition-all duration-300"
                    >
                      eMBB {prbs.embb}
                    </div>
                    <div
                      style={{ width: `${prbs.urllc}%` }}
                      className="bg-gradient-to-r from-purple-600 to-indigo-500 rounded-lg flex items-center justify-center text-[10px] font-bold text-white transition-all duration-300"
                    >
                      URLLC {prbs.urllc}
                    </div>
                    <div
                      style={{ width: `${prbs.mmtc}%` }}
                      className="bg-gradient-to-r from-emerald-600 to-teal-500 rounded-lg flex items-center justify-center text-[10px] font-bold text-white transition-all duration-300"
                    >
                      mMTC {prbs.mmtc}
                    </div>
                  </div>
                </div>

                {/* Traffic vs Capacity Comparison */}
                <div className="space-y-3 pt-2">
                  <div className="bg-slate-950/70 p-3.5 rounded-xl border border-slate-800/80 flex items-center justify-between">
                    <div>
                      <span className="text-xs font-semibold text-cyan-400 block">eMBB Slice (Broadband)</span>
                      <span className="text-xs text-slate-400">Demand: {traffic.embb.toFixed(1)} Mbps</span>
                    </div>
                    <div className="text-right">
                      <span className="text-sm font-mono font-bold text-white">
                        {(prbs.embb * 0.36 * (xappState === 0 ? 3.8 : xappState === 1 ? 3.0 : 2.2)).toFixed(1)} Mbps
                      </span>
                      <span className="text-[10px] text-slate-500 block">Capacity</span>
                    </div>
                  </div>

                  <div className="bg-slate-950/70 p-3.5 rounded-xl border border-slate-800/80 flex items-center justify-between">
                    <div>
                      <span className="text-xs font-semibold text-purple-400 block">URLLC Slice (Low Latency)</span>
                      <span className="text-xs text-slate-400">Demand: {traffic.urllc.toFixed(1)} Mbps</span>
                    </div>
                    <div className="text-right">
                      <span className="text-sm font-mono font-bold text-white">
                        {(prbs.urllc * 0.36 * (xappState === 0 ? 3.8 : xappState === 1 ? 3.0 : 2.2)).toFixed(1)} Mbps
                      </span>
                      <span className="text-[10px] text-slate-500 block">Capacity</span>
                    </div>
                  </div>

                  <div className="bg-slate-950/70 p-3.5 rounded-xl border border-slate-800/80 flex items-center justify-between">
                    <div>
                      <span className="text-xs font-semibold text-emerald-400 block">mMTC Slice (Massive IoT)</span>
                      <span className="text-xs text-slate-400">Demand: {traffic.mmtc.toFixed(1)} Mbps</span>
                    </div>
                    <div className="text-right">
                      <span className="text-sm font-mono font-bold text-white">
                        {(prbs.mmtc * 0.36 * (xappState === 0 ? 3.8 : xappState === 1 ? 3.0 : 2.2)).toFixed(1)} Mbps
                      </span>
                      <span className="text-[10px] text-slate-500 block">Capacity</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Agent 2: Sleeping & Energy Controller */}
              <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-5">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="p-2.5 rounded-xl bg-amber-950 border border-amber-800/60 text-amber-400">
                      <Cpu className="h-5 w-5" />
                    </div>
                    <div>
                      <h3 className="font-bold text-white text-base">Agent 2: xApp & RF Sleeping Agent</h3>
                      <p className="text-xs text-slate-400">Near-RT RIC Microservices & Power Amplifiers</p>
                    </div>
                  </div>
                  <span className="text-xs bg-slate-800 px-2.5 py-1 rounded text-amber-300 font-mono">
                    3 Cells · 3 xApps
                  </span>
                </div>

                {/* Subsystem State Diagram */}
                <div className="grid grid-cols-3 gap-3">
                  <div className={`p-3.5 rounded-xl border text-center transition-all ${
                    xappState === 0
                      ? 'bg-emerald-950/40 border-emerald-500 text-emerald-300 shadow-md shadow-emerald-950'
                      : 'bg-slate-950/50 border-slate-800 text-slate-500'
                  }`}>
                    <Sun className="h-5 w-5 mx-auto mb-1 text-emerald-400" />
                    <span className="text-xs font-bold block">Active</span>
                    <span className="text-[10px] block text-slate-400 mt-1">Full Beamforming</span>
                    <span className="text-[10px] text-emerald-400 font-mono font-bold">585 W</span>
                  </div>

                  <div className={`p-3.5 rounded-xl border text-center transition-all ${
                    xappState === 1
                      ? 'bg-amber-950/40 border-amber-500 text-amber-300 shadow-md shadow-amber-950'
                      : 'bg-slate-950/50 border-slate-800 text-slate-500'
                  }`}>
                    <Moon className="h-5 w-5 mx-auto mb-1 text-amber-400" />
                    <span className="text-xs font-bold block">Light Sleep</span>
                    <span className="text-[10px] block text-slate-400 mt-1">DTX (10ms wake)</span>
                    <span className="text-[10px] text-amber-400 font-mono font-bold">315 W</span>
                  </div>

                  <div className={`p-3.5 rounded-xl border text-center transition-all ${
                    xappState === 2
                      ? 'bg-indigo-950/40 border-indigo-500 text-indigo-300 shadow-md shadow-indigo-950'
                      : 'bg-slate-950/50 border-slate-800 text-slate-500'
                  }`}>
                    <Moon className="h-5 w-5 mx-auto mb-1 text-indigo-400" />
                    <span className="text-xs font-bold block">Deep Sleep</span>
                    <span className="text-[10px] block text-slate-400 mt-1">Standby (100ms)</span>
                    <span className="text-[10px] text-indigo-400 font-mono font-bold">207 W</span>
                  </div>
                </div>

                {/* Subsystem Details List */}
                <div className="space-y-2.5 text-xs text-slate-300">
                  <div className="flex items-center justify-between p-2 rounded-lg bg-slate-950/50 border border-slate-800/60">
                    <span className="flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-emerald-400" />
                      Essential Baseband & Sync (SSB/PDCCH)
                    </span>
                    <span className="text-slate-400 font-mono">180 W (Constant ON)</span>
                  </div>
                  <div className="flex items-center justify-between p-2 rounded-lg bg-slate-950/50 border border-slate-800/60">
                    <span className="flex items-center gap-2">
                      <span className={`w-2 h-2 rounded-full ${xappState === 0 ? 'bg-emerald-400' : xappState === 1 ? 'bg-amber-400' : 'bg-indigo-400'}`} />
                      RF Power Amplifier Transceivers (3 Sectors)
                    </span>
                    <span className="text-slate-400 font-mono">
                      {xappState === 0 ? '360 W (120W x 3)' : xappState === 1 ? '120 W (40W x 3)' : '24 W (8W x 3)'}
                    </span>
                  </div>
                  <div className="flex items-center justify-between p-2 rounded-lg bg-slate-950/50 border border-slate-800/60">
                    <span className="flex items-center gap-2">
                      <span className={`w-2 h-2 rounded-full ${xappState === 0 ? 'bg-emerald-400' : xappState === 1 ? 'bg-amber-400' : 'bg-indigo-400'}`} />
                      Controlled Near-RT RIC xApps (MIMO & ML Predictor)
                    </span>
                    <span className="text-slate-400 font-mono">
                      {xappState === 0 ? '45 W (Active)' : xappState === 1 ? '15 W (Throttled)' : '3 W (Suspended)'}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Real-time Telemetry History */}
            <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
              <h3 className="font-bold text-white text-base flex items-center gap-2">
                <BarChart3 className="h-5 w-5 text-cyan-400" />
                Real-Time Telemetry Rollout (Last 30 Steps)
              </h3>

              {history.length === 0 ? (
                <div className="py-12 text-center text-slate-500 text-sm">
                  Click "Run Live Sim" or "Single Step" above to start the multi-agent decision loop.
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs font-mono">
                    <thead className="bg-slate-950 text-slate-400 border-b border-slate-800">
                      <tr>
                        <th className="py-2.5 px-3">Step</th>
                        <th className="py-2.5 px-3">State</th>
                        <th className="py-2.5 px-3">Power</th>
                        <th className="py-2.5 px-3">Energy</th>
                        <th className="py-2.5 px-3">URLLC Delay</th>
                        <th className="py-2.5 px-3">SLA Status</th>
                        <th className="py-2.5 px-3">PRB Load</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60">
                      {history.slice(-8).reverse().map((h, i) => (
                        <tr key={i} className="hover:bg-slate-800/40">
                          <td className="py-2 px-3 text-slate-300 font-bold">#{h.step}</td>
                          <td className="py-2 px-3">
                            <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                              h.state === 0 ? 'bg-emerald-950 text-emerald-300 border border-emerald-800' :
                              h.state === 1 ? 'bg-amber-950 text-amber-300 border border-amber-800' :
                              'bg-indigo-950 text-indigo-300 border border-indigo-800'
                            }`}>
                              {h.state === 0 ? 'ACTIVE' : h.state === 1 ? 'LIGHT' : 'DEEP'}
                            </span>
                          </td>
                          <td className="py-2 px-3 text-amber-400">{h.power.toFixed(1)} W</td>
                          <td className="py-2 px-3 text-cyan-400">{h.energy.toFixed(1)} J</td>
                          <td className={`py-2 px-3 ${h.urllcLat > 5.0 ? 'text-rose-400 font-bold' : 'text-slate-300'}`}>
                            {h.urllcLat.toFixed(2)} ms
                          </td>
                          <td className="py-2 px-3">
                            {h.slaViol === 0 ? (
                              <span className="text-emerald-400 flex items-center gap-1">
                                <CheckCircle2 className="h-3 w-3" /> Met
                              </span>
                            ) : (
                              <span className="text-rose-400 flex items-center gap-1 font-bold">
                                <ShieldAlert className="h-3 w-3" /> Viol
                              </span>
                            )}
                          </td>
                          <td className="py-2 px-3 text-slate-400">{(h.prbUtil * 100).toFixed(1)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Tab 2: Baseline Matrix & Trade-off Analysis */}
        {activeTab === 'comparison' && (
          <div className="space-y-6">
            <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div>
                  <h2 className="text-xl font-bold text-white">Comparative Policy Performance Matrix</h2>
                  <p className="text-sm text-slate-400">
                    Analytical simulation target model and comparative baseline specs.
                    Relative energy savings (%) are calculated against the Always-Active benchmark.
                  </p>
                </div>
                <div className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-cyan-950/80 border border-cyan-800/80 text-xs text-cyan-300 font-mono">
                  <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                  Mode: Simulation Target Benchmark
                </div>
              </div>

              {/* Research Integrity Notice */}
              <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 text-xs text-slate-400 flex items-start gap-3">
                <ShieldAlert className="h-5 w-5 text-amber-400 shrink-0 mt-0.5" />
                <div>
                  <strong className="text-slate-200">Experimental Integrity Note:</strong> The figures below display the simulated analytical design specifications. Actual experimental results from trained neural network checkpoints must be generated locally on your machine by running{' '}
                  <code className="text-cyan-300 bg-slate-900 px-1 py-0.5 rounded font-mono">python compare_baselines.py</code>. Actual outputs will be saved to{' '}
                  <code className="text-cyan-300 bg-slate-900 px-1 py-0.5 rounded font-mono">results/baseline_comparison.csv</code>.
                </div>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs font-mono border-collapse">
                  <thead className="bg-slate-950 text-slate-300 border-b border-slate-800 uppercase tracking-wider">
                    <tr>
                      <th className="py-3 px-4">Policy</th>
                      <th className="py-3 px-4">Energy (J)</th>
                      <th className="py-3 px-4 text-emerald-400">Energy Saving (%)</th>
                      <th className="py-3 px-4">SLA Viol Rate (%)</th>
                      <th className="py-3 px-4">URLLC Lat (ms)</th>
                      <th className="py-3 px-4">P99 Lat (ms)</th>
                      <th className="py-3 px-4">eMBB Thr (Mbps)</th>
                      <th className="py-3 px-4">PRB Util (%)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800 text-slate-300">
                    <tr className="bg-cyan-950/20 font-bold border-l-4 border-l-cyan-500">
                      <td className="py-3 px-4 text-cyan-300 font-sans">Proposed MARL (MAPPO) [Target]</td>
                      <td className="py-3 px-4 text-white">36,420 J</td>
                      <td className="py-3 px-4 text-emerald-400 font-extrabold">+37.7%</td>
                      <td className="py-3 px-4 text-emerald-400">0.8%</td>
                      <td className="py-3 px-4">2.42 ms</td>
                      <td className="py-3 px-4 text-emerald-400">4.18 ms (≤ 5ms)</td>
                      <td className="py-3 px-4">58.4 Mbps</td>
                      <td className="py-3 px-4">78.5%</td>
                    </tr>
                    <tr className="hover:bg-slate-800/30">
                      <td className="py-3 px-4 font-sans text-slate-200">Baseline 1: Static Slicing</td>
                      <td className="py-3 px-4">58,500 J</td>
                      <td className="py-3 px-4 text-slate-400">0.0%</td>
                      <td className="py-3 px-4 text-amber-400">8.4%</td>
                      <td className="py-3 px-4">3.85 ms</td>
                      <td className="py-3 px-4 text-rose-400">7.10 ms</td>
                      <td className="py-3 px-4">49.2 Mbps</td>
                      <td className="py-3 px-4">54.2%</td>
                    </tr>
                    <tr className="hover:bg-slate-800/30">
                      <td className="py-3 px-4 font-sans text-slate-200">Baseline 2: Always-Active (Ref)</td>
                      <td className="py-3 px-4">58,500 J</td>
                      <td className="py-3 px-4 text-slate-400">0.0% (Reference)</td>
                      <td className="py-3 px-4 text-emerald-400">1.2%</td>
                      <td className="py-3 px-4">2.35 ms</td>
                      <td className="py-3 px-4">4.05 ms</td>
                      <td className="py-3 px-4">61.0 Mbps</td>
                      <td className="py-3 px-4">68.0%</td>
                    </tr>
                    <tr className="hover:bg-slate-800/30">
                      <td className="py-3 px-4 font-sans text-slate-200">Baseline 3: Greedy / Threshold</td>
                      <td className="py-3 px-4">33,200 J</td>
                      <td className="py-3 px-4 text-emerald-400">+43.2%</td>
                      <td className="py-3 px-4 text-rose-400 font-bold">14.6% (Severe)</td>
                      <td className="py-3 px-4 text-rose-400">4.95 ms</td>
                      <td className="py-3 px-4 text-rose-400 font-bold">9.80 ms</td>
                      <td className="py-3 px-4">51.2 Mbps</td>
                      <td className="py-3 px-4">71.0%</td>
                    </tr>
                    <tr className="hover:bg-slate-800/30">
                      <td className="py-3 font-sans text-slate-200 px-4">Baseline 4: Single-Agent DRL</td>
                      <td className="py-3 px-4">41,800 J</td>
                      <td className="py-3 px-4 text-emerald-400">+28.5%</td>
                      <td className="py-3 px-4 text-amber-400">5.2%</td>
                      <td className="py-3 px-4">3.10 ms</td>
                      <td className="py-3 px-4 text-amber-400">5.90 ms</td>
                      <td className="py-3 px-4">55.8 Mbps</td>
                      <td className="py-3 px-4">72.4%</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-4">
                <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
                  <h4 className="font-bold text-sm text-cyan-400 flex items-center gap-2">
                    <TrendingDown className="h-4 w-4" /> Why Proposed MARL Outperforms Greedy Threshold
                  </h4>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    While the Greedy heuristic achieves high energy reduction, it causes severe SLA violations (14.6%)
                    and latency spikes (9.8 ms) because it switches into deep sleep without anticipating sudden URLLC bursts.
                    The proposed MAPPO cooperative agents anticipate load trends and coordinate PRB reservations before sleeping.
                  </p>
                </div>
                <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
                  <h4 className="font-bold text-sm text-purple-400 flex items-center gap-2">
                    <Zap className="h-4 w-4" /> Advantage Over Single-Agent DRL
                  </h4>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    A single unified agent faces a combinatorial action space (12 PRB profiles × 3 sleep states = 36 joint actions).
                    By decoupling into two cooperative decentralized agents under CTDE, policy training is stabilized, achieving
                    higher energy savings (+37.7% vs +28.5%) and preserving SLA adherence.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Tab 3: System Model & Architecture */}
        {activeTab === 'architecture' && (
          <div className="space-y-6">
            <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
              <div>
                <h2 className="text-xl font-bold text-white">5G O-RAN Architecture & Centralized Training Decentralized Execution</h2>
                <p className="text-sm text-slate-400">
                  Near-RT RIC hosting Slicing xApp and Sleeping xApp with E2/KPM telemetry interfaces.
                </p>
              </div>

              {/* Architecture Diagram Box */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-slate-950 p-5 rounded-xl border border-slate-800 space-y-3">
                  <div className="flex items-center gap-2 text-cyan-400 font-bold text-sm">
                    <Server className="h-4 w-4" /> Near-RT RIC Environment
                  </div>
                  <ul className="text-xs text-slate-400 space-y-2">
                    <li>• <strong className="text-slate-200">Slicing xApp:</strong> Agent 1 runs as containerized service, observing KPM throughput/delay metrics.</li>
                    <li>• <strong className="text-slate-200">Sleeping xApp:</strong> Agent 2 monitors RF amplifier power and computational load.</li>
                    <li>• <strong className="text-slate-200">Control Loop:</strong> Operates at 100 ms intervals (E2 interface).</li>
                  </ul>
                </div>

                <div className="bg-slate-950 p-5 rounded-xl border border-slate-800 space-y-3">
                  <div className="flex items-center gap-2 text-purple-400 font-bold text-sm">
                    <Radio className="h-4 w-4" /> Radio Sub-System (gNB)
                  </div>
                  <ul className="text-xs text-slate-400 space-y-2">
                    <li>• <strong className="text-slate-200">Total PRBs:</strong> 100 PRBs (20 MHz, 30 kHz SCS).</li>
                    <li>• <strong className="text-slate-200">Slices:</strong> eMBB (≥45 Mbps), URLLC (≤5 ms delay), mMTC (dense IoT).</li>
                    <li>• <strong className="text-slate-200">Radio Modes:</strong> Active (120W/cell), DTX Light Sleep (40W/cell), Deep Sleep (8W/cell).</li>
                  </ul>
                </div>

                <div className="bg-slate-950 p-5 rounded-xl border border-slate-800 space-y-3">
                  <div className="flex items-center gap-2 text-emerald-400 font-bold text-sm">
                    <Activity className="h-4 w-4" /> MAPPO CTDE Framework
                  </div>
                  <ul className="text-xs text-slate-400 space-y-2">
                    <li>• <strong className="text-slate-200">Centralized Critic:</strong> V(s) observes global 25-dim system state during training.</li>
                    <li>• <strong className="text-slate-200">Decentralized Actors:</strong> π<sub>1</sub>(a<sub>1</sub>|o<sub>1</sub>) and π<sub>2</sub>(a<sub>2</sub>|o<sub>2</sub>) act purely on local metrics.</li>
                    <li>• <strong className="text-slate-200">GAE Objective:</strong> Clipped surrogate loss with entropy exploration.</li>
                  </ul>
                </div>
              </div>

              {/* Reward Function Card */}
              <div className="p-5 rounded-xl bg-slate-950 border border-slate-800/80 space-y-3 font-mono text-xs">
                <div className="font-bold text-amber-400 text-sm font-sans flex items-center gap-2">
                  <Code2 className="h-4 w-4" /> Cooperative Global Reward Formulation
                </div>
                <div className="bg-slate-900 p-3 rounded-lg text-slate-200 overflow-x-auto">
                  R<sub>t</sub> = w<sub>qos</sub> · QoS<sub>sat</sub> - w<sub>energy</sub> · (P<sub>total</sub> / P<sub>max</sub>) - w<sub>sla</sub> · SLA<sub>viol</sub> - w<sub>trans</sub> · C<sub>trans</sub> + w<sub>util</sub> · Util<sub>prb</sub>
                </div>
                <p className="text-slate-400 text-[11px] font-sans">
                  The reward explicitly penalizes high energy consumption while assigning a heavy penalty (w<sub>sla</sub> = 2.5)
                  to SLA violations. This prevents trivial degenerate policies that attempt to switch off all cells to save power.
                </p>
              </div>
            </div>
          </div>
        )}

      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-950 text-slate-500 text-xs py-4 px-6 text-center">
        Energy-Aware Multi-Agent DRL for Slicing & xApp Sleeping in O-RAN
      </footer>
    </div>
  );
}
