import { useState, useEffect, useRef } from 'react';
import WaveSurfer from 'wavesurfer.js';
import RegionsPlugin from 'wavesurfer.js/dist/plugins/regions.esm.js';

const THEMES = {
  minimalLight: {
    id: "minimalLight",
    name: "Web 1.0 Light",
    description: "Classic Web 1.0 style.",
    bgApp: "bg-white text-black font-serif",
    bgMain: "bg-white",
    sidebar: "hidden", 
    topbar: "border-b-2 border-black bg-gray-100 p-2 mb-4 text-center relative",
    topbarTitle: "text-2xl font-bold text-black mb-2",
    topbarTabActive: "bg-white border-t border-l border-r border-black font-bold px-4 py-1 mx-1 inline-block",
    topbarTabInactive: "bg-gray-200 border border-gray-400 text-blue-600 underline px-4 py-1 mx-1 hover:bg-gray-300 inline-block",
    viewTitle: "text-2xl font-bold border-b border-gray-400 mb-4 pb-2",
    viewDesc: "text-sm mb-4",
    card: "border border-black p-4 mb-4 bg-white mx-auto",
    label: "font-bold text-sm block mb-1",
    input: "border border-black p-1 bg-white text-black mb-2 w-full",
    inputInfo: "border border-gray-400 bg-gray-100 p-2 text-sm mb-2",
    inputInfoSub: "text-gray-600 italic",
    inputInfoHighlight: "font-bold",
    checkbox: "mr-2",
    checkboxText: "text-sm",
    checkboxContainer: "mb-4",
    buttonPrimary: "border border-black bg-gray-200 text-black px-4 py-1 hover:bg-gray-300 cursor-pointer active:bg-gray-400 font-bold",
    buttonSecondary: "border border-black bg-gray-200 text-black px-4 py-1 hover:bg-gray-300 cursor-pointer active:bg-gray-400",
    successMsg: "border border-green-600 bg-green-100 text-green-800 p-2 font-bold mb-2",
    errorMsg: "border border-red-600 bg-red-100 text-red-800 p-2 font-bold mb-2",
    toggleIcon: "🌙"
  },
  minimalDark: {
    id: "minimalDark",
    name: "Web 1.0 Dark",
    description: "Classic Web 1.0 style Dark.",
    bgApp: "bg-black text-white font-serif",
    bgMain: "bg-black",
    sidebar: "hidden", 
    topbar: "border-b-2 border-white bg-gray-800 p-2 mb-4 text-center relative",
    topbarTitle: "text-2xl font-bold text-white mb-2",
    topbarTabActive: "bg-black border-t border-l border-r border-white font-bold px-4 py-1 mx-1 inline-block",
    topbarTabInactive: "bg-gray-700 border border-gray-500 text-blue-300 underline px-4 py-1 mx-1 hover:bg-gray-600 inline-block",
    viewTitle: "text-2xl font-bold border-b border-gray-500 mb-4 pb-2",
    viewDesc: "text-sm mb-4",
    card: "border border-white p-4 mb-4 bg-black mx-auto",
    label: "font-bold text-sm block mb-1",
    input: "border border-white p-1 bg-black text-white mb-2 w-full",
    inputInfo: "border border-gray-500 bg-gray-800 p-2 text-sm mb-2",
    inputInfoSub: "text-gray-400 italic",
    inputInfoHighlight: "font-bold",
    checkbox: "mr-2",
    checkboxText: "text-sm",
    checkboxContainer: "mb-4",
    buttonPrimary: "border border-white bg-gray-700 text-white px-4 py-1 hover:bg-gray-600 cursor-pointer active:bg-gray-500 font-bold",
    buttonSecondary: "border border-white bg-gray-700 text-white px-4 py-1 hover:bg-gray-600 cursor-pointer active:bg-gray-500",
    successMsg: "border border-green-400 bg-green-900 text-green-200 p-2 font-bold mb-2",
    errorMsg: "border border-red-400 bg-red-900 text-red-200 p-2 font-bold mb-2",
    toggleIcon: "☀️"
  }
};

function View0({ theme }) {
 const t = theme;
 const [files, setFiles] = useState([]);
 const [form, setForm] = useState({ audio_file: "", filename: "" });
 const [status, setStatus] = useState(null);
 const [loading, setLoading] = useState(false);
 const [ws, setWs] = useState(null);
 const [wsRegions, setWsRegions] = useState(null);
 const [zoom, setZoom] = useState(10);
 const containerRef = useRef(null);

 useEffect(() => {
 fetch("http://localhost:8000/api/inference_files")
 .then(res => res.json())
 .then(data => {
 setFiles(data.wav_files);
 if (data.wav_files.length > 0) {
 setForm(prev => ({ 
 ...prev, 
 audio_file: data.wav_files[0],
 filename: data.wav_files[0].split('.')[0] + "-annotated.txt"
 }));
 }
 }).catch(err => console.log("Failed to fetch audio files", err));
 }, []);

 useEffect(() => {
 if (!containerRef.current || !form.audio_file) return;

 if (ws) {
 ws.destroy();
 }

 const regions = RegionsPlugin.create();
 
 const wavesurfer = WaveSurfer.create({
 container: containerRef.current,
 waveColor: 'rgb(200, 0, 200)',
 progressColor: 'rgb(100, 0, 100)',
 url: `http://localhost:8000/api/audio/${form.audio_file}`,
 plugins: [regions],
 height: 128,
 normalize: false,
 minPxPerSec: Number(zoom)
 });

 regions.enableDragSelection({ color: 'rgba(255, 0, 0, 0.1)' });

 regions.on('region-updated', (region) => {
   if (region.end - region.start < 0.1) {
     region.remove();
   }
 });

 setWs(wavesurfer);
 setWsRegions(regions);

 return () => {
 wavesurfer.destroy();
 };
 // eslint-disable-next-line react-hooks/exhaustive-deps
 }, [form.audio_file]);

 useEffect(() => {
 if (ws) {
 try {
 ws.zoom(Number(zoom));
 } catch (e) {
 // Audio not yet loaded, safe to ignore
 }
 }
 }, [zoom, ws]);

 const handleAutoDetect = async () => {
 setLoading(true);
 setStatus("Detecting speech regions... (This may take a minute)");
 try {
 const res = await fetch("http://localhost:8000/api/vad_segments", {
 method: "POST",
 headers: { "Content-Type": "application/json" },
 body: JSON.stringify({ filename: form.audio_file })
 });
 const data = await res.json();
 if (res.ok) {
 wsRegions.clearRegions();
 data.segments.forEach(seg => {
 wsRegions.addRegion({
 start: seg.start,
 end: seg.end,
 color: 'rgba(0, 255, 0, 0.1)'
 });
 });
 setStatus(`✅ Detected ${data.segments.length} regions.`);
 } else {
 setStatus(`❌ Error: ${data.detail}`);
 }
 } catch (e) {
 setStatus(`❌ Error: ${e.message}`);
 }
 setLoading(false);
 };

 const handleExport = async () => {
 if (!wsRegions) return;
 const regions = wsRegions.getRegions().map(r => ({
 start: r.start,
 end: r.end
 })).sort((a, b) => a.start - b.start);
 
 if (regions.length === 0) {
 setStatus("❌ Error: No regions to export");
 return;
 }

 setLoading(true);
 try {
 const res = await fetch("http://localhost:8000/api/save_elan", {
 method: "POST",
 headers: { "Content-Type": "application/json" },
 body: JSON.stringify({ regions, filename: form.filename })
 });
 const data = await res.json();
 if (res.ok) {
 setStatus(`✅ Exported successfully to ${data.filepath}`);
 } else {
 setStatus(`❌ Error: ${data.detail}`);
 }
 } catch (e) {
 setStatus(`❌ Error: ${e.message}`);
 }
 setLoading(false);
 };

 return (
 <div className="max-w-3xl mx-auto text-center ">
 <h2 className={`text-3xl font-bold mb-6 ${t.viewTitle}`}>0. Auto-segment Long Audio</h2>
 <p className={`${t.viewDesc} mb-8`}>Select a long audio file, run SpeechBrain VAD to find sentence boundaries, and visually adjust them before exporting.</p>
 
 <div className={`${t.card} p-6 `}>
 <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
 <div>
 <label className={`block text-sm font-medium ${t.label} mb-2`}>Audio File</label>
 <select 
 className={`w-full p-3 ${t.input}`} 
 value={form.audio_file} 
 onChange={e => {
 setForm({...form, audio_file: e.target.value, filename: e.target.value.split('.')[0] + "-annotated.txt"});
 setStatus(null);
 }}
 >
 {files.length === 0 && <option>No files found</option>}
 {files.map(f => <option key={f} value={f}>{f}</option>)}
 </select>
 </div>
 <div>
 <label className={`block text-sm font-medium ${t.label} mb-2`}>Export Filename</label>
 <input type="text" className={`w-full p-3 ${t.input}`} value={form.filename} onChange={e => setForm({...form, filename: e.target.value})} />
 </div>
 </div>

 <div className="mb-4">
 <div ref={containerRef} className={`w-full border ${t.inputInfo}`} />
 <div className={`flex flex-col sm:flex-row gap-4 items-center justify-between mt-3 p-3 border ${t.inputInfo}`}>
 <div className="flex items-center gap-3 w-full sm:w-1/2">
 <label className={`text-sm font-medium ${t.label} whitespace-nowrap`}>Zoom:</label>
 <input 
 type="range" 
 min="10" 
 max="1000" 
 value={zoom} 
 onChange={(e) => setZoom(e.target.value)} 
 className="w-full"
 />
 </div>
 <div className="flex gap-2">
 <button onClick={() => {
 if (ws) {
 const wrapper = ws.getWrapper();
 wrapper.scrollBy({ left: -200, behavior: 'smooth' });
 }
 }} className={`px-3 py-1.5 text-sm ${t.buttonSecondary}`}>
 ← Pan
 </button>
 <button onClick={() => ws && ws.playPause()} className={`px-4 py-1.5 text-sm font-bold ${t.buttonSecondary}`}>
 Play / Pause
 </button>
 <button onClick={() => {
 if (ws) {
 const wrapper = ws.getWrapper();
 wrapper.scrollBy({ left: 200, behavior: 'smooth' });
 }
 }} className={`px-3 py-1.5 text-sm ${t.buttonSecondary}`}>
 Pan →
 </button>
 </div>
 </div>
 </div>

 <div className="flex gap-4 mb-6">
 <button 
 onClick={handleAutoDetect} 
 disabled={loading || !form.audio_file}
 className={`flex-1 py-3 disabled:opacity-50 disabled:cursor-not-allowed active:scale-[0.99] ${t.buttonSecondary}`}
 >
 {loading ? "Processing..." : "Auto-Detect Regions"}
 </button>
 <button 
 onClick={handleExport} 
 disabled={loading || !form.filename}
 className={`flex-1 py-3 disabled:opacity-50 disabled:cursor-not-allowed active:scale-[0.99] ${t.buttonPrimary}`}
 >
 Export to ELAN TXT
 </button>
 </div>

 {status && (
 <div className={`mt-6 p-4 border ${status.includes('❌') ? t.errorMsg : (status.includes('✅') ? t.successMsg : t.inputInfo)}`}>
 {status}
 </div>
 )}
 </div>
 </div>
 );
}

function View1({ theme }) {
 const t = theme;
 const [files, setFiles] = useState({ txt_files: [], wav_files: [] });
 const [txtFile, setTxtFile] = useState("");
 const [wavFile, setWavFile] = useState("");
 const [gender, setGender] = useState("m");
 const [status, setStatus] = useState(null);
 const [loading, setLoading] = useState(false);

 useEffect(() => {
 fetch("http://localhost:8000/api/files")
 .then(res => res.json())
 .then(data => {
 setFiles(data);
 if (data.txt_files.length > 0) setTxtFile(data.txt_files[0]);
 if (data.wav_files.length > 0) setWavFile(data.wav_files[0]);
 }).catch(err => console.log("Failed to fetch files", err));
 }, []);

 const handleProcess = async () => {
 setLoading(true);
 setStatus("Processing... This may take a moment.");
 try {
 const res = await fetch("http://localhost:8000/api/process_elan", {
 method: "POST",
 headers: { "Content-Type": "application/json" },
 body: JSON.stringify({ txt_file: txtFile, wav_file: wavFile, gender })
 });
 const data = await res.json();
 if (res.ok) {
 setStatus(`✅ Success! Added ${data.rows_added} rows to metadata.`);
 } else {
 setStatus(`❌ Error: ${data.detail}`);
 }
 } catch (e) {
 setStatus(`❌ Error: ${e.message}`);
 }
 setLoading(false);
 };

 const getCode = (f) => f ? f.split('-').pop().split('.')[0].toUpperCase() : 'UNKNOWN';
 const getPrefix = (f) => f ? f.split('.')[0] : 'UNKNOWN';

 return (
 <div className="max-w-3xl mx-auto text-center ">
 <h2 className={`text-3xl font-bold mb-6 ${t.viewTitle}`}>1. ELAN to WAV and CSV</h2>
 <p className={`${t.viewDesc} mb-8`}>Process your ELAN annotation files and automatically split your audio.</p>
 
 <div className={`${t.card} p-6 `}>
 <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
 <div>
 <label className={`block text-sm font-medium ${t.label} mb-2`}>Annotation File (.txt)</label>
 <select className={`w-full p-3 ${t.input}`} value={txtFile} onChange={e => setTxtFile(e.target.value)}>
 {files.txt_files.length === 0 && <option>No files found</option>}
 {files.txt_files.map(f => <option key={f} value={f}>{f}</option>)}
 </select>
 </div>
 <div>
 <label className={`block text-sm font-medium ${t.label} mb-2`}>Source Audio (.wav)</label>
 <select className={`w-full p-3 ${t.input}`} value={wavFile} onChange={e => setWavFile(e.target.value)}>
 {files.wav_files.length === 0 && <option>No files found</option>}
 {files.wav_files.map(f => <option key={f} value={f}>{f}</option>)}
 </select>
 </div>
 </div>

 <div className="mb-8">
 <label className={`block text-sm font-medium ${t.label} mb-2`}>Speaker Gender</label>
 <select className={`w-full md:w-1/2 p-3 ${t.input}`} value={gender} onChange={e => setGender(e.target.value)}>
 <option value="m">Male (m)</option>
 <option value="f">Female (f)</option>
 <option value="x">Other (x)</option>
 </select>
 </div>

 <div className={`${t.inputInfo} p-4 mb-8`}>
 <p className={`text-sm ${t.inputInfoSub} mb-1`}>Auto-detected properties:</p>
 <p><span className={`font-mono ${t.inputInfoHighlight}`}>Speaker Code:</span> {getCode(txtFile)}</p>
 <p><span className={`font-mono ${t.inputInfoHighlight}`}>Audio Prefix:</span> {getPrefix(wavFile)}</p>
 </div>

 <button 
 onClick={handleProcess} 
 disabled={loading || !txtFile || !wavFile}
 className={`w-full py-4 disabled:opacity-50 disabled:cursor-not-allowed active:scale-[0.99] ${t.buttonPrimary}`}
 >
 {loading ? "Processing..." : "Process Files"}
 </button>

 {status && (
 <div className={`mt-6 p-4 border ${status.includes('❌') ? t.errorMsg : t.successMsg}`}>
 {status}
 </div>
 )}
 </div>
 </div>
 );
}

function View2({ theme }) {
 const t = theme;
 const [form, setForm] = useState({ train_pct: 80, valid_pct: 10, test_pct: 10, max_duration: 15, file_prefix: "cim-wav2vec2", use_code_switched: false, use_doubtful: false });
 const [status, setStatus] = useState(null);
 const [loading, setLoading] = useState(false);

 const handleGenerate = async () => {
 setLoading(true);
 setStatus("Generating partitions...");
 try {
 const res = await fetch("http://localhost:8000/api/generate_splits", {
 method: "POST",
 headers: { "Content-Type": "application/json" },
 body: JSON.stringify(form)
 });
 const data = await res.json();
 if (res.ok) {
 setStatus(`✅ Success! Train: ${data.train}, Valid: ${data.valid}, Test: ${data.test}`);
 } else {
 setStatus(`❌ Error: ${data.detail}`);
 }
 } catch (e) {
 setStatus(`❌ Error: ${e.message}`);
 }
 setLoading(false);
 };

 return (
 <div className="max-w-3xl mx-auto text-center ">
 <h2 className={`text-3xl font-bold mb-6 ${t.viewTitle}`}>2. Generate Wav2Vec2 Train Files</h2>
 <p className={`${t.viewDesc} mb-8`}>Split your preprocessed audio metadata into training, validation, and test sets.</p>
 
 <div className={`${t.card} p-6 `}>
 <div className="grid grid-cols-3 gap-4 mb-6">
 {["train_pct", "valid_pct", "test_pct"].map(f => (
 <div key={f}>
 <label className={`block text-sm font-medium ${t.label} mb-2 capitalize`}>{f.replace('_pct', ' %')}</label>
 <input type="number" className={`w-full p-3 ${t.input}`} value={form[f]} onChange={e => setForm({...form, [f]: parseInt(e.target.value)})} />
 </div>
 ))}
 </div>

 <div className="grid grid-cols-2 gap-4 mb-8">
 <div>
 <label className={`block text-sm font-medium ${t.label} mb-2`}>Max Duration (s)</label>
 <input type="number" className={`w-full p-3 ${t.input}`} value={form.max_duration} onChange={e => setForm({...form, max_duration: parseInt(e.target.value)})} />
 </div>
 <div>
 <label className={`block text-sm font-medium ${t.label} mb-2`}>Output Prefix</label>
 <input type="text" className={`w-full p-3 ${t.input}`} value={form.file_prefix} onChange={e => setForm({...form, file_prefix: e.target.value})} />
 </div>
 </div>

 <div className={`flex gap-6 mb-8 p-4 border ${t.checkboxContainer}`}>
 <label className="flex items-center space-x-3 cursor-pointer group">
 <input type="checkbox" className={t.checkbox} checked={form.use_code_switched} onChange={e => setForm({...form, use_code_switched: e.target.checked})} />
 <span className={t.checkboxText}>Include Code-Switched Data</span>
 </label>
 <label className="flex items-center space-x-3 cursor-pointer group">
 <input type="checkbox" className={t.checkbox} checked={form.use_doubtful} onChange={e => setForm({...form, use_doubtful: e.target.checked})} />
 <span className={t.checkboxText}>Include Doubtful Data</span>
 </label>
 </div>

 <button 
 onClick={handleGenerate} 
 disabled={loading}
 className={`w-full py-4 disabled:opacity-50 disabled:cursor-not-allowed active:scale-[0.99] ${t.buttonSecondary}`}
 >
 {loading ? "Generating..." : "Generate Train Files"}
 </button>

 {status && (
 <div className={`mt-6 p-4 border ${status.includes('❌') ? t.errorMsg : t.successMsg}`}>
 {status}
 </div>
 )}
 </div>
 </div>
 );
}

function View3({ theme }) {
 const t = theme;
 const [form, setForm] = useState({ train_csv: "cim-wav2vec2-train.csv", valid_csv: "cim-wav2vec2-valid.csv", test_csv: "cim-wav2vec2-test.csv", epochs: 34, ngrams: 4, run_id: "01", lang_prefix: "cim", lmplz_path: "", device: "all" });
 const [status, setStatus] = useState(null);
 const [loading, setLoading] = useState(false);
 const [devices, setDevices] = useState([]);

 useEffect(() => {
 fetch("http://localhost:8000/api/devices")
 .then(res => res.json())
 .then(data => {
 setDevices(data.devices || []);
 if (data.devices && data.devices.length > 0) {
 const hasAll = data.devices.some(d => d.id === "all");
 setForm(prev => ({...prev, device: hasAll ? "all" : data.devices[0].id}));
 }
 }).catch(err => console.log("Failed to fetch devices", err));
 }, []);

 const handleTrain = async () => {
 setLoading(true);
 setStatus("Starting training...");
 try {
 const res = await fetch("http://localhost:8000/api/train", {
 method: "POST",
 headers: { "Content-Type": "application/json" },
 body: JSON.stringify(form)
 });
 const data = await res.json();
 if (res.ok) {
 setStatus(`✅ Success! ${data.message}`);
 } else {
 setStatus(`❌ Error: ${data.detail}`);
 }
 } catch (e) {
 setStatus(`❌ Error: ${e.message}`);
 }
 setLoading(false);
 };

 return (
 <div className="max-w-3xl mx-auto text-center ">
 <h2 className={`text-3xl font-bold mb-6 ${t.viewTitle}`}>3. Train Wav2Vec2 Model</h2>
 <p className={`${t.viewDesc} mb-8`}>Train your model using the generated partitions. This will take some time and run in the background.</p>
 
 <div className={`${t.card} p-6 `}>
 <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
 <div>
 <label className={`block text-sm font-medium ${t.label} mb-2`}>Train CSV</label>
 <input type="text" className={`w-full p-3 ${t.input}`} value={form.train_csv} onChange={e => setForm({...form, train_csv: e.target.value})} />
 </div>
 <div>
 <label className={`block text-sm font-medium ${t.label} mb-2`}>Valid CSV</label>
 <input type="text" className={`w-full p-3 ${t.input}`} value={form.valid_csv} onChange={e => setForm({...form, valid_csv: e.target.value})} />
 </div>
 <div>
 <label className={`block text-sm font-medium ${t.label} mb-2`}>Test CSV</label>
 <input type="text" className={`w-full p-3 ${t.input}`} value={form.test_csv} onChange={e => setForm({...form, test_csv: e.target.value})} />
 </div>
 </div>

 <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
 <div>
 <label className={`block text-sm font-medium ${t.label} mb-2`}>Epochs</label>
 <input type="number" className={`w-full p-3 ${t.input}`} value={form.epochs} onChange={e => setForm({...form, epochs: parseInt(e.target.value)})} />
 </div>
 <div>
 <label className={`block text-sm font-medium ${t.label} mb-2`}>KenLM n-grams</label>
 <input type="number" className={`w-full p-3 ${t.input}`} value={form.ngrams} onChange={e => setForm({...form, ngrams: parseInt(e.target.value)})} />
 </div>
 <div>
 <label className={`block text-sm font-medium ${t.label} mb-2`}>Run ID</label>
 <input type="text" className={`w-full p-3 ${t.input}`} value={form.run_id} onChange={e => setForm({...form, run_id: e.target.value})} />
 </div>
 <div>
 <label className={`block text-sm font-medium ${t.label} mb-2`}>Lang Prefix</label>
 <input type="text" className={`w-full p-3 ${t.input}`} value={form.lang_prefix} onChange={e => setForm({...form, lang_prefix: e.target.value})} />
 </div>
 </div>

 <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
 <div>
 <label className={`block text-sm font-medium ${t.label} mb-2`}>lmplz Path (Optional)</label>
 <input type="text" placeholder="e.g. C:\tools\lmplz.exe" className={`w-full p-3 ${t.input}`} value={form.lmplz_path} onChange={e => setForm({...form, lmplz_path: e.target.value})} />
 </div>
 <div>
 <label className={`block text-sm font-medium ${t.label} mb-2`}>Training Device</label>
 <select className={`w-full p-3 ${t.input}`} value={form.device} onChange={e => setForm({...form, device: e.target.value})}>
 {devices.length === 0 && <option value="all">All GPUs (Default)</option>}
 {devices.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
 </select>
 </div>
 </div>

 <button 
 onClick={handleTrain} 
 disabled={loading}
 className={`w-full py-4 disabled:opacity-50 disabled:cursor-not-allowed active:scale-[0.99] ${t.buttonPrimary}`}
 >
 {loading ? "Starting..." : "Start Training"}
 </button>

 {status && (
 <div className={`mt-6 p-4 border ${status.includes('❌') ? t.errorMsg : t.successMsg}`}>
 {status}
 </div>
 )}
 </div>
 </div>
 );
}

function View4({ theme }) {
 const t = theme;
 const [files, setFiles] = useState([]);
 const [checkpoints, setCheckpoints] = useState([]);
 const [form, setForm] = useState({ audio_file: "", checkpoint: "" });
 const [status, setStatus] = useState(null);
 const [loading, setLoading] = useState(false);

 useEffect(() => {
 fetch("http://localhost:8000/api/inference_files")
 .then(res => res.json())
 .then(data => {
 setFiles(data.wav_files);
 if (data.wav_files.length > 0) setForm(prev => ({ ...prev, audio_file: data.wav_files[0] }));
 }).catch(err => console.log("Failed to fetch audio files", err));

 fetch("http://localhost:8000/api/checkpoints")
 .then(res => res.json())
 .then(data => {
 setCheckpoints(data.checkpoints);
 if (data.checkpoints.length > 0) setForm(prev => ({ ...prev, checkpoint: data.checkpoints[0] }));
 }).catch(err => console.log("Failed to fetch checkpoints", err));
 }, []);

 const handleTranscribe = async () => {
 setLoading(true);
 setStatus("Transcribing... This may take a while depending on file length.");
 try {
 const res = await fetch("http://localhost:8000/api/transcribe_long", {
 method: "POST",
 headers: { "Content-Type": "application/json" },
 body: JSON.stringify(form)
 });
 const data = await res.json();
 if (res.ok) {
 setStatus(`✅ Success! TSV saved to ${data.tsv_file}`);
 } else {
 setStatus(`❌ Error: ${data.detail}`);
 }
 } catch (e) {
 setStatus(`❌ Error: ${e.message}`);
 }
 setLoading(false);
 };

 return (
 <div className="max-w-3xl mx-auto text-center ">
 <h2 className={`text-3xl font-bold mb-6 ${t.viewTitle}`}>4. Transcribe Long Recording</h2>
 <p className={`${t.viewDesc} mb-8`}>Use a trained model to transcribe a longer audio/video file.</p>
 
 <div className={`${t.card} p-6 `}>
 <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
 <div>
 <label className={`block text-sm font-medium ${t.label} mb-2`}>Audio File</label>
 <select className={`w-full p-3 ${t.input}`} value={form.audio_file} onChange={e => setForm({...form, audio_file: e.target.value})}>
 {files.length === 0 && <option>No files found</option>}
 {files.map(f => <option key={f} value={f}>{f}</option>)}
 </select>
 </div>
 <div>
 <label className={`block text-sm font-medium ${t.label} mb-2`}>Model Checkpoint</label>
 <select className={`w-full p-3 ${t.input}`} value={form.checkpoint} onChange={e => setForm({...form, checkpoint: e.target.value})}>
 {checkpoints.length === 0 && <option>No checkpoints found</option>}
 {checkpoints.map(f => <option key={f} value={f}>{f}</option>)}
 </select>
 </div>
 </div>

 <button 
 onClick={handleTranscribe} 
 disabled={loading || !form.audio_file || !form.checkpoint}
 className={`w-full py-4 disabled:opacity-50 disabled:cursor-not-allowed active:scale-[0.99] ${t.buttonSecondary}`}
 >
 {loading ? "Transcribing..." : "Start Transcription"}
 </button>

 {status && (
 <div className={`mt-6 p-4 border ${status.includes('❌') ? t.errorMsg : t.successMsg}`}>
 {status}
 </div>
 )}
 </div>
 </div>
 );
}

function View5({ theme }) {
 const t = theme;
 const [checkpoints, setCheckpoints] = useState([]);
 const [checkpoint, setCheckpoint] = useState("");
 const [status, setStatus] = useState(null);
 const [loading, setLoading] = useState(false);
 const [recording, setRecording] = useState(false);
 const [audioUrl, setAudioUrl] = useState("");
 const mediaRecorder = useRef(null);
 const audioChunks = useRef([]);
 const [audioBlob, setAudioBlob] = useState(null);

 useEffect(() => {
 fetch("http://localhost:8000/api/checkpoints")
 .then(res => res.json())
 .then(data => {
 setCheckpoints(data.checkpoints);
 if (data.checkpoints.length > 0) setCheckpoint(data.checkpoints[0]);
 }).catch(err => console.log("Failed to fetch checkpoints", err));
 }, []);

 const startRecording = async () => {
 try {
 const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
 mediaRecorder.current = new MediaRecorder(stream);
 mediaRecorder.current.ondataavailable = e => {
 audioChunks.current.push(e.data);
 };
 mediaRecorder.current.onstop = () => {
 const blob = new Blob(audioChunks.current, { type: "audio/webm" });
 setAudioBlob(blob);
 setAudioUrl(URL.createObjectURL(blob));
 audioChunks.current = [];
 };
 mediaRecorder.current.start();
 setRecording(true);
 setStatus("🔴 Recording...");
 } catch (e) {
 setStatus(`❌ Error accessing microphone: ${e.message}`);
 }
 };

 const stopRecording = () => {
 if (mediaRecorder.current) {
 mediaRecorder.current.stop();
 mediaRecorder.current.stream.getTracks().forEach(track => track.stop());
 setRecording(false);
 setStatus("Recording complete. Ready to transcribe.");
 }
 };

 const handleTranscribe = async () => {
 if (!audioBlob) return;
 setLoading(true);
 setStatus("Transcribing...");
 try {
 const formData = new FormData();
 formData.append("checkpoint", checkpoint);
 formData.append("audio", audioBlob, "recording.webm");

 const res = await fetch("http://localhost:8000/api/transcribe_mic", {
 method: "POST",
 body: formData
 });
 const data = await res.json();
 if (res.ok) {
 setStatus(`✅ Transcription: ${data.transcription}`);
 } else {
 setStatus(`❌ Error: ${data.detail}`);
 }
 } catch (e) {
 setStatus(`❌ Error: ${e.message}`);
 }
 setLoading(false);
 };

 return (
 <div className="max-w-3xl mx-auto text-center ">
 <h2 className={`text-3xl font-bold mb-6 ${t.viewTitle}`}>5. Transcribe from Mic</h2>
 <p className={`${t.viewDesc} mb-8`}>Record your voice using the microphone and test the trained model.</p>
 
 <div className={`${t.card} p-6 `}>
 <div className="mb-8">
 <label className={`block text-sm font-medium ${t.label} mb-2`}>Model Checkpoint</label>
 <select className={`w-full md:w-1/2 p-3 ${t.input}`} value={checkpoint} onChange={e => setCheckpoint(e.target.value)}>
 {checkpoints.length === 0 && <option>No checkpoints found</option>}
 {checkpoints.map(f => <option key={f} value={f}>{f}</option>)}
 </select>
 </div>

 <div className="flex gap-4 items-center mb-8">
 <button 
 onClick={recording ? stopRecording : startRecording} 
 className={`px-6 py-3 font-bold active:scale-[0.99] flex items-center gap-2 ${recording ? 'bg-red-500 hover:bg-red-600 text-white' : 'bg-slate-200 hover:bg-slate-300 text-slate-800'}`}
 >
 {recording ? "⏹ Stop Recording" : "⏺ Start Recording"}
 </button>
 {audioUrl && !recording && (
 <audio src={audioUrl} controls className="h-10 outline-none"></audio>
 )}
 </div>

 <button 
 onClick={handleTranscribe} 
 disabled={loading || !audioBlob || !checkpoint}
 className={`w-full py-4 disabled:opacity-50 disabled:cursor-not-allowed active:scale-[0.99] ${t.buttonPrimary}`}
 >
 {loading ? "Transcribing..." : "Transcribe Recording"}
 </button>

 {status && (
 <div className={`mt-6 p-4 border ${status.includes('❌') ? t.errorMsg : (status.includes('✅') ? t.successMsg : t.inputInfo)}`}>
 {status}
 </div>
 )}
 </div>
 </div>
 );
}

const TABS = [
 { id: 0, name: "VAD Segmentation", title: "Auto-segment Long Audio" },
 { id: 1, name: "Data Preparation", title: "ELAN to WAV and CSV" },
 { id: 2, name: "Partitioning", title: "Generate Train Files" },
 { id: 3, name: "Training", title: "Train Wav2Vec2 Model" },
 { id: 4, name: "Inference", title: "Transcribe Long Recording" },
 { id: 5, name: "Microphone", title: "Transcribe from Mic" },
];

export default function App() {
  const [activeTab, setActiveTab] = useState(0);
  const [baseFolder, setBaseFolder] = useState("");
  const [selectingFolder, setSelectingFolder] = useState(false);
  const [activeThemeId, setActiveThemeId] = useState(() => {
    const saved = localStorage.getItem('cherokee-asr-theme');
    return saved === 'minimalLight' ? 'minimalLight' : 'minimalDark';
  });

  const t = THEMES[activeThemeId];

  useEffect(() => {
    localStorage.setItem('cherokee-asr-theme', activeThemeId);
  }, [activeThemeId]);

  useEffect(() => {
    fetch("http://localhost:8000/api/config/folder")
      .then(res => res.json())
      .then(data => setBaseFolder(data.folder))
      .catch(err => console.log("Failed to fetch folder", err));
  }, []);

  const handleBrowseFolder = async () => {
    setSelectingFolder(true);
    try {
      const res = await fetch("http://localhost:8000/api/config/select_folder", { method: "POST" });
      const data = await res.json();
      setBaseFolder(data.folder);
    } catch (err) {
      console.log("Failed to select folder", err);
    }
    setSelectingFolder(false);
  };

  const toggleTheme = () => {
    setActiveThemeId(prev => prev === 'minimalLight' ? 'minimalDark' : 'minimalLight');
  };

  return (
    <div className={`min-h-screen ${t.bgApp}`}>
      {/* Top Navigation */}
      <header className={t.topbar}>
        <div className={t.topbarTitle}>Transcription Tools</div>
        <button 
          onClick={toggleTheme} 
          className={`${t.buttonSecondary} absolute top-2 right-4 text-lg`} 
          style={{ padding: '2px 6px', lineHeight: '1' }}
          title="Toggle Theme"
        >
          {t.toggleIcon}
        </button>
        <div className="mb-2 text-sm">
          Base Folder: {baseFolder || "Loading..."} 
          <button onClick={handleBrowseFolder} disabled={selectingFolder} className={`ml-2 ${t.buttonSecondary}`}>
            {selectingFolder ? "Selecting..." : "Browse"}
          </button>
        </div>
        <nav>
          {TABS.map(tab => (
            <button 
              key={tab.id} 
              onClick={() => setActiveTab(tab.id)}
              className={activeTab === tab.id ? t.topbarTabActive : t.topbarTabInactive}
            >
              {tab.id}: {tab.name}
            </button>
          ))}
        </nav>
      </header>

      {/* Main Content Centered */}
      <main className={`p-4 ${t.bgMain}`}>
        <div key={baseFolder} className="mx-auto max-w-4xl text-center">
          {activeTab === 0 && <View0 theme={t} />}
          {activeTab === 1 && <View1 theme={t} />}
          {activeTab === 2 && <View2 theme={t} />}
          {activeTab === 3 && <View3 theme={t} />}
          {activeTab === 4 && <View4 theme={t} />}
          {activeTab === 5 && <View5 theme={t} />}
        </div>
      </main>
    </div>
  );
}
