import re

with open("frontend/src/App.jsx", "r", encoding="utf-8") as f:
    content = f.read()

# Replace THEMES
new_themes = """const THEMES = {
  minimalLight: {
    id: "minimalLight",
    name: "Web 1.0 Light",
    description: "Classic Web 1.0 style.",
    bgApp: "bg-white text-black font-serif",
    bgMain: "bg-white",
    sidebar: "hidden", 
    topbar: "border-b-2 border-black bg-gray-100 p-2 mb-4 text-center",
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
    toggleIcon: "Toggle Theme"
  },
  minimalDark: {
    id: "minimalDark",
    name: "Web 1.0 Dark",
    description: "Classic Web 1.0 style Dark.",
    bgApp: "bg-black text-white font-serif",
    bgMain: "bg-black",
    sidebar: "hidden", 
    topbar: "border-b-2 border-white bg-gray-800 p-2 mb-4 text-center",
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
    toggleIcon: "Toggle Theme"
  }
};
"""

content = re.sub(r'const THEMES = \{.*?\n\};\n', new_themes, content, flags=re.DOTALL)

# Replace App component
new_app = """export default function App() {
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
        <div className={t.topbarTitle}>Cherokee ASR</div>
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
        
        <div className="text-center mt-8">
          <button onClick={toggleTheme} className={t.buttonSecondary}>
            {t.toggleIcon}
          </button>
        </div>
      </main>
    </div>
  );
}
"""

content = re.sub(r'export default function App\(\) \{.*?\n\}\n$', new_app, content, flags=re.DOTALL)

with open("frontend/src/App.jsx", "w", encoding="utf-8") as f:
    f.write(content)
