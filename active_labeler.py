#!/usr/bin/env python3
import http.server
import socketserver
import json
import urllib.parse
import os
import csv
import sys
import argparse

PORT = 8000
CSV_FILE = "batch_inference_results.csv"
TRAIN_FILE = "train_labeled.csv"

# Custom HTTP request handler
class LabelingToolHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Enable CORS and disable caching for API/assets
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path == "/api/data":
            # Read batch_inference_results.csv and return it as JSON
            if not os.path.exists(CSV_FILE):
                self.send_error_json(404, f"File {CSV_FILE} not found.")
                return

            try:
                data = []
                with open(CSV_FILE, mode='r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        data.append({
                            "file_path": row.get("file_path", ""),
                            "filename": row.get("filename", ""),
                            "greedy_transcription": row.get("greedy_transcription", ""),
                            "greedy_confidence": float(row.get("greedy_confidence", 0.0)) if row.get("greedy_confidence") else 0.0
                        })
                
                # Sort segments by confidence ascending (lowest confidence first)
                data.sort(key=lambda x: x["greedy_confidence"])

                # Load existing labels if train_labeled.csv exists to preserve state
                labeled_map = {}
                if os.path.exists(TRAIN_FILE):
                    with open(TRAIN_FILE, mode='r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            # Map path to sentence
                            labeled_map[row.get("path", "")] = row.get("sentence", "")

                # Inject existing labels
                for row in data:
                    row["labeled_sentence"] = labeled_map.get(row["file_path"], "")

                response_bytes = json.dumps({"status": "success", "data": data}).encode('utf-8')
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_bytes)))
                self.end_headers()
                self.wfile.write(response_bytes)
            except Exception as e:
                self.send_error_json(500, str(e))
            return

        elif path in ["", "/", "/index.html"]:
            # Serve the SPA UI
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(UI_HTML.encode('utf-8'))
            return

        # Serve static audio files and other assets
        super().do_GET()

    def do_POST(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path == "/api/save":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                payload = json.loads(post_data.decode('utf-8'))
                labels = payload.get("labels", [])  # List of {path: ..., sentence: ...}

                # Save to train_labeled.csv
                with open(TRAIN_FILE, mode='w', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["path", "sentence"])
                    for label in labels:
                        writer.writerow([label.get("path"), label.get("sentence")])

                response_bytes = json.dumps({"status": "success", "message": f"Successfully saved {len(labels)} labels to {TRAIN_FILE}"}).encode('utf-8')
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_bytes)))
                self.end_headers()
                self.wfile.write(response_bytes)
            except Exception as e:
                self.send_error_json(500, str(e))
            return

        self.send_error_json(404, "Endpoint not found")

    def send_error_json(self, code, message):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        err_bytes = json.dumps({"status": "error", "message": message}).encode('utf-8')
        self.send_header("Content-Length", str(len(err_bytes)))
        self.end_headers()
        self.wfile.write(err_bytes)

# SPA interface html
UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Active Labeling Tool - Batch Inference Results</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0b0f19;
            --bg-secondary: #161f30;
            --bg-tertiary: #1f2d44;
            --accent-primary: #4f46e5;
            --accent-hover: #6366f1;
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --text-tertiary: #6b7280;
            --success: #10b981;
            --warning: #f59e0b;
            --border-color: rgba(255, 255, 255, 0.08);
            --transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            display: flex;
            height: 100vh;
            overflow: hidden;
        }

        /* Sidebar styling */
        .sidebar {
            width: 420px;
            background-color: var(--bg-secondary);
            border-right: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            height: 100%;
        }

        .sidebar-header {
            padding: 24px;
            border-bottom: 1px solid var(--border-color);
        }

        .sidebar-header h1 {
            font-size: 1.25rem;
            font-weight: 700;
            letter-spacing: -0.025em;
            margin-bottom: 6px;
            background: linear-gradient(135deg, #a5b4fc 0%, #818cf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .sidebar-header p {
            font-size: 0.85rem;
            color: var(--text-secondary);
        }

        .filter-box {
            padding: 16px 24px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            gap: 12px;
            align-items: center;
        }

        .search-input {
            flex: 1;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 8px 12px;
            color: var(--text-primary);
            font-family: inherit;
            font-size: 0.9rem;
            outline: none;
            transition: var(--transition);
        }

        .search-input:focus {
            border-color: var(--accent-primary);
        }

        .save-btn {
            background-color: var(--success);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: 600;
            font-size: 0.9rem;
            cursor: pointer;
            transition: var(--transition);
        }

        .save-btn:hover {
            filter: brightness(1.1);
        }

        .segment-list {
            flex: 1;
            overflow-y: auto;
            padding: 12px 24px;
        }

        .segment-item {
            background-color: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 12px;
            cursor: pointer;
            transition: var(--transition);
            position: relative;
        }

        .segment-item:hover {
            transform: translateY(-2px);
            border-color: var(--accent-hover);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }

        .segment-item.active {
            border-color: var(--accent-primary);
            background-color: rgba(79, 70, 229, 0.1);
            box-shadow: 0 0 0 1px var(--accent-primary);
        }

        .segment-meta {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }

        .badge-confidence {
            font-size: 0.75rem;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 9999px;
            text-transform: uppercase;
        }

        .badge-low {
            background-color: rgba(239, 68, 68, 0.15);
            color: #ef4444;
        }

        .badge-medium {
            background-color: rgba(245, 158, 11, 0.15);
            color: #f59e0b;
        }

        .badge-high {
            background-color: rgba(16, 185, 129, 0.15);
            color: #10b981;
        }

        .segment-filename {
            font-size: 0.75rem;
            color: var(--text-tertiary);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 180px;
        }

        .segment-text-preview {
            font-size: 0.9rem;
            color: var(--text-primary);
            line-height: 1.4;
        }

        .badge-labeled {
            position: absolute;
            top: 16px;
            right: 16px;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: var(--success);
            box-shadow: 0 0 8px var(--success);
        }

        /* Detail/Workspace Area */
        .workspace {
            flex: 1;
            padding: 48px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            overflow-y: auto;
        }

        .card {
            background-color: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 40px;
            width: 100%;
            max-width: 800px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
            display: flex;
            flex-direction: column;
            gap: 24px;
        }

        .card h2 {
            font-size: 1.5rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }

        .audio-player-container {
            background: var(--bg-tertiary);
            border-radius: 12px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 12px;
            border: 1px solid var(--border-color);
        }

        audio {
            width: 100%;
            outline: none;
        }

        .audio-path {
            font-size: 0.8rem;
            color: var(--text-secondary);
            font-family: monospace;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .form-group label {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .input-text {
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 14px;
            color: var(--text-primary);
            font-size: 1.1rem;
            font-family: inherit;
            outline: none;
            transition: var(--transition);
        }

        .input-text:focus {
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 1px var(--accent-primary);
        }

        .button-group {
            display: flex;
            justify-content: space-between;
            margin-top: 12px;
        }

        .btn {
            padding: 12px 24px;
            font-size: 0.95rem;
            font-weight: 600;
            border-radius: 10px;
            cursor: pointer;
            border: none;
            transition: var(--transition);
        }

        .btn-primary {
            background-color: var(--accent-primary);
            color: white;
        }

        .btn-primary:hover {
            background-color: var(--accent-hover);
        }

        .btn-secondary {
            background-color: var(--bg-tertiary);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
        }

        .btn-secondary:hover {
            background-color: rgba(255, 255, 255, 0.05);
        }

        .empty-state {
            text-align: center;
            color: var(--text-secondary);
        }

        .empty-state h3 {
            font-size: 1.25rem;
            margin-bottom: 8px;
            color: var(--text-primary);
        }

        /* Toast notification */
        .toast {
            position: fixed;
            bottom: 24px;
            right: 24px;
            background-color: var(--success);
            color: white;
            padding: 12px 24px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            transform: translateY(100px);
            opacity: 0;
            transition: var(--transition);
            z-index: 1000;
        }

        .toast.show {
            transform: translateY(0);
            opacity: 1;
        }
    </style>
</head>
<body>
    <div class="sidebar">
        <div class="sidebar-header">
            <h1>Active Labeler</h1>
            <p>Sort: Lowest Confidence first</p>
        </div>
        <div class="filter-box">
            <input type="text" class="search-input" placeholder="Search transcription..." id="searchInput">
            <button class="save-btn" onclick="saveAllLabels()">Save CSV</button>
        </div>
        <div class="segment-list" id="segmentList">
            <!-- Items loaded dynamically -->
        </div>
    </div>

    <div class="workspace" id="workspace">
        <div class="empty-state">
            <h3>Select a segment to start</h3>
            <p>Low confidence entries are sorted at the top of the list.</p>
        </div>
    </div>

    <div class="toast" id="toast">Saved successfully!</div>

    <script>
        let segments = [];
        let currentIndex = -1;

        async function loadData() {
            try {
                const res = await fetch('/api/data');
                const result = await res.json();
                if (result.status === 'success') {
                    segments = result.data;
                    renderList();
                } else {
                    alert('Error loading data: ' + result.message);
                }
            } catch (err) {
                console.error(err);
                alert('Could not fetch data from server.');
            }
        }

        function getConfidenceClass(conf) {
            if (conf < 0.8) return 'badge-low';
            if (conf < 0.95) return 'badge-medium';
            return 'badge-high';
        }

        function renderList() {
            const listEl = document.getElementById('segmentList');
            const searchVal = document.getElementById('searchInput').value.toLowerCase();
            listEl.innerHTML = '';

            segments.forEach((seg, index) => {
                if (searchVal && !seg.greedy_transcription.toLowerCase().includes(searchVal)) {
                    return;
                }

                const item = document.createElement('div');
                item.className = `segment-item ${index === currentIndex ? 'active' : ''}`;
                item.onclick = () => selectSegment(index);

                const hasLabel = !!seg.labeled_sentence;
                const confClass = getConfidenceClass(seg.greedy_confidence);

                item.innerHTML = `
                    <div class="segment-meta">
                        <span class="badge-confidence ${confClass}">${seg.greedy_confidence.toFixed(4)}</span>
                        <span class="segment-filename" title="${seg.filename}">${seg.filename}</span>
                    </div>
                    <div class="segment-text-preview">
                        ${hasLabel ? `<strong>[L]</strong> ${seg.labeled_sentence}` : seg.greedy_transcription}
                    </div>
                    ${hasLabel ? `<div class="badge-labeled"></div>` : ''}
                `;
                listEl.appendChild(item);
            });
        }

        function selectSegment(index) {
            currentIndex = index;
            renderList();
            renderWorkspace();
        }

        function renderWorkspace() {
            const workspaceEl = document.getElementById('workspace');
            if (currentIndex === -1 || !segments[currentIndex]) {
                workspaceEl.innerHTML = `
                    <div class="empty-state">
                        <h3>Select a segment to start</h3>
                        <p>Low confidence entries are sorted at the top of the list.</p>
                    </div>
                `;
                return;
            }

            const seg = segments[currentIndex];
            const audioPath = seg.file_path; 
            const defaultSentence = seg.labeled_sentence || seg.greedy_transcription;

            workspaceEl.innerHTML = `
                <div class="card">
                    <h2>Label Segment</h2>
                    <div class="audio-player-container">
                        <audio id="audioPlayer" src="/${audioPath}" controls autoplay></audio>
                        <div class="audio-path">${audioPath}</div>
                    </div>

                    <div class="form-group">
                        <label>Original Transcription (Confidence: ${seg.greedy_confidence.toFixed(4)})</label>
                        <div class="input-text" style="background-color: var(--bg-tertiary); opacity: 0.8; font-size: 0.95rem;">
                            ${seg.greedy_transcription || '<i>(Empty)</i>'}
                        </div>
                    </div>

                    <div class="form-group">
                        <label for="labelInput">Correction / Labeled Sentence</label>
                        <input type="text" class="input-text" id="labelInput" value="${defaultSentence}" placeholder="Type exact transcription here...">
                    </div>

                    <div class="button-group">
                        <button class="btn btn-secondary" onclick="navigate(-1)">Previous</button>
                        <button class="btn btn-primary" onclick="submitLabel()">Save & Next</button>
                    </div>
                </div>
            `;

            // Focus input element
            setTimeout(() => {
                const input = document.getElementById('labelInput');
                if (input) {
                    input.focus();
                    input.select();
                }
            }, 50);
        }

        function submitLabel() {
            const inputVal = document.getElementById('labelInput').value.trim();
            if (currentIndex !== -1 && segments[currentIndex]) {
                segments[currentIndex].labeled_sentence = inputVal;
                showToast("Label stored locally");
                
                // Advance to next
                navigate(1);
            }
        }

        function navigate(direction) {
            const nextIdx = currentIndex + direction;
            if (nextIdx >= 0 && nextIdx < segments.length) {
                selectSegment(nextIdx);
            } else {
                renderList();
                renderWorkspace();
            }
        }

        async function saveAllLabels() {
            // Get all items that have a labeled sentence
            const labeled = segments
                .filter(seg => !!seg.labeled_sentence)
                .map(seg => ({
                    path: seg.file_path,
                    sentence: seg.labeled_sentence
                }));

            try {
                const res = await fetch('/api/save', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ labels: labeled })
                });
                const result = await res.json();
                if (result.status === 'success') {
                    showToast("CSV successfully saved!");
                    renderList();
                } else {
                    alert("Failed to save: " + result.message);
                }
            } catch (err) {
                console.error(err);
                alert("Error calling save API");
            }
        }

        function showToast(msg) {
            const toast = document.getElementById('toast');
            toast.innerText = msg;
            toast.classList.add('show');
            setTimeout(() => {
                toast.classList.remove('show');
            }, 3000);
        }

        // Search listener
        document.getElementById('searchInput').addEventListener('input', () => {
            renderList();
        });

        // Keybindings
        window.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && e.target.id === 'labelInput') {
                submitLabel();
            }
        });

        // Init
        loadData();
    </script>
</body>
</html>
"""

def main():
    parser = argparse.ArgumentParser(description="Active Labeler Server")
    parser.add_argument("--port", type=int, default=PORT, help="Port to listen on")
    args = parser.parse_args()

    # Change to root dir containing script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Start server
    handler = LabelingToolHandler
    with socketserver.TCPServer(("", args.port), handler) as httpd:
        print(f"Active Labeler Server running at: http://localhost:{args.port}/")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            sys.exit(0)

if __name__ == "__main__":
    main()
