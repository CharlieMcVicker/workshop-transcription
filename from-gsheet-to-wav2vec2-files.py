# Load other libraries
import pandas as pd
import random
import os
import json
import ipywidgets as widgets
from IPython.display import display

# Load libraries for access to Google Spreadsheets
from google.colab import auth
auth.authenticate_user()
import gspread
from google.colab import drive

# It needs this permission to access the ASR spreadsheets in your GDrive
from google.auth import default
creds, _ = default()
gc = gspread.authorize(creds)

# It needs this permission to read and write ASR files into your GDrive
drive.mount('/content/drive/')

# Use this if you have to remount the drive
# You need to do this if you update the files in the Google Drive while
# you are executing this notebook.
drive.mount('/content/drive/',force_remount=True)
#gc = gspread.authorize(GoogleCredentials.get_application_default())
gc = gspread.authorize(creds)

# ============================================================
#  SPECIFY FILES TO MAKE TRAINING FILES
# ============================================================
#  Reads the same Google Sheet as the previous notebook (detected
#  automatically), then you choose the partition / model options.
#  Click "✅ Confirm selections" to set every variable the rest of
#  this notebook needs (and to check the split adds up to 100%).
#
#  You normally won't edit anything here: the dropdowns list the
#  folders in your MyDrive. If your Drive lives somewhere unusual,
#  adjust MYDRIVE_ROOT below.
# ------------------------------------------------------------

# ----- Config -----------------------------------------------
MYDRIVE_ROOT = "/content/drive/MyDrive"           # where installation folders live
DEFAULT_INSTALLATION_FOLDER = "202606-cim-asr"    # default selection if present
SHEET_NAME_PREFIX = "asr-transcriptions-"         # sheet = <prefix><sandbox>
SOFTWARE_OPTIONS = ["wav2vec2", "ds", "whisper", "wavlm"]

# Optional manual override, used only if auto-detection can't find the
# sheet (maps a sandbox name -> URL).
SANDBOX_SHEET_URLS = {
    # "sandbox-user": "https://docs.google.com/spreadsheets/d/.../edit?usp=sharing",
}
# ------------------------------------------------------------


# ---------- helpers: folders on disk ------------------------
def list_installations():
    if not os.path.isdir(MYDRIVE_ROOT):
        return []
    return sorted(d for d in os.listdir(MYDRIVE_ROOT)
                  if os.path.isdir(os.path.join(MYDRIVE_ROOT, d)))

def sandbox_root():
    inst = install_dd.value
    return os.path.join(MYDRIVE_ROOT, inst) if inst else ""

def list_sandboxes():
    root = sandbox_root()
    if not root or not os.path.isdir(root):
        return []
    return sorted(d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d)))


# ---------- helpers: locate the Google Sheet URL ------------
def _read_gsheet_url(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
    except (OSError, ValueError):
        return ""
    doc_id = meta.get("doc_id") or ""
    if not doc_id and meta.get("resource_id"):
        rid = meta["resource_id"]
        doc_id = rid.split(":", 1)[1] if ":" in rid else rid
    if not doc_id and meta.get("url") and "id=" in meta["url"]:
        doc_id = meta["url"].split("id=", 1)[1].split("&", 1)[0]
    if doc_id:
        return f"https://docs.google.com/spreadsheets/d/{doc_id}/edit?usp=sharing"
    return meta.get("url", "")

def find_sheet_url(sandbox):
    """Look on disk for the sheet named exactly '<prefix><sandbox>'."""
    if not sandbox:
        return ""
    root = sandbox_root()
    stem = (SHEET_NAME_PREFIX + sandbox).lower()
    search_dirs = [root, os.path.join(root, sandbox)]
    for d in list(search_dirs):
        if os.path.isdir(d):
            search_dirs += [os.path.join(d, n) for n in os.listdir(d)
                            if os.path.isdir(os.path.join(d, n))]
    seen = set()
    for d in search_dirs:
        if not d or d in seen or not os.path.isdir(d):
            continue
        seen.add(d)
        for name in os.listdir(d):
            base, ext = os.path.splitext(name)
            if ext.lower() not in ("", ".gsheet"):
                continue
            if base.lower() == stem:        # exactly "<prefix><selected folder>"
                url = _read_gsheet_url(os.path.join(d, name))
                if url:
                    return url
    return ""

def _drive_escape(s):
    return s.replace("\\", "\\\\").replace("'", "\\'")

def _drive_folder_id(service, installation):
    q = (f"name = '{_drive_escape(installation)}' and "
         f"mimeType = 'application/vnd.google-apps.folder' and "
         f"'root' in parents and trashed = false")
    res = service.files().list(
        q=q, fields="files(id, name)", pageSize=10,
        supportsAllDrives=True, includeItemsFromAllDrives=True).execute()
    files = res.get("files", [])
    return files[0]["id"] if files else ""

def find_sheet_url_via_api(sandbox, log=print):
    """Ask Drive for '<prefix><sandbox>', searching ONLY inside the
    selected installation folder. Triggers a one-time auth prompt."""
    if not sandbox:
        return ""
    installation = install_dd.value
    try:
        from google.colab import auth
        from googleapiclient.discovery import build
        auth.authenticate_user()
        service = build("drive", "v3")
        folder_id = _drive_folder_id(service, installation) if installation else ""
        if installation and not folder_id:
            log(f"   couldn't locate the installation folder "
                f"'{installation}' in your Drive")
            return ""
        name = SHEET_NAME_PREFIX + sandbox
        q = (f"name = '{_drive_escape(name)}' and "
             f"mimeType = 'application/vnd.google-apps.spreadsheet' and "
             f"trashed = false and '{folder_id}' in parents")
        res = service.files().list(
            q=q, fields="files(id, name)", pageSize=5,
            supportsAllDrives=True, includeItemsFromAllDrives=True).execute()
        files = res.get("files", [])
        if files:
            if len(files) > 1:
                log(f"   note: {len(files)} sheets named '{name}' in "
                    f"'{installation}'; using the first.")
            fid = files[0]["id"]
            return f"https://docs.google.com/spreadsheets/d/{fid}/edit?usp=sharing"
        log(f"   (no spreadsheet named '{name}' inside '{installation}')")
    except Exception as e:                              # noqa: BLE001
        log(f"   Drive API lookup failed: {e}")
    return ""

def sheet_url_for(sandbox):
    return find_sheet_url(sandbox) or SANDBOX_SHEET_URLS.get(sandbox, "")

def audio_path_for(installation, sandbox):
    if not installation or not sandbox:
        return ""
    return f"{MYDRIVE_ROOT}/{installation}/{sandbox}/wav/"


# ---------- build the widgets -------------------------------
label_style = {"description_width": "220px"}
row_layout  = widgets.Layout(width="640px")

# Installation + sandbox drive everything below.
_inst_options = list_installations()
install_dd = widgets.Dropdown(
    description="Installation folder:", options=_inst_options,
    value=(DEFAULT_INSTALLATION_FOLDER if DEFAULT_INSTALLATION_FOLDER in _inst_options
           else (_inst_options[0] if _inst_options else None)),
    style=label_style, layout=row_layout)

sandbox_dd = widgets.Dropdown(description="Sandbox (destinationSandbox):",
                              options=[], style=label_style, layout=row_layout)

url_tb = widgets.Text(description="Sheet URL (urlSandbox):",
                      placeholder="auto-detected from the sheet in this folder",
                      style=label_style, layout=widgets.Layout(width="640px"))
sheet_btn = widgets.Button(description="🔗 Find Sheet URL",
                           tooltip="Look the sheet up in Google Drive "
                                   "(asks to authorize the first time)",
                           layout=widgets.Layout(width="180px"))
path_lbl = widgets.HTML()

# Model / data options.
software_dd = widgets.Dropdown(description="Software:", options=SOFTWARE_OPTIONS,
                               value="wav2vec2", style=label_style, layout=row_layout)
prefix_tb = widgets.Text(description="File prefix (filePrefix):",
                         value="cim-wav2vec2", style=label_style, layout=row_layout)
maxdur_tb = widgets.BoundedIntText(description="Max WAV duration (s):",
                                   value=15, min=1, max=600,
                                   style=label_style, layout=row_layout)

cs_cb = widgets.Checkbox(value=False, description="Use code-switched data",
                         indent=False)
dd_cb = widgets.Checkbox(value=False, description="Use doubtful data",
                         indent=False)

# Train / valid / test split.
train_pct = widgets.BoundedIntText(description="Train %:", value=80, min=0, max=100,
                                   step=5, style=label_style, layout=row_layout)
valid_pct = widgets.BoundedIntText(description="Validation %:", value=10, min=0, max=100,
                                   step=5, style=label_style, layout=row_layout)
test_pct  = widgets.BoundedIntText(description="Test %:", value=10, min=0, max=100,
                                   step=5, style=label_style, layout=row_layout)
pct_lbl = widgets.HTML()

confirm_btn = widgets.Button(description="✅ Confirm selections",
                             button_style="success",
                             layout=widgets.Layout(width="200px"))
status_out = widgets.Output()


# ---------- keep things in sync -----------------------------
def update_url_and_path(*_):
    sb = sandbox_dd.value
    url_tb.value = sheet_url_for(sb)
    path = audio_path_for(install_dd.value, sb)
    if path and os.path.isdir(path):
        path_lbl.value = (f"<span style='color:#1a7f37'>🎧 pathAudioFilesInTraining = "
                          f"<code>{path}</code></span>")
    elif path:
        path_lbl.value = (f"<span style='color:#c0392b'>🎧 pathAudioFilesInTraining = "
                          f"<code>{path}</code> (folder not found yet)</span>")
    else:
        path_lbl.value = ""

def populate_sandboxes(*_):
    sbs = list_sandboxes()
    default_sb = ("sandbox-user" if "sandbox-user" in sbs
                  else (sbs[0] if sbs else None))
    sandbox_dd.options = sbs
    sandbox_dd.value = default_sb
    update_url_and_path()

def on_software_change(change):
    # keep filePrefix in step with the model, unless it's been customised
    old, new = change.get("old"), change["new"]
    if prefix_tb.value.strip() in ("", f"cim-{old}"):
        prefix_tb.value = f"cim-{new}"

def on_pct_change(*_):
    total = train_pct.value + valid_pct.value + test_pct.value
    if total == 100:
        pct_lbl.value = "<span style='color:#1a7f37'>split adds up to 100% ✓</span>"
    else:
        pct_lbl.value = (f"<span style='color:#c0392b'>split adds up to {total}% "
                         f"— must be 100%</span>")

def on_find_sheet(_):
    with status_out:
        status_out.clear_output(wait=True)
        sb = sandbox_dd.value
        if not sb:
            print("⚠️ Pick a sandbox first.")
            return
        url = sheet_url_for(sb)
        if url:
            url_tb.value = url
            print(f"✅ Found the sheet on disk:\n   {url}")
            return
        print(f"Looking up '{SHEET_NAME_PREFIX}{sb}' in Google Drive "
              f"(you may be asked to authorize)…")
        url = find_sheet_url_via_api(sb)
        if url:
            url_tb.value = url
            print(f"✅ Found via Drive:\n   {url}")
        else:
            print("   Paste the URL manually if this keeps failing.")

install_dd.observe(populate_sandboxes, names="value")
sandbox_dd.observe(update_url_and_path, names="value")
software_dd.observe(on_software_change, names="value")
for w in (train_pct, valid_pct, test_pct):
    w.observe(on_pct_change, names="value")
sheet_btn.on_click(on_find_sheet)


# ---------- write the variables the notebook expects --------
def apply_selections():
    global destinationSandbox, installationFolder, urlSandbox
    global useCodeSwitchedData, useDoutbfulData
    global percentageTrainSet, percentageValidSet, percentageTestSet
    global maxWavDuration, software, filePrefix, pathAudioFilesInTraining
    installationFolder       = install_dd.value
    destinationSandbox       = sandbox_dd.value
    urlSandbox               = url_tb.value.strip()
    useCodeSwitchedData      = int(cs_cb.value)
    useDoutbfulData          = int(dd_cb.value)     # original spelling kept
    percentageTrainSet       = train_pct.value
    percentageValidSet       = valid_pct.value
    percentageTestSet        = test_pct.value
    maxWavDuration           = maxdur_tb.value
    software                 = software_dd.value
    filePrefix               = prefix_tb.value.strip()
    pathAudioFilesInTraining = audio_path_for(install_dd.value, sandbox_dd.value)

def on_confirm(_):
    apply_selections()
    with status_out:
        status_out.clear_output(wait=True)
        problems = []
        if not os.path.isdir(MYDRIVE_ROOT):
            problems.append(f"MyDrive not found at {MYDRIVE_ROOT} "
                            f"(is Google Drive mounted?).")
        elif not installationFolder:
            problems.append("No installation folder selected.")
        if not destinationSandbox:
            problems.append("No sandbox selected.")
        if not urlSandbox:
            problems.append("Sheet URL is empty — click 🔗 Find Sheet URL "
                            "or paste it in.")
        total = percentageTrainSet + percentageValidSet + percentageTestSet
        if total != 100:
            problems.append(f"Train/valid/test split is {total}%, must be 100%.")
        if software not in SOFTWARE_OPTIONS:
            problems.append(f"Software must be one of {SOFTWARE_OPTIONS}.")

        if problems:
            print("⚠️  Please fix:")
            for p in problems:
                print("   • " + p)
            return

        print("✅  All set. The notebook will use:")
        print(f"   installationFolder       = {installationFolder}")
        print(f"   destinationSandbox       = {destinationSandbox}")
        print(f"   urlSandbox               = {urlSandbox}")
        print(f"   useCodeSwitchedData      = {useCodeSwitchedData}")
        print(f"   useDoutbfulData          = {useDoutbfulData}")
        print(f"   percentageTrainSet       = {percentageTrainSet}")
        print(f"   percentageValidSet       = {percentageValidSet}")
        print(f"   percentageTestSet        = {percentageTestSet}")
        print(f"   maxWavDuration           = {maxWavDuration}")
        print(f"   software                 = {software}")
        print(f"   filePrefix               = {filePrefix}")
        print(f"   pathAudioFilesInTraining = {pathAudioFilesInTraining}")

confirm_btn.on_click(on_confirm)


# ---------- show it -----------------------------------------
on_pct_change()
if not _inst_options:
    apply_selections()                            # avoid NameErrors downstream
    display(widgets.HTML(
        f"<b style='color:#c0392b'>No folders found in {MYDRIVE_ROOT}.</b> "
        f"Mount Google Drive (uncomment the mount line at the top) and/or fix "
        f"MYDRIVE_ROOT, then re-run this cell."))
else:
    populate_sandboxes()                          # fill sandbox + url + path
    apply_selections()                            # so later cells never NameError
    display(widgets.VBox([
        widgets.HTML("<b>1 · Data source</b>"),
        install_dd, sandbox_dd,
        widgets.HBox([url_tb, sheet_btn]),
        path_lbl,
        widgets.HTML("<b>2 · Model &amp; data options</b>"),
        software_dd, prefix_tb, maxdur_tb, cs_cb, dd_cb,
        widgets.HTML("<b>3 · Train / validation / test split</b>"),
        train_pct, valid_pct, test_pct, pct_lbl,
        widgets.HTML("&nbsp;"),
        confirm_btn, status_out,
    ]))


#==============================================
# Read in sandbox information
#==============================================

savePath = "/content/drive/MyDrive/"+installationFolder+"/" + destinationSandbox + "/" # This is where the metadata files will be stored
gsheetURL = urlSandbox

def saveFile(string, path):
  f = open(path, "w")
  f.write(string)
  f.close()

# Open Google Spreadsheet

wb = gc.open_by_url(gsheetURL)
sheet = wb.worksheet('wav-metadata')
rows = sheet.get_all_values()

metadata = pd.DataFrame(rows)
metadata.iloc[0]
metadata.columns[0]

metadata.columns = metadata.iloc[0]
metadata = metadata.iloc[1:]

if (useCodeSwitchedData == 0):
  metadata = metadata[metadata['codeSwitch'] != "1"]

if (useDoutbfulData == 0):
  metadata = metadata[metadata['needsFurtherCheck'] != "1"]

metadata = metadata[metadata['transcript'] != ""]

#metadata['duration_seconds'] = pd.to_numeric(metadata['duration_seconds'])
metadata['duration_seconds'] = pd.to_numeric(metadata['duration_seconds'].str.replace(',', '.'))
metadata = metadata[metadata['duration_seconds'] <= maxWavDuration]

print(metadata)

longfilenames = []
sentences = []
listOfNumbers = []
filesizes = []
j = 0

for i, r in metadata.iterrows():
  longfilenames.append(pathAudioFilesInTraining + r['wav_filename'])
  sentences.append(r['transcript'])
  filesizes.append(r['wav_filesize'])
  listOfNumbers.append(j)
  j = j+1

random.shuffle(listOfNumbers)

samplesUpToTrainPartition = int(round(len(listOfNumbers) * (percentageTrainSet/100),0))
samplesUpToTestPartition = int(round(len(listOfNumbers) * ((percentageTrainSet + percentageValidSet)/100),0))

trainPath = []
trainText = []
trainSize = []
validPath = []
validText = []
validSize = []
testPath = []
testText = []
testSize = []

counterTrain = 0
counterValid = 0
counterTest = 0

for i in range(0,samplesUpToTrainPartition):
  counterTrain = counterTrain+1
  trainPath.append(longfilenames[listOfNumbers[i]])
  trainText.append(sentences[listOfNumbers[i]])
  trainSize.append(filesizes[listOfNumbers[i]])

for i in range(samplesUpToTrainPartition,samplesUpToTestPartition):
  counterValid = counterValid + 1
  validPath.append(longfilenames[listOfNumbers[i]])
  validText.append(sentences[listOfNumbers[i]])
  validSize.append(filesizes[listOfNumbers[i]])

for i in range(samplesUpToTestPartition,len(listOfNumbers)):
  counterTest = counterTest + 1
  testPath.append(longfilenames[listOfNumbers[i]])
  testText.append(sentences[listOfNumbers[i]])
  testSize.append(filesizes[listOfNumbers[i]])

print("Training samples:   " + str(counterTrain))
print("Validation samples: " + str(counterValid))
print("Test samples:       " + str(counterTest))

#===========================================================================
# Write files as CSVs
#===========================================================================

def writeCSVFile(header, software, inPath, inSentences, inSizes, inFilename):

  output = header + "\n"

  if (software == "wav2vec2"):
    for i in range(0,len(inPath)): output = output + inPath[i] + "," + inSentences[i] + "\n"
  elif (software == "ds"):
    for i in range(0,len(inPath)): output = output + inPath[i] + "," + inSizes[i] + "," + inSentences[i] + "\n"
  output = output[:-1]
  f = open(inFilename, "w")
  f.write(output)
  f.close()

header = ""

if (software == "wav2vec2"):
  header = "path,sentence"
elif (software == "ds"):
  header = "wav_filename,wav_filesize,transcript"

#filenameTrain = savePath + software+"-train.csv"
#filenameValid = savePath + software+"-valid.csv"
#filenameTest =  savePath + software+"-test.csv"

filenameTrain = savePath + filePrefix+"-train.csv"
filenameValid = savePath + filePrefix+"-valid.csv"
filenameTest =  savePath + filePrefix+"-test.csv"


print(filenameTrain)
print(filenameValid)
print(filenameTest)

writeCSVFile(header, software, trainPath, trainText, trainSize, filenameTrain)
writeCSVFile(header, software, validPath, validText, validSize, filenameValid)
writeCSVFile(header, software, testPath, testText, testSize, filenameTest)