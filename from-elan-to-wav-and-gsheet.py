# Load relevant libraries
import math
import pandas as pd
from datetime import date
import os
import wave
import contextlib

import json
import ipywidgets as widgets
from IPython.display import display

# from google.colab import drive

# It needs this permission to access the ASR spreadsheets in your GDrive
from google.colab import auth
auth.authenticate_user()
import gspread
from google.auth import default
creds, _ = default()
gc = gspread.authorize(creds)

# It needs this permission to read and write ASR files into your GDrive
# drive.mount('/content/drive/', force_remount=True)

# ============================================================
#  PICK YOUR FILES  —  no typing of filenames required
# ============================================================
#  HOW TO USE
#    1. (If your files live on Google Drive) run the mount cell
#       once per session — see the commented line just below.
#    2. Run this cell.
#    3. Pick the installation folder, then the sandbox, then the
#       files. Speaker code and audio prefix fill in automatically.
#    4. Click  "✅ Confirm selections".  This checks the files
#       really exist and sets every variable the rest of the
#       notebook needs.
#
#  You normally won't need to edit anything: the cell lists the
#  folders in your MyDrive and works down from there. If your
#  Drive lives somewhere unusual, adjust MYDRIVE_ROOT below.
# ------------------------------------------------------------

# ----- Config -----------------------------------------------
# Where your installation folders live (the top level of Drive):
MYDRIVE_ROOT = "/content/drive/MyDrive"

# Which installation folder to select by default. If it isn't
# present, the dropdown still lists every folder in MyDrive so
# you can choose another one.
DEFAULT_INSTALLATION_FOLDER = "202606-cim-asr"

# Subfolder, inside each sandbox, that holds the .txt and .wav:
ELAN_SUBFOLDER = "processed-elan-files"

# The Sheet URL is detected automatically from the Google Sheet that
# lives in the installation folder, named "<prefix><sandbox>", e.g.
# "asr-transcriptions-sandbox-user". Change the prefix if yours differs.
SHEET_NAME_PREFIX = "asr-transcriptions-"

# Optional manual override, used only if auto-detection can't find the
# sheet (maps a sandbox name -> URL).
SANDBOX_SHEET_URLS = {
    # "sandbox-user": "https://docs.google.com/spreadsheets/d/.../edit?usp=sharing",
}
# ------------------------------------------------------------


# ---------- helpers: read what's actually on disk -----------
def list_installations():
    """Every folder directly inside MyDrive."""
    if not os.path.isdir(MYDRIVE_ROOT):
        return []
    return sorted(d for d in os.listdir(MYDRIVE_ROOT)
                  if os.path.isdir(os.path.join(MYDRIVE_ROOT, d)))

def sandbox_root():
    """The chosen installation folder = the place that holds the sandboxes."""
    inst = install_dd.value
    return os.path.join(MYDRIVE_ROOT, inst) if inst else ""

def list_sandboxes():
    """Sandbox folders inside the currently selected installation folder."""
    root = sandbox_root()
    if not root or not os.path.isdir(root):
        return []
    return sorted(d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d)))

def resolve_elan_folder(sandbox):
    """Find the folder that actually holds the .txt / .wav files.
    Tries, in order: the expected subfolder; a same-named subfolder
    ignoring case; the sandbox folder itself; then any nested folder
    (down to 3 levels) that contains both a .txt and a .wav."""
    root = sandbox_root()
    if not sandbox or not root:
        return None
    base = os.path.join(root, sandbox)

    expected = os.path.join(base, ELAN_SUBFOLDER)
    if os.path.isdir(expected):
        return expected

    if os.path.isdir(base):
        for name in os.listdir(base):
            p = os.path.join(base, name)
            if os.path.isdir(p) and name.lower() == ELAN_SUBFOLDER.lower():
                return p
        if any(f.lower().endswith((".txt", ".wav")) for f in os.listdir(base)):
            return base
        for dirpath, dirnames, files in os.walk(base):
            if dirpath[len(base):].count(os.sep) >= 3:
                dirnames[:] = []
            low = [f.lower() for f in files]
            if any(f.endswith(".txt") for f in low) and \
               any(f.endswith(".wav") for f in low):
                return dirpath

    return expected   # nothing found; return expected so we can report it

def list_files(sandbox, extension):
    folder = resolve_elan_folder(sandbox)
    if not folder or not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder)
                  if f.lower().endswith(extension))

def _read_gsheet_url(path):
    """Pull the document id out of a .gsheet JSON pointer and build a link."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
    except (OSError, ValueError):
        return ""
    doc_id = meta.get("doc_id") or ""
    if not doc_id and meta.get("resource_id"):          # e.g. "spreadsheet:1AbC..."
        rid = meta["resource_id"]
        doc_id = rid.split(":", 1)[1] if ":" in rid else rid
    if not doc_id and meta.get("url") and "id=" in meta["url"]:
        doc_id = meta["url"].split("id=", 1)[1].split("&", 1)[0]
    if doc_id:
        return f"https://docs.google.com/spreadsheets/d/{doc_id}/edit?usp=sharing"
    return meta.get("url", "")

def find_sheet_url(sandbox):
    """Find the Google Sheet 'asr-transcriptions-<sandbox>' on disk and
    return its URL, or '' if not found / not readable. Searches the
    installation folder (the level above the sandbox), the sandbox folder
    itself, and one level of subfolders. NOTE: Colab's mounted Drive often
    does NOT expose Google Sheets as files at all — in that case this
    returns '' and the Drive-API lookup below is needed instead."""
    if not sandbox:
        return ""
    root = sandbox_root()
    stem = (SHEET_NAME_PREFIX + sandbox).lower()

    search_dirs = [root, os.path.join(root, sandbox)]
    for d in list(search_dirs):                         # add immediate subfolders
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
            if base.lower() == stem:        # exactly "asr-transcriptions-<folder>"
                url = _read_gsheet_url(os.path.join(d, name))
                if url:
                    return url
    return ""

def _drive_escape(s):
    """Escape a value for a Drive API query string."""
    return s.replace("\\", "\\\\").replace("'", "\\'")

def _drive_folder_id(service, installation):
    """Resolve the Drive file-id of the installation folder, which sits
    directly under My Drive. Returns '' if it can't be found."""
    q = (f"name = '{_drive_escape(installation)}' and "
         f"mimeType = 'application/vnd.google-apps.folder' and "
         f"'root' in parents and trashed = false")
    res = service.files().list(
        q=q, fields="files(id, name)", pageSize=10,
        supportsAllDrives=True, includeItemsFromAllDrives=True).execute()
    files = res.get("files", [])
    return files[0]["id"] if files else ""

def find_sheet_url_via_api(sandbox, log=print):
    """Ask Google Drive for the sheet named 'asr-transcriptions-<sandbox>',
    searching ONLY inside the selected installation folder (so identically
    named sheets elsewhere in the Drive are ignored). Triggers a one-time
    auth prompt. Works even when the mounted Drive hides Google files."""
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
    """Auto-detected URL first, manual override second."""
    return find_sheet_url(sandbox) or SANDBOX_SHEET_URLS.get(sandbox, "")

# ---------- helpers: auto-fill from your naming pattern -----
def derive_speaker_code(tab_filename):
    """ 'RRMSAvaikiP24-msc.txt' -> 'MSC' """
    stem = os.path.splitext(tab_filename)[0]
    return stem.rsplit("-", 1)[1].upper() if "-" in stem else ""

def derive_prefix(wav_filename):
    """ 'RRMSAvaikiP24.wav' -> 'RRMSAvaikiP24' """
    return os.path.splitext(wav_filename)[0]


# ---------- build the widgets -------------------------------
label_style = {"description_width": "175px"}
row_layout  = widgets.Layout(width="600px")

# Installation dropdown FIRST — everything below cascades from it.
_inst_options = list_installations()
install_dd = widgets.Dropdown(
    description="Installation folder:", options=_inst_options,
    value=(DEFAULT_INSTALLATION_FOLDER if DEFAULT_INSTALLATION_FOLDER in _inst_options
           else (_inst_options[0] if _inst_options else None)),
    style=label_style, layout=row_layout)

# Sandbox dropdowns start empty; populate_sandboxes() fills them
# once we know which installation folder is selected.
source_dd = widgets.Dropdown(description="Your sandbox (files from):",
                             options=[], style=label_style, layout=row_layout)
dest_dd   = widgets.Dropdown(description="Send output to:",
                             options=[], style=label_style, layout=row_layout)

tab_dd = widgets.Dropdown(description="Tab file (.txt):",
                          style=label_style, layout=row_layout)
wav_dd = widgets.Dropdown(description="Audio file (.wav):",
                          style=label_style, layout=row_layout)

speaker_tb = widgets.Text(description="Speaker code:",
                          placeholder="auto-filled from .txt name",
                          style=label_style, layout=row_layout)
prefix_tb  = widgets.Text(description="Audio prefix:",
                          placeholder="auto-filled from .wav name",
                          style=label_style, layout=row_layout)

gender_dd = widgets.Dropdown(
    description="Speaker gender:",
    options=[("female (f)", "f"), ("male (m)", "m"), ("other (x)", "x")],
    style=label_style, layout=row_layout)

url_tb = widgets.Text(description="Sheet URL:",
                      placeholder="auto-detected from the sheet in this folder",
                      style=label_style, layout=widgets.Layout(width="600px"))

refresh_btn = widgets.Button(description="↻ Refresh lists",
                             layout=widgets.Layout(width="180px"))
confirm_btn = widgets.Button(description="✅ Confirm selections",
                             button_style="success",
                             layout=widgets.Layout(width="200px"))
sheet_btn   = widgets.Button(description="🔗 Find Sheet URL",
                             tooltip="Look the sheet up in Google Drive "
                                     "(asks to authorize the first time)",
                             layout=widgets.Layout(width="180px"))
status_out  = widgets.Output()
folder_lbl  = widgets.HTML()


# ---------- keep the dropdowns in sync ----------------------
def populate_sandboxes(*_):
    """Refill the sandbox dropdowns from the selected installation folder."""
    sbs = list_sandboxes()
    default_sb = ("sandbox-user" if "sandbox-user" in sbs
                  else (sbs[0] if sbs else None))
    source_dd.options = sbs
    dest_dd.options = sbs
    source_dd.value = default_sb       # triggers refresh_files via observer
    dest_dd.value = default_sb
    refresh_files()                    # in case the value didn't change

def refresh_files(*_):
    sb = source_dd.value
    if not sb:
        tab_dd.options = []
        wav_dd.options = []
        folder_lbl.value = (
            f"<span style='color:#c0392b'>⚠️ No sandbox folders found in "
            f"<code>{os.path.join(MYDRIVE_ROOT, str(install_dd.value))}</code>."
            f" Pick a different installation folder above.</span>")
        return

    folder = resolve_elan_folder(sb)
    tabs = list_files(sb, ".txt")
    wavs = list_files(sb, ".wav")
    tab_dd.options = tabs
    wav_dd.options = wavs

    if folder and os.path.isdir(folder) and (tabs or wavs):
        folder_lbl.value = (
            f"<span style='color:#1a7f37'>📂 Reading from "
            f"<code>{folder}</code> — found {len(tabs)} .txt, "
            f"{len(wavs)} .wav</span>")
    else:
        folder_lbl.value = (
            f"<span style='color:#c0392b'>⚠️ No .txt/.wav found. Looked in "
            f"<code>{folder}</code>.<br>Check that this sandbox holds your "
            f"files, or adjust <code>ELAN_SUBFOLDER</code> at the top of the "
            f"cell, then click ↻ Refresh.</span>")

    if tab_dd.value:
        speaker_tb.value = derive_speaker_code(tab_dd.value)
    if wav_dd.value:
        prefix_tb.value = derive_prefix(wav_dd.value)

def on_tab_change(change):
    if change["new"]:
        speaker_tb.value = derive_speaker_code(change["new"])

def on_wav_change(change):
    if change["new"]:
        prefix_tb.value = derive_prefix(change["new"])

def on_dest_change(change):
    url_tb.value = sheet_url_for(change["new"])

def on_find_sheet(_):
    with status_out:
        status_out.clear_output(wait=True)
        sb = dest_dd.value
        if not sb:
            print("⚠️ Pick a destination sandbox first.")
            return
        url = sheet_url_for(sb)                         # try the filesystem
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
source_dd.observe(refresh_files, names="value")
tab_dd.observe(on_tab_change, names="value")
wav_dd.observe(on_wav_change, names="value")
dest_dd.observe(on_dest_change, names="value")
refresh_btn.on_click(lambda _: (populate_sandboxes(), refresh_files()))
sheet_btn.on_click(on_find_sheet)


# ---------- write the variables the notebook expects --------
def apply_selections():
    global destinationSandbox, installationFolder, nameTabFile
    global speakerCode, bigWavFile, prefixSmallAudioFiles
    global speakerGender, urlSandbox
    destinationSandbox    = dest_dd.value
    installationFolder    = install_dd.value
    nameTabFile           = tab_dd.value or ""
    speakerCode           = speaker_tb.value.strip()
    bigWavFile            = wav_dd.value or ""
    prefixSmallAudioFiles = prefix_tb.value.strip()
    speakerGender         = gender_dd.value
    urlSandbox            = url_tb.value.strip()

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
        if not nameTabFile:
            problems.append("No .txt file selected.")
        if not bigWavFile:
            problems.append("No .wav file selected.")
        if not speakerCode:
            problems.append("Speaker code is empty.")
        if not prefixSmallAudioFiles:
            problems.append("Audio prefix is empty.")
        if speakerGender not in ("m", "f", "x"):
            problems.append("Gender must be m, f or x.")

        if problems:
            print("⚠️  Please fix:")
            for p in problems:
                print("   • " + p)
            return

        # gentle, non-blocking convention check
        expected_tab = f"{prefixSmallAudioFiles}-{speakerCode.lower()}.txt"
        note = ""
        if nameTabFile != expected_tab:
            note = (f"\n   note: tab file is '{nameTabFile}', "
                    f"convention would expect '{expected_tab}' — "
                    f"fine if intentional.")

        print("✅  All set. The notebook will use:")
        print(f"   installationFolder    = {installationFolder}")
        print(f"   destinationSandbox    = {destinationSandbox}")
        print(f"   nameTabFile           = {nameTabFile}")
        print(f"   speakerCode           = {speakerCode}")
        print(f"   bigWavFile            = {bigWavFile}")
        print(f"   prefixSmallAudioFiles = {prefixSmallAudioFiles}")
        print(f"   speakerGender         = {speakerGender}")
        print(f"   urlSandbox            = {urlSandbox}" + note)

confirm_btn.on_click(on_confirm)


# ---------- show it -----------------------------------------
if not _inst_options:
    apply_selections()                            # avoid NameErrors downstream
    display(widgets.HTML(
        f"<b style='color:#c0392b'>No folders found in {MYDRIVE_ROOT}.</b> "
        f"Mount Google Drive (uncomment the mount line at the top) and/or fix "
        f"MYDRIVE_ROOT, then re-run this cell."))
else:
    populate_sandboxes()                          # fill sandboxes + files
    url_tb.value = sheet_url_for(dest_dd.value)    # auto-detect the Sheet URL
    apply_selections()                            # so later cells never NameError
    display(widgets.VBox([
        widgets.HTML("<b>Choose your files, then click Confirm.</b>"),
        install_dd,
        source_dd, folder_lbl, tab_dd, wav_dd,
        speaker_tb, prefix_tb, gender_dd,
        dest_dd,
        widgets.HBox([url_tb, sheet_btn]),
        widgets.HBox([refresh_btn, confirm_btn]),
        status_out,
    ]))


#==============================================
# Read in sandbox information
#==============================================

foldersToFindFiles = "/content/drive/MyDrive/" + installationFolder + "/" + destinationSandbox + "/processed-elan-files/"
folderWhereWavsAreStored = "/content/drive/MyDrive/" + installationFolder + "/" + destinationSandbox + "/wav/"
gsheetURL = urlSandbox

#==============================================
# Support functions
#==============================================

def countDigits(number):
  count=0
  while(number>0):
    count=count+1
    number=number//10
  return(count)

def addZerosToNumber(number, maxNumber):

  digitsMax = countDigits(maxNumber)
  digitsNumber = countDigits(number)

  zerosToAdd = digitsMax - digitsNumber
  zeros = ""

  for i in range(0,zerosToAdd):
    zeros += "0"

  retNum = zeros + str(number)
  return(retNum)

def reformatTranscription(input):

  output = input

  # Modify this function if you need to do extra editing
  # of your ELAN transcription (e.g. replacing special
  # characters or eliminating punctuation)

  punctuation = [ "[", "]", "\"", "(", ")", ".", "\u0f7b", "_", "|", "》", "?", "!", "/", ',', '-', '?', '<', '…', '>' ]

  for p in punctuation: output = output.replace(p, " ")
  for i in range(0,5): output = output.replace("  "," ")
  output = output.strip().lower()

  return(output)

#==============================================
# Read in tab-separated file
#==============================================

tabFile = open(foldersToFindFiles + nameTabFile, 'r')
tabFile = tabFile.readlines()

timeStart = []
timeEnd = []
transcriptions = []

for line in tabFile:

  line = line.replace("\n","")
  lineSplit = line.split("\t")
  print(lineSplit)

  transcriptionColumn = len(lineSplit)-1

  start = float(lineSplit[2])
  end = float(lineSplit[3])
  transcription = lineSplit[transcriptionColumn]
  duration = end - start

  if (duration > 0 and reformatTranscription(transcription) != ""):
    timeStart.append(lineSplit[2])
    timeEnd.append(lineSplit[3])
    transcriptions.append(lineSplit[transcriptionColumn])

print("I found " + str(len(transcriptions)) + " valid and non-empty annotations in the file " + nameTabFile)

#============================================================
# If the audio is in an MP3 file, convert it to a .wav file
#============================================================

if (".mp3" in bigWavFile):
  newaudioFile = bigWavFile.replace(".mp3", ".wav")
  command = "ffmpeg -y -i "
  command += foldersToFindFiles + bigWavFile
  command += " -acodec pcm_u8 -ar 22050 "
  command += foldersToFindFiles + newaudioFile

  !$command
  bigWavFile = newaudioFile

#============================================================
# Separate the big file into smaller wav files
#============================================================

filenames = []
print(len(timeStart))


points = []
for start, end in zip(timeStart, timeEnd):
  i = len(points)-1
  tempName = speakerCode + "-" + prefixSmallAudioFiles + "-" + addZerosToNumber(i+1,len(timeStart)) + ".wav"
  tempName2 = speakerCode + "-" + prefixSmallAudioFiles + "-" + addZerosToNumber(i+2,len(timeStart)) + ".wav"
  if points and points[-1] == start:
    points.append(end)
    filenames.append(tempName)
  else:
    points.append(start)
    points.append(end)
    filenames.append(tempName2)

pointSeq = ','.join(str(t) for t in points)
inFileName = foldersToFindFiles + bigWavFile
zeros = countDigits(len(timeStart))
outFileName = folderWhereWavsAreStored + speakerCode + '-' + prefixSmallAudioFiles + f'-%0{zeros}d.wav'
!ffmpeg -y -i "$inFileName" -f segment -ac 1 -ar 16000 -async 1 -segment_times $pointSeq $outFileName


#==================================================================
# Read GSheet as Pandas, add new rows, and rewrite the
# sandbox GSheet with the new Pandas dataframe.
#==================================================================

from gspread_dataframe import set_with_dataframe

wb = gc.open_by_url(gsheetURL)
sheet = wb.worksheet('wav-metadata')
rows = sheet.get_all_values()
sandboxRows = pd.DataFrame(rows)

rowsAtStart = len(sandboxRows.index)

today = date.today()
todaysDate = today.strftime("%Y%m%d")

for i in range(0,len(timeStart)):
  with contextlib.closing(wave.open(folderWhereWavsAreStored + filenames[i],'r')) as f:
      frames = f.getnframes()
      rate = f.getframerate()
      duration = frames / float(rate)
  inValues = [filenames[i], destinationSandbox.replace("sandbox-",""), todaysDate, speakerCode, speakerGender, os.path.getsize(folderWhereWavsAreStored + filenames[i]), duration, "", "", reformatTranscription(transcriptions[i]), transcriptions[i]]
  sandboxRows.loc[len(sandboxRows)] = inValues

set_with_dataframe(sheet, sandboxRows)  # Write all rows to sandbox GSheet
sheet.delete_rows(1,1)                     # Erase the pandas numerical headers
rowsAtEnd = len(sandboxRows.index)

print("Total rows before new data: " + str(rowsAtStart))
print("Total rows after new data:  " + str(rowsAtEnd))
print("Added rows:                 " + str(rowsAtEnd-rowsAtStart))