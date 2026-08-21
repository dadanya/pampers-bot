# Qwen Audio 3.0 Voice Trial Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Сгенерировать один OGG/Opus-образец Qwen Audio 3.0 с существующим голосом ДимаПамперс без изменения работающего бота.

**Architecture:** Проба вызывает подтверждённый Higgsfield job type `qwen_audio_tts` с совместимым голосом, скачивает единственный результат в папку артефактов и проверяет аудиопоток `ffprobe`. Рабочий `voice.py` и запущенный polling остаются на Minimax.

**Tech Stack:** Higgsfield CLI, PowerShell, Qwen Audio 3.0 TTS Flash, ffprobe, OGG/Opus.

## Global Constraints

- Ровно один синтез: job type `qwen_audio_tts`, модель управляется этим типом задания, текст `Проверка голоса Дима Памперс`, `voice_id=f40ad9f4-947e-42a4-869d-96e2ef7cb23d`, `voice_type=element`, `format=ogg_opus`, `sample_rate=48000`.
- Не изменять `voice.py`, `bot.py`, `.env`, `memory.db`, историю, persona-файлы или процессы.
- В артефактах допустим только `artifacts/voice-trials/qwen-audio-dimapampers.ogg`.
- До отдельного одобрения качество рабочая переменная VOICE_MODEL остаётся `minimax`.
- Не публиковать секреты, API-ключи, внутренние URL или чужие сообщения.
- `E:\pampers-bot` не является Git-репозиторием: контрольные пункты завершаются проверками без коммита.

---

### Task 1: Создать и проверить единственный Qwen-образец

**Files:**
- Create at runtime: `artifacts/voice-trials/qwen-audio-dimapampers.ogg`
- Verify unchanged: `voice.py`, `bot.py`, `.env`, `memory.db`

**Interfaces:**
- Consumes: existing Higgsfield element voice `f40ad9f4-947e-42a4-869d-96e2ef7cb23d`, `higgsfield.cmd generate create qwen_audio_tts`.
- Produces: one OGG/Opus sample or a non-secret error; no production behavior changes.

- [ ] **Step 1: Record read-only baseline and remove no files**

Run:

```powershell
$files = @('voice.py', 'bot.py', '.env')
Get-FileHash -Algorithm SHA256 $files | Select-Object Path, Hash
Get-CimInstance Win32_Process |
  Where-Object {
    $_.Name -match '^python(?:w)?\\.exe$' -and
    $_.CommandLine -match '(^|[\\\\/\\s\\"])(bot\\.py)(\\s|\\"|$)'
  } |
  Select-Object ProcessId, ParentProcessId
Test-Path -LiteralPath 'E:\pampers-bot\artifacts\voice-trials\qwen-audio-dimapampers.ogg'
```

Expected: hashes and one root bot chain are recorded; the target artifact does not already exist.

- [ ] **Step 2: Create the trial directory**

Run:

```powershell
New-Item -ItemType Directory -Force -Path 'E:\pampers-bot\artifacts\voice-trials' | Out-Null
```

Expected: only the artifact directory is created if missing.

- [ ] **Step 3: Run the exact one Qwen synthesis and download its result**

Run:

```powershell
$trialDir = 'E:\pampers-bot\artifacts\voice-trials'
$raw = higgsfield.cmd generate create qwen_audio_tts `
  --prompt 'Проверка голоса Дима Памперс' `
  --voice_id 'f40ad9f4-947e-42a4-869d-96e2ef7cb23d' `
  --voice_type element `
  --format ogg_opus `
  --sample_rate 48000 `
  --wait --json
if ($LASTEXITCODE -ne 0) { throw 'Qwen Audio trial did not complete' }
$jobs = $raw | ConvertFrom-Json
$resultUrl = $jobs[0].result_url
if ([string]::IsNullOrWhiteSpace($resultUrl)) { throw 'Qwen Audio returned no result URL' }
$ogg = Join-Path $trialDir 'qwen-audio-dimapampers.ogg'
Invoke-WebRequest -Uri $resultUrl -OutFile $ogg
if ((Get-Item -LiteralPath $ogg).Length -le 0) { throw 'Qwen Audio returned an empty artifact' }
```

Expected: one non-empty OGG artifact and no other generation call.

- [ ] **Step 4: Verify OGG/Opus 48 kHz mono**

Run:

```powershell
$ogg = 'E:\pampers-bot\artifacts\voice-trials\qwen-audio-dimapampers.ogg'
$probe = ffprobe -v error -select_streams a:0 `
  -show_entries stream=codec_name,sample_rate,channels `
  -of default=noprint_wrappers=1 $ogg
if (($probe -notmatch 'codec_name=opus') -or
    ($probe -notmatch 'sample_rate=48000') -or
    ($probe -notmatch 'channels=1')) {
  throw 'Qwen artifact is not OGG/Opus 48 kHz mono'
}
Get-Item -LiteralPath $ogg | Select-Object FullName, Length, LastWriteTime
```

Expected: codec `opus`, sample rate `48000`, one channel.

- [ ] **Step 5: Recheck unchanged production state and hand off the sample**

Run the Step 1 hashes and process check again. Expected: identical hashes and unchanged bot process chain. Present the local OGG file to the user; state that production remains on Minimax and wait for separate approval before any switch.
