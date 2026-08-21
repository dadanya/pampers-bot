# CozyVoice Compatibility Trial Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Создать один изолированный OGG-образец голоса ДимаПамперс через `cozy_voice` и не менять работающего бота.

**Architecture:** Проба использует существующую CLI-команду Higgsfield напрямую с текущими `VOICE_ID` и `voice_type=element`. Внешний результат скачивается в отдельную папку артефактов и преобразуется `ffmpeg` в OGG/Opus для прослушивания; production-код и конфигурация не редактируются.

**Tech Stack:** Higgsfield CLI, PowerShell, ffmpeg/ffprobe, Telegram-совместимый OGG/Opus.

## Global Constraints

- Выполнить ровно один успешный запрос генерации с `--model cozy_voice`, `--voice_id f40ad9f4-947e-42a4-869d-96e2ef7cb23d`, `--voice_type element` и текстом `Проверка голоса Дима Памперс`. Не передавать параметры `format` и `language_boost`: CLI подтверждает, что они поддерживаются только для `minimax`.
- Не изменять `voice.py`, `bot.py`, `.env`, `memory.db`, историю, persona-файлы или запущенный процесс бота.
- Не печатать в ответах или журналах секреты, токены и переменные окружения.
- Если запрос не поддерживает текущий голос, сообщить точную несекретную ошибку и не переключать модель.
- Итоговый файл должен быть OGG с Opus, 48 kHz, mono; только после пользовательского одобрения качества допускается отдельная задача переключения `VOICE_MODEL`.
- `E:\pampers-bot` не является Git-репозиторием: контрольные пункты фиксируются командами и списком файлов, без коммитов.

---

### Task 1: Изолированная проба CozyVoice

**Files:**
- Create at runtime: `artifacts/voice-trials/cozy-voice-dimapampers.mp3`
- Create at runtime: `artifacts/voice-trials/cozy-voice-dimapampers.ogg`
- Verify unchanged: `voice.py`, `bot.py`, `.env`, `memory.db`

**Interfaces:**
- Consumes: `voice.py` constants `VOICE_ID = "f40ad9f4-947e-42a4-869d-96e2ef7cb23d"`, `voice_type = "element"`, current bot process chain.
- Produces: OGG/Opus file for user listening, or a non-secret incompatibility/error result.

- [ ] **Step 1: Capture a read-only baseline of production state**

Run:

```powershell
$files = @('voice.py', 'bot.py', '.env', 'memory.db')
Get-FileHash -Algorithm SHA256 $files | Select-Object Path, Hash
Get-CimInstance Win32_Process |
  Where-Object {
    $_.Name -match '^python(?:w)?\\.exe$' -and
    $_.CommandLine -match '(^|[\\\\/\\s\\"])(bot\\.py)(\\s|\\"|$)'
  } |
  Select-Object ProcessId, ParentProcessId, CommandLine
```

Expected: hashes and exactly one root bot process chain are recorded; no process is stopped.

- [ ] **Step 2: Create the separate artifact directory**

Run:

```powershell
$trialDir = 'E:\pampers-bot\artifacts\voice-trials'
New-Item -ItemType Directory -Force -Path $trialDir | Out-Null
```

Expected: only `E:\pampers-bot\artifacts\voice-trials` is created if absent.

- [ ] **Step 3: Run exactly one CozyVoice generation**

Run:

```powershell
$trialDir = 'E:\pampers-bot\artifacts\voice-trials'
$raw = higgsfield.cmd generate create text2speech_v2 `
  --prompt 'Проверка голоса Дима Памперс' `
  --voice_id 'f40ad9f4-947e-42a4-869d-96e2ef7cb23d' `
  --voice_type element `
  --model cozy_voice `
  --wait --json
if ($LASTEXITCODE -ne 0) { throw 'CozyVoice trial failed before an artifact was created' }
$jobs = $raw | ConvertFrom-Json
$resultUrl = $jobs[0].result_url
if ([string]::IsNullOrWhiteSpace($resultUrl)) { throw 'CozyVoice returned no result URL' }
Invoke-WebRequest -Uri $resultUrl -OutFile (Join-Path $trialDir 'cozy-voice-dimapampers.mp3')
```

Expected: one MP3 artifact exists and has a nonzero byte length. No `voice.py` setting changes.

- [ ] **Step 4: Convert and verify the Telegram-ready sample**

Run:

```powershell
$trialDir = 'E:\pampers-bot\artifacts\voice-trials'
$mp3 = Join-Path $trialDir 'cozy-voice-dimapampers.mp3'
$ogg = Join-Path $trialDir 'cozy-voice-dimapampers.ogg'
ffmpeg -y -i $mp3 -c:a libopus -b:a 32k -ar 48000 -ac 1 $ogg
if ($LASTEXITCODE -ne 0) { throw 'OGG conversion failed' }
$probe = ffprobe -v error -select_streams a:0 `
  -show_entries stream=codec_name,sample_rate,channels `
  -of default=noprint_wrappers=1 $ogg
if (($probe -notmatch 'codec_name=opus') -or
    ($probe -notmatch 'sample_rate=48000') -or
    ($probe -notmatch 'channels=1')) {
  throw 'Converted file is not Telegram-ready OGG/Opus 48 kHz mono'
}
Get-Item -LiteralPath $ogg | Select-Object FullName, Length, LastWriteTime
```

Expected: non-empty `cozy-voice-dimapampers.ogg`, encoded Opus at 48 kHz mono.

- [ ] **Step 5: Confirm that the production bot was untouched**

Run the Step 1 hash and process commands again.

Expected: all four hashes are identical to the baseline; the same root bot chain remains running. The only new files are the MP3 and OGG trial artifacts.

- [ ] **Step 6: Present the OGG and wait for a separate model-switch approval**

Send the resulting `cozy-voice-dimapampers.ogg` to the user. Report whether the current Higgsfield voice ID was accepted, the exact codec check, and that VOICE_MODEL remains `minimax`. Do not edit `voice.py` or restart the bot in this task.
