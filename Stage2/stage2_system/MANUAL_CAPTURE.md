# Manual capture and DeepSeek analysis

In the existing desktop UI, press **Capture & Analyze** under the left camera.
The single/left camera is captured; the same JPEG is shown in **Last Processed
Frame** and sent to DeepSeek. The bottom panel shows:

```text
Result — image-based estimate:
Cup, height: 10 cm, color: blue
```

The prompt is: “Let me know the estimated height, object color and the object
on top of the box here.” Height excludes the supporting box. No reference box
dimensions are assumed. Height can be `unknown` when the image cannot support
an estimate; this is not stereo/depth measurement.

## Run on Windows

From the repository root in PowerShell:

```powershell
python -m pip install -r Stage2/stage2_system/requirements-laptop.txt
$env:DEEPSEEK_API_KEY = Read-Host "DeepSeek API key" -MaskInput
python Stage2/stage2_system/laptop/main.py --single-cam 0
```

`-MaskInput` requires PowerShell 7.1+. In Windows PowerShell 5.1, use:

```powershell
$secret = Read-Host "DeepSeek API key" -AsSecureString
$env:DEEPSEEK_API_KEY = [System.Net.NetworkCredential]::new("", $secret).Password
python Stage2/stage2_system/laptop/main.py --single-cam 0
```

Use your actual camera index. The key stays in this shell's environment; do not
commit it or paste it into a Python file. Launch from this same terminal so the
app inherits it. No key is bundled with this feature.

The manual button needs the camera and internet, but does not require the VM or
SSH size service to be connected and does not send navigation commands. The
existing main app still initializes its usual map and robot-link components.
Do not pass `--mock-llm` for a real DeepSeek request.

For an offline demo, use the existing mock flags:

```powershell
python Stage2/stage2_system/laptop/main.py --mock-camera --mock-llm --mock-size
```

Demo results are labelled `[DEMO]`. A physical camera failure that makes the
existing manager fall back to a synthetic feed is rejected for live uploads.

## Behavior

- Uses `deepseek-flash` at `https://api.deepseek.com/chat/completions` with one
  base64 JPEG, JSON output and thinking disabled. Existing `requests` and
  Pydantic dependencies suffice; no extra provider SDK is required.
- One click makes at most one API request. Connect/read timeouts and API or
  parsing errors are reported in the bottom panel. The preview stays responsive.
- Captures older than two seconds are rejected. The button is disabled during
  a level or another manual request. Conflicting GUI perception/level commands
  wait until manual analysis has finished. Navigation Stop remains available.
- **Show level results** restores the existing bottom-panel level result.
  Manual analysis does not write into the Level 1–3 results dictionary.
- The image and structured answer are saved under
  `Stage2/stage2_system/runs/manual_captures/` with unique filenames.
- Closing the GUI prevents further UI updates from the worker. An already sent
  API request may finish; closing does not revoke an upload or its billing.
- The Level 1–3 API adapter remains unchanged (currently a stub). The new
  DeepSeek client is isolated to manual capture; `Ping LLM` still describes the
  existing level adapter, not this new client.

## Review and tests

New logic is in `laptop/manual_capture.py` and `laptop/llm/object_analysis.py`.
`laptop/gui.py` supplies the button and main-thread result updates; `camera.py`
adds a recent-frame snapshot accessor. `mission.py`, Stage 1, and robot code
are unchanged.

```powershell
python -m unittest discover -s Stage2/stage2_system/tests -p test_manual_capture.py -v
```

These are offline tests; no real key, camera, or robot is used.

Provider references:
[DeepSeek vision](https://api-docs.deepseek.com/guides/vision/),
[JSON output](https://api-docs.deepseek.com/guides/json_mode/),
[thinking toggle](https://api-docs.deepseek.com/guides/thinking_mode/).
