# Privacy check

Date: 2026-10-16. Source branch `e2e-ai` at `d59c645`, plus the uncommitted privacy and doc edits on top of that commit. Scanner: ripgrep 15, `rg -uu` (`--hidden --no-ignore`) so gitignored logs and notes were included. `rg` skips binary files by default. Globs below skip `.git`, `.venv`, and `__pycache__`.

The live search strings are not repeated in this file. `tests/test_branding.py` builds them from pieces. Class names below stand in for those strings. Every zero-hit command adds `--glob !PRIVACY_CHECK.md`. The branding test excludes this file for the same reason.

## What was removed

Personal: a person's given name, family name, login, and email handle; email addresses; absolute home and app paths (`C:\Users\...`, `/c/Users/...`, `C--Users-...`); temp paths; a private local project path that was rewritten to a generic relative path.

Company and internal names, whole word or phrase, case-insensitive: the two-letter company abbreviation, the company pair, the full company name, the department abbreviation (plain and HTML), the department name, the control platform with and without its trailing letter, the vibration tool, the monitoring product, and the three-letter on-site tool.

Frame model names stay: 6B, 7EA, 7FA, 7HA, 9E, 9FA, 9HA. Historian tag fragments such as `TTXD_GE1x` stay. The underscore and the digit are word characters, so that fragment is not a whole-word company abbreviation. HETS and DLN stay. They are not on the list above.

## How text was rewritten

Two scratch scripts under `%TEMP%` (not in either repo) walked text files. Pass 1 skipped `.git`, `.venv`, `__pycache__`, `.scratch`, binaries, and any file whose first 8 KB contained a NUL (that hid UTF-16). Pass 2 decoded UTF-16 and covered `.scratch` text plus `*.err`.

Substitutions, in order: email addresses to `[redacted-email]`; this app's absolute path to a relative path; the private local project path to `local-models/`; temp paths to `%TEMP%`; home paths to `~`; the full name, initials, possessives, login, and email handle to "the engineer", "Reviewer", or "user"; company and department phrases to "OEM" before the short tokens. A quoted copy of the company-abbreviation regex in private notes was rewritten so the note no longer contained the live token.

Pass 1: 351 text files, 51 changed. Replacement counts:

| Token class | Count |
|---|---:|
| MSYS-style home path | 538 |
| Absolute app path | 510 |
| Login name | 304 |
| Temp path | 126 |
| the department abbreviation | 120 |
| Email address | 84 |
| Full name | 83 |
| Private project path | 81 |
| Home path | 69 |
| Assistant session path | 68 |
| Company abbreviation | 56 |
| Company pair | 55 |
| Given name | 36 |
| the department abbreviation as HTML | 30 |
| Private project name | 26 |
| the company pair | 25 |
| the full company name | 22 |
| Department symbol form | 21 |
| Monitoring and Diagnostics | 19 |
| Possessive name | 16 |
| Quoted department regex | 15 |
| App path without a slash | 12 |
| Initials | 5 |
| Quoted company regex | 2 |

Pass 1 files (path relative to the source root), with the classes that hit:

- `claude_peralert_review.log` — email 15, app path 217, app path without slash 3, temp 90, home 22, MSYS 256, session path 39, private project 4, possessive 1, given name 2, login 176, the full company name 4, the department abbreviation 6, the company pair 6
- `claude_run.log` — given name 1
- `claude_v2.log` — email 65, app path 281, app path without slash 7, private project path 12, temp 31, home 46, MSYS 282, session path 29, private project name 17, full name 72, initials 4, possessive 10, given name 13, login 128, company pair 51, the full company name 16, the department abbreviation HTML 28, the department abbreviation 107, Monitoring and Diagnostics 16, department symbol 21, the company pair 16, company abbreviation 54, quoted department regex 13
- `claude_v2c_prompt.txt` — email 1, full name 1, given name 1, the full company name 1, the department abbreviation HTML 1, the department abbreviation 1, the company pair 1, quoted company regex 1, quoted department regex 1
- `claude_v2_prompt.txt` — email 1, app path without slash 1, full name 1
- `DESIGN.md` — private project name 1, full name 1
- `DESIGN_DYNAMIC.md` — full name 1
- `DESIGN_E2E.md` — full name 1
- `DYNAMIC_STATUS.md` — possessive 1, given name 3
- `E2E_BRIEF.md` — private project path 1, home 1, full name 1, given name 8
- `E2E_BRIEF_V2.md` — email 1, private project path 2, full name 2, possessive 3, given name 4, company pair 1, the full company name 1, the department abbreviation HTML 1, the department abbreviation 2, the company pair 1, quoted company regex 1, quoted department regex 1
- `grok_dynamic.log` — the department abbreviation 1, the company pair 1, company abbreviation 1
- `grok_v2.log` — full name 1
- `grok_v2_prompt.txt` — email 1, app path without slash 1, full name 1
- `PROGRESS.md` — given name 2
- `PROGRESS_V2.md` — private project name 1, given name 1, company abbreviation 1
- `README.md` — initials 1 (the public README was then rewritten)
- `REVIEW_V2.md` — possessive 1
- `SENTINEL_BRIEF.md` — private project path 2, private project name 1, full name 1, given name 1
- `data/cases/SGT-20261012-155038-56a4/llm_prompt.txt` — company pair 1, the department abbreviation 1, Monitoring and Diagnostics 1
- `data/cases/SGT-20261012-155038-56a4/report.json` — private project path 4
- `data/cases/SGT-20261012-155329-092b/llm_prompt.txt` — company pair 1, the department abbreviation 1, Monitoring and Diagnostics 1
- `data/cases/SGT-20261012-155329-092b/report.json` — private project path 4
- `data/cases/SGT-20261012-155601-a27f/llm_prompt.txt` — company pair 1, the department abbreviation 1, Monitoring and Diagnostics 1
- `data/cases/SGT-20261012-155601-a27f/report.json` — private project path 4
- `data/cases/SGT-20261012-163211-cbf7/report.json` — app path 1, private project path 4
- `data/e2e/registry/v0001/encoder/training.json` — app path 1
- `data/e2e/registry/v0002/encoder/training.json` — app path 1
- `docs/sentinel_overview.html` — private project name 1
- `eval/EVAL_DYNAMIC.md` — app path 1
- `eval/results_dynamic.json` — app path 1
- `eval/results_peralert.json` — app path 1
- `eval/examples/sensorfault-tc_sensor_fault-step_lookalike-000.json` — app path 1, temp 1
- `eval/examples/test-cold_spot_hardware-base-000.json` — app path 1, temp 1
- `eval/examples/test-combustion_instability-base-000.json` — app path 1, temp 1
- `eval/examples/test-dynamics_probe_fault-base-002.json` — app path 1, temp 1
- `eval/examples/test-tc_sensor_fault-base-000.json` — app path 1, temp 1
- `models/e2e/legacy/encoder/training.json` — app path 1, now `models\tspulse\clf`
- `scripts/overview_template.html` — private project name 1
- `validation/cold_spot_01_report.json` and the combustion and high-exhaust-spread copies, plus the same three names under `validation/round1`, `round2`, and `round3` — private project path 4 each (12 files)

Pass 2 (UTF-16 decoded, then written back as UTF-8, and `.scratch` / `.err` text). Replacement counts:

- `claude_e2e.log` — 2125 (was UTF-16 LE, so pass 1 skipped it)
- `.scratch/ui.err` — 2575
- `.scratch/v2/patch_brand.py` — 26
- `.scratch/review/scan.py` — 8
- `.scratch/install_test.log` — 6
- `.scratch/v2/brand_scan.py` — 6
- `.scratch/ovdom.html` — 5
- `.scratch/v2/pytest1.log` — 4
- `.scratch/v2/gemma_smoke_case/llm_prompt.txt` — 3
- `.scratch/cdp_ui.py` — 2
- `.scratch/v2/gemma_smoke_case/report.json` — 2
- `verify/eval.txt` (under `.scratch`, UTF-16) — 2
- one each: `.scratch/doctor_auto.txt`, `.scratch/doctor_gpu.txt`, `.scratch/dom.html`, `.scratch/ft.log`, `.scratch/gen_append.py`, `.scratch/overview.log`, `.scratch/tsp_speed.py`, `.scratch/review/progress.py`, `.scratch/v2/baseline_eval.log`, `.scratch/verify/doctor.txt`, `.scratch/verify/run.ps1`, `.scratch/train_final.err`
- `claude_run2.log` is UTF-16 and had nothing to replace

Removed, not redacted: `.scratch/edgeprof`, `.scratch/edgeprof2`, `.scratch/edgeprof3`. Those were browser profiles. Caches stored the home path in binary files that cannot be edited as text.

Hand edits after the scripts, so the shipped wording is clean:

- `tests/test_branding.py` builds the banned vocabulary from pieces. The identifier that held the three-letter tool name was `osm`, which the case-insensitive whole-word scan matched. It is now `onsite`.
- `tests/test_feedback.py` reviewer string is `Reviewer`.
- `tests/test_single_file.py` and `scripts/build_single_file.py` say "an engineer".
- `scripts/overview_template.html` and `docs/sentinel_overview.html`: relative install path, reviewer label `Reviewer`, and "earlier prototype" where a private project name had been rewritten into an awkward phrase.
- `README.md`, `docs/INSTALL.md`, `docs/USER_GUIDE.md`, `docs/ARCHITECTURE.md`, `CHANGELOG.md`, and `docs/GITHUB_ABOUT.txt` are the public docs.
- `.gitignore` ignores logs, prompt files, progress, design, review, fix notes, `DYNAMIC_STATUS.md`, briefs, `.venv`, `data/*` (except `data/.gitkeep`), `settings.local.json`, model weights (except `models/README.md` and `models/e2e/`), `.scratch`, and `knowledge/user/`.

## Reviewed and left as non-findings

- Phone-number pattern: no hits.
- A serial-number word scan: no hits.
- Extra tool names checked in pass 1 (ToolboxST, ControlST, Predix, Bently, System 1, and the same family): no hits.
- Filenames: no hits for the personal tokens, the company pair, or the monitoring product.
- Three synthetic `csv.gz` rows matched the given name or the family name as compressed bytes. Decompressed text was clean. Those datasets are not in the public copy.
- Gemma GGUF: the bytes `gmail` are tokenizer vocabulary (`gmail` / `Gmail`), not an address. Qwen GGUF: one weight-byte sequence in the Qwen file looks like the login when case is ignored. It is not the login. Both weight files are excluded (multi-GB). TS Pulse checkpoints are excluded too. `models/e2e` is 18.6 MB and is included.
- `.venv` is third-party packages and is excluded from the scan and from the public copy.
- Screenshots under `.scratch` are binary PNGs. `rg` does not OCR them. They are not copied.
- `DEPLOY.md` names a laptop model. That model is not on the ban list. The file stays in the private source and is not copied. `docs/INSTALL.md` is the public install guide.
- NTFS owner metadata is the local account. Git does not store it.
- No `LICENSE` file was present to keep.

## Commands

Run from the source root. `rg` exit code 1 means zero matches.

```powershell
rg -uu -i --count-matches --glob "!.git/**" --glob "!.venv/**" --glob "!**/__pycache__/**" --glob "!PRIVACY_CHECK.md" "\b(the given name|the family name|the-login|the-email-handle)\b" .
rg -uu -i --count-matches --glob "!.git/**" --glob "!.venv/**" --glob "!**/__pycache__/**" --glob "!PRIVACY_CHECK.md" "[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}" .
rg -uu -i --count-matches --glob "!.git/**" --glob "!.venv/**" --glob "!**/__pycache__/**" --glob "!PRIVACY_CHECK.md" "Users[/\\]the-login|C:[/\\]+Users[/\\]+the-login|/c/Users/the-login" .
rg -uu -i --count-matches --glob "!.git/**" --glob "!.venv/**" --glob "!**/__pycache__/**" --glob "!PRIVACY_CHECK.md" "\bGE\b|the company pair|the full company name|the department abbreviation|M&amp;D|Monitoring\s*(&|and)\s*Diagnostics|Mark\s*VIe?\b|System\s*One|the monitoring product|\bOSM\b" .
rg -uu --count-matches --glob "!.git/**" --glob "!.venv/**" --glob "!**/__pycache__/**" --glob "!PRIVACY_CHECK.md" "\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]\d{4}\b" .
rg -uu --files --glob "!.git/**" --glob "!.venv/**" --glob "!**/__pycache__/**" . | rg -i "the-given-name|the-family-name|the-login|the-email-handle|company-pair|monitoring-product"
```

## Source tree result

Recorded after this file existed, with `PRIVACY_CHECK.md` excluded:

<!-- ZERO_HIT_START -->
Run from the source root on 2026-10-16, `PRIVACY_CHECK.md` excluded:

```
===== PERSON =====
ZERO
===== EMAIL =====
ZERO
===== PATH =====
ZERO
===== OEM =====
ZERO
===== PHONE =====
ZERO
===== FILENAMES =====
ZERO
```

`rg` exit code was 1 for each content search (zero matches). The same person pattern without the checklist exclusion lists only `PRIVACY_CHECK.md`.
<!-- ZERO_HIT_END -->

An earlier draft of this file quoted the live tokens, so a scan that did not exclude it matched this file only. Those tokens are not repeated here.

## Public copy

`sentinel_gt_ai_v2` is rebuilt from this tree. It receives code, tests, scripts, knowledge, docs, small synthetic samples, `models/e2e`, `models/README.md`, install files, and this checklist.

It also receives two fixtures the tests and the overview builder read, both already clean:

- `eval/acceptance_peralert.jsonl` — 320 synthetic held-out ids, content hashes, frames, and alert names. No plant export.
- `validation/` — the three sample reports (and the earlier rounds) plus `VALIDATION.md`. Absolute paths in those JSON files were rewritten in the passes above.

`DESIGN.md` is clean of the tokens in this checklist. `.gitignore` still ignores `DESIGN*.md` at the repo root. A copy is at `docs/dev/DESIGN.md` (the ignore rule is negated for that path) so the overview builder can embed the decision tables. `scripts/build_overview.py` reads `DESIGN.md` when that file is present, and `docs/dev/DESIGN.md` otherwise. The heading of section 8 says "earlier prototype".

It does not receive logs, prompt files, `PROGRESS_*`, root `DESIGN*`, `REVIEW_*`, `FIX_*`, `DYNAMIC_STATUS.md`, briefs, `DEPLOY.md`, the rest of `eval/` (result JSON and write-ups), `data/cases`, `data/e2e`, `.venv`, `.scratch`, `settings.local.json`, Gemma or Qwen weights, or TS Pulse checkpoints. Case labels from local work stay on this machine. `data/` in the public tree is `data/.gitkeep` only.

The local pytest run used a gitignored hard link of the Gemma file, a gitignored copy of the TS Pulse snapshots, and `llama-cpp-python` 0.3.36 from the CPU wheel index. None of those weights are in the commit. `install.ps1` is what a new laptop uses to download them.

## Public git history

History was not rewritten.

Before this update, `git log -p --all` in `sentinel_gt_ai_v2` (HEAD `77e4c8d`):

| Pattern | Matching lines |
|---|---:|
| Person name / login | 8 |
| Email | 23 |
| Home path | 1 |
| Company / department / tool names | 2 |

`git log --format` authors, both commits: the personal name and email the patterns above match (`77e4c8d`, `e55f3e8`).

The staged patch deletes lines from that old tree (a name in the old readme and overview, reviewer strings in two tests, one home path in the overview, and the old branding-test vocabulary) and adds this file, which quotes the commands. A diff marker in front of `@pytest` can satisfy the email pattern; the working tree does not. History was not rewritten.

<!-- HISTORY_AFTER_START -->
`git commit` in the public repo was not created. Exit 128: that repo has no `user.name` or `user.email`, auto-detect refused, and this pass did not set an identity. HEAD is still `77e4c8d`. The index is staged. `git log -p --all` is unchanged: person 8, email 23, home path 1, company/tool 2.

Staged patch against `77e4c8d`: person 11, email 11, home path 2, company/tool 21. After a commit whose author line does not match, `git log -p --all` would be person 19, email 34, home path 3, company/tool 23.

The email hits inside the staged patch are the search command in this file, plus diff lines whose leading marker makes a pytest decorator look like an address. The working-tree email scan is zero once this file is excluded. The new person and company/tool hits are the commands in this file. The deleted lines are the old public tree.
<!-- HISTORY_AFTER_END -->
