# Mint theme

The NiceGUI dashboard is black and mint, in the dark industrial layout: a top header and rounded charcoal cards.

## What changed

Colors live in one `THEME` block in the `sentinel.ui` section of `sentinel_gt_ai.py`. Backgrounds stay near-black and charcoal (`#070B14`, `#121A2B`). The accent is mint `#3EE6A8`, hover `#2BC991`, soft tint `#3EE6A81A`. Buttons, the active tab, chips, focus rings, links, and the headline chart series use that accent. Text on a mint button is near-black `#0B1220`.

Novel and alarms stay red. Warning stays yellow `#E6C15A`. Neither is the old amber.

The left drawer is gone. The header has the shield mark, SENTINEL GT AI, the tabs ANALYZE / TEACH / MODELS / SETTINGS, and `Local · {device} · model {version}`. Analyze is three columns on one 1440×900 screen: case input and the pipeline checklist, signature / confidence / novelty plus the tag charts, and the site report, nearest cases, and the review gate (Good / Not good / Edit). The full document opens in a dialog. The paper sheet stays light with dark ink. Teach, Models, and Settings use the same header and cards.

## Contrast

- Mint `#3EE6A8` on near-black `#0B1220` is about 12:1. Mint button labels use `#0B1220`.
- White on the brighter alarm red fails, so the Not-good button fill is `#9E2433` (about 7:1 with white). Novel and alarm text on charcoal uses `#FFB4B8`.
- Warning yellow `#E6C15A` on charcoal stays distinct from mint and from red.
- Mint on the paper report is about 1.5:1, so report links are `#0C6E4A` (about 6:1 on `#F4F6F2`).
- The chart residual window uses `MINT_BAND`, `rgba(62, 230, 168, 0.22)`. The exact token `#3EE6A81A` is `MINT_TINT`, used for chips and hovers. At about 10% alpha that tint is too faint to read as a band on charcoal.

## Left unchanged

The overview pages, the Streamlit chart palette, and `scripts/charts_svg.py` still use their own oranges. Those are not the NiceGUI dashboard.

## Checked

- `pytest -m "not slow"`: 357 passed, 2 skipped.
- The five screenshots in `docs/screenshots/` were retaken at 1440×900 from the synthetic file `cold_spot_01.csv`, in a temporary data directory. The Analyze page fits that viewport without the page itself scrolling.
- The commit is local. It was not pushed.

STATUS: COMPLETE
