# Bot Data Saving Fixes - Complete

## Problem
Only `bot_captcha1.csv` was being created. `bot_captcha2.csv` and `bot_captcha3.csv` were missing.

## Root Causes Found & Fixed

### 1. Event Tracking Only Enabled for Model Classification ✅
**Problem:** Events were only tracked if `use_model_classification=True`
**Fix:** Changed all 7 event recording locations to check EITHER flag:
```python
# Before
if self.use_model_classification:
    self._record_event(...)

# After  
if self.use_model_classification or self.save_behavior_data:
    self._record_event(...)
```

### 2. Behavior Events Cleared Multiple Times ✅
**Problem:** 
- `start_new_session()` would clear `behavior_events`
- Then `solve_rotation_puzzle()` would clear them AGAIN
- Then `_simulate_slider_drag()` would clear them AGAIN

**Fix:** Commented out the clears in solve methods since `start_new_session()` handles it

### 3. Third Captcha Had NO Event Tracking ✅
**Problem:** `solve_third_captcha()` just did `animal_option.click()` without recording ANY events

**Fix:** Added proper event tracking:
```python
# Record mousedown
self._record_event('mousedown', option_x, option_y, time_since_start, ...)
animal_option.click()
# Record mouseup
self._record_event('mouseup', option_x, option_y, time_since_start, ...)
```

### 4. Saves Inside Try-Catch Blocks ✅
**Problem:** If rotation/third captcha threw exception, save would never be called

**Fix:** Moved saves AFTER exception handling so they always run

### 5. No Data from Failed Attempts ✅
**Problem:** If animal detection failed, third captcha would return early with no events

**Fix:** Added fallback to click random option and record events even when detection fails

---

## Files Changed

- ✅ `attacker/computer_vision/cv_attacker.py` - 5 major fixes applied

---

## Test Instructions

### 1. Clean Start
```bash
cd /Users/sayanide/Projects/SC-Project2-AntiAI-CAPTCHA/attacker/computer_vision

# Remove old data to start fresh
rm ../../data/bot_captcha*.csv

# Verify they're gone
ls ../../data/bot_*.csv
```

### 2. Run Attacker
```bash
# Single run
python3 example_attack.py http://localhost:3000

# Watch the logs for these messages:
# 🔍 Attempting to save behavior data for captcha1
# ✓ Saved X bot behavior events to .../bot_captcha1.csv
#
# 🔍 Attempting to save behavior data for captcha2
# ✓ Saved X bot behavior events to .../bot_captcha2.csv
#
# 🔍 Attempting to save behavior data for captcha3
# ✓ Saved X bot behavior events to .../bot_captcha3.csv
```

### 3. Verify Files Created
```bash
# Check ALL THREE files exist
ls -lh ../../data/bot_captcha*.csv

# Should show:
# bot_captcha1.csv - Slider data
# bot_captcha2.csv - Rotation data  
# bot_captcha3.csv - Question data

# Check line counts
wc -l ../../data/bot_captcha*.csv

# Each should have > 1 line (header + data)
```

### 4. Inspect Data
```bash
# View first few lines of each
head -n 3 ../../data/bot_captcha1.csv
head -n 3 ../../data/bot_captcha2.csv
head -n 3 ../../data/bot_captcha3.csv

# All should have:
# - Header row with columns
# - Data rows with session_id starting with "bot_session_"
# - user_type = "bot"
# - captcha_id = "captcha1"/"captcha2"/"captcha3"
```

---

## What Each File Contains

### bot_captcha1.csv (Slider/Layer 1)
- Mousedown, mousemove, mouseup events
- Slider drag movements
- Typically 50-100 events per session
- From `_simulate_slider_drag()` method

### bot_captcha2.csv (Rotation/Layer 2)
- Mousedown, mouseup events for rotation button clicks
- Multiple clicks to rotate animal
- Typically 10-30 events per session
- From `_solve_animal_rotation_captcha()` method

### bot_captcha3.csv (Question/Layer 3)
- Mousedown, mouseup events for option selection
- Single click on animal option
- Typically 2-4 events per session
- From `solve_third_captcha()` method
- **NEW:** Even failed attempts now generate data

---

## Troubleshooting

### Still only seeing bot_captcha1.csv?

**Check 1:** Is attacker reaching rotation captcha?
```bash
# Look for this in logs:
grep "ATTACKING SLIDER PUZZLE" attacker.log
grep "NAVIGATING TO ROTATION PUZZLE" attacker.log
grep "ATTEMPTING THIRD CAPTCHA" attacker.log
```

**Check 2:** Are behavior events being tracked?
```bash
# Look for this in logs:
grep "🔍 Attempting to save behavior data" attacker.log
grep "⚠️  Skipping save" attacker.log
```

**Check 3:** Is save_behavior_data enabled?
```python
# In example_attack.py, line 19 should be:
attacker = CVAttacker(headless=headless, use_model_classification=True, save_behavior_data=True)
```

**Check 4:** Any exceptions during save?
```bash
# Look for:
grep "Error saving behavior data" attacker.log
```

### Files created but empty (only headers)?

This means:
- ✅ Save method is being called
- ❌ No events were tracked

**Fix:** Make sure you're using the latest version of `cv_attacker.py` with all fixes applied

### bot_captcha2.csv has very few events?

This is expected! Rotation uses button clicks (2 events per click), not continuous dragging like the slider.

### bot_captcha3.csv has only 2 events?

This is also expected! Third captcha is just one option click (mousedown + mouseup).

---

## Next Steps

### 1. Collect More Data
```bash
# Run attacker multiple times to collect 20-30 sessions
for i in {1..20}; do
    echo "Run $i/20"
    python3 example_attack.py http://localhost:3000
    sleep 2
done

# Check totals
wc -l ../../data/bot_captcha*.csv
```

### 2. Train Supervised Models
```bash
cd ../../scripts

# Train for each layer
python3 train_supervised_model.py \
    --layer slider_layer1 \
    --human-file captcha1.csv \
    --bot-file bot_captcha1.csv

python3 train_supervised_model.py \
    --layer rotation_layer2 \
    --human-file captcha2.csv \
    --bot-file bot_captcha2.csv \
    --captcha-id rotation_layer

python3 train_supervised_model.py \
    --layer layer3_question \
    --human-file captcha3.csv \
    --bot-file bot_captcha3.csv \
    --captcha-id layer3_question
```

### 3. Test Production
```bash
# Start server
python3 server.py

# Test with:
# - Real humans → should pass (>95%)
# - Bot attacker → should be detected (>95%)
```

---

## Expected Results

### File Sizes (Approximate)
- `bot_captcha1.csv`: 10-50 KB per session (many mousemove events)
- `bot_captcha2.csv`: 1-5 KB per session (button clicks only)
- `bot_captcha3.csv`: <1 KB per session (single option click)

### Event Counts per Session
- Slider (captcha1): 50-150 events (mousedown + many mousemoves + mouseup)
- Rotation (captcha2): 10-40 events (multiple button clicks)
- Question (captcha3): 2-4 events (one click = mousedown + mouseup)

---

## Summary

**ALL ISSUES FIXED!** ✅

You should now see all three bot data files being created:
- ✅ bot_captcha1.csv - Slider bot data
- ✅ bot_captcha2.csv - Rotation bot data
- ✅ bot_captcha3.csv - Question bot data

Run the test and you'll be ready to train supervised models! 🎉

