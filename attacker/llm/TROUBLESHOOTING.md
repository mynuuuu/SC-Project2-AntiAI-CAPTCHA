# Troubleshooting Guide

## OpenAI Quota Error

### Error Message
```
Error code: 429 - {'error': {'message': 'You exceeded your current quota...', 'type': 'insufficient_quota'}}
```

### Solution
1. **Check your OpenAI account billing**:
   - Visit: https://platform.openai.com/account/billing
   - Add payment method if needed
   - Add credits to your account

2. **Verify API Key**:
   - Make sure your API key is valid
   - Check if the key has access to GPT-4o model
   - Some keys may only have access to certain models

3. **Alternative Models**:
   - Try `gpt-4-vision-preview` if `gpt-4o` is not available
   - Or use `gpt-4-turbo` (if available)

## Data Format Verification

The bot data is saved in the finalized format matching `bot_captcha1.csv`:
- **No `metadata_json` column** (removed to match finalized format)
- Standard columns: `session_id`, `timestamp`, `time_since_start`, `time_since_last_event`, `event_type`, `client_x`, `client_y`, etc.

## Files Generated

After running the attacker, bot data is saved to:
- `data/captcha1_bot.csv` (slider captcha 1)
- `data/captcha2_bot.csv` (slider captcha 2)  
- `data/captcha3_bot.csv` (slider captcha 3)
- `data/rotation_layer_bot.csv` (rotation captcha)
- `data/layer3_question_bot.csv` (animal question)

All files match the finalized format (no metadata_json column).

