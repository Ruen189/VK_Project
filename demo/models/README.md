# Model weights

After training, export binary weights (default float16):

```bash
cd tools/train
python export_web_weights.py --checkpoint ./checkpoints/model.pt --out ../../models/enhance_params.bin
# float32:  --dtype float32
# legacy JSON: --format json --out ../../models/enhance_params.json
```

Then `npm run build` copies `models/` → `demo/models/`.

In the demo, select **Модель TinyCNN**. Without this file the demo uses **эвристику** only.
