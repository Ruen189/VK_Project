# Model weights

After training:

```bash
cd tools/train
python export_web_weights.py --checkpoint ./checkpoints/model.pt --out ../../models/enhance_params.json
# optional ONNX:
# python export_onnx.py --checkpoint ./checkpoints/model.pt --out ../../models/enhance_params.onnx
```

Then `npm run build` copies `models/` → `demo/models/`.

In the demo, select **Модель TinyCNN**. Without this file the demo uses **эвристику** only.
