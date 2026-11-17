# Final Deep Learning Model: SAM-Road Vector Topology Architecture

## 1. Architecture Overview
The core model in `models/road_model.py` integrates four primary sub-networks:
1. `sam_encoder.py`: SAM ViT-B visual backbone (12 blocks, 768-dim, blocks 1-8 frozen, blocks 9-12 fine-tuned, neck projecting to 256 channels).
2. `geometry_decoder.py`: 4-stage ConvTranspose2d upsampler from (256, 32, 32) to (2, 512, 512), predicting Road Centerline and Intersection probability heatmaps.
3. `nms_extractor.py`: GPU 2D Max Pooling (15x15 kernel, min distance 16px) extracting discrete intersection vertices V.
4. `topology_decoder.py`: Bilinear feature sampler from SAM latents + 2-layer Graph Transformer predicting edge connectivity E.

## 2. Training
```bash
python train.py
```
- FP16 Mixed Precision (`torch.cuda.amp.autocast`).
- 16-Step Gradient Accumulation (effective batch size 16 on 4GB GTX 1650).
- AdamW optimizer with differential learning rates (3e-4 decoder, 3e-5 encoder).
- 20 epochs on CityScale dataset (55% total loss drop, 61% geometry error reduction).

## 3. Standalone Inference
```bash
python predict.py --image test_samples/test_satellite.jpg
```
Outputs:
- Vector graph overlay PNG (`overlay.png`)
- Topology adjacency graph JSON (`graph.json`)
