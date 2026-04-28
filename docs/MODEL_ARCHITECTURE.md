# Model Architecture

Model name:

```text
marine3d_transformer
```

The model receives the reconstructed input tensor concatenated with the `visible_mask`. For 8 target variables this yields 16 input channels:

```text
input_channels = input + visible_mask
output_channels = 8 reconstructed ocean variables
```

The backbone is a 3D reconstruction network with:

- 3D U-Net style encoder for local volumetric features.
- Transformer/mixer bottleneck for broader spatial-depth context.
- 3D decoder for dense volumetric prediction.

Topography is not appended as an input image channel. In the final chain, topography is used as runtime constraint and evaluation metadata through wet masks, near-bottom masks, slope weights, roughness, and patch-level topo metadata.
