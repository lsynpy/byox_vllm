# Consistency Check Between ByoX and Nano

This document outlines the steps to verify consistency between the ByoX and Nano implementations.

## Checklist

- [x] Check token ID consistency between ByoX and Nano
- [ ] Compare embedding layer outputs
- [ ] Compare first decoder layer outputs
  - [ ] Input LayerNorm output
  - [ ] Self-Attention output
  - [ ] Post-Attention LayerNorm output
  - [ ] MLP output
    - [ ] MLP LayerNorm output
    - [ ] MLP dense layers output
  - [ ] Final decoder layer output
- [ ] Compare subsequent decoder layer outputs (if first layer matches)
- [ ] Compare RMSNorm output (if all decoder layers match)
- [ ] Compare final logits/LM head output (if previous steps match)
