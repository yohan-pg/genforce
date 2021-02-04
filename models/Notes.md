

# Differences from our approach
- PixelNorm in mapper & generator: doesn't center
- Applies a new mean after the coloring (like stylegan)
- "w scale": scale factor for the affine projection (how is it different from the default torch init?)
- truncation on w plus, applied both during training and during inference: for the first layers, truncate w to be close to the mean measured by a moving average over w's obtained in training. 
- they also implement the stylegan training style mixing: with a certain probaiblity, mix two styles, by taking the first n layers of one style and the rest from there other. n is decided randomly.
- progressive growth: init resolution at 4, progressively grow by adding layers. when a new layer is added, slowly interpolate its contribution up to 1 (half the time), and then train with it fixed at 1 for the other half. After that, add a new layer and repeat.
- Tanh instead of sigmoid at the end: plays better with leaky relu apparantly. 
- Leaky relu in generator.
- Blur after upsample to reduce upsampling artifacts following nearest upsampling instead of a better interpolation or whatever.
- Const init to ones.
- Fused scale: choice between conv2d transpose or upsample. They start using it after a certain resolution
- In the anime example, not usre how progressive growth goes. 


# Debugging

√ grey init
√ bias missing 
√ no centering
√ compared blockwise with block size 1 to adain, looks all good
- w scaled by a constant factor
- train with block size 1 compared to adain (does the MLP make a difference?)
- bug d'affichage?
- shared params for style to mean and style to cov
- relu in MLP




# Experiments todo

√ Blocksize 1/64 Affine

- Blocksize 64 512 MLP

- Baseline longer train
- Blocksize ? ? MLP + longer training
- Blocksize ? Affine + longer training
