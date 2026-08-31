# Mandelbrot Generalization

- Infinite complexity prevents overfitting at any partitioning, but a half-plane split is visually striking; ~perfect accuracy can only be achieved by generalization
- FFNs trained on backprop will never generalize, but RNNs coould, but appear to not

It is clear that due to no-free-lunch "theorem", we cannot find a perfect optimizer. But it's clear practically, that SGD kicks ass. It's likely that with few inductive biases, and the problems they're suited to, that optimizers can be ranked, some more general than others. SGD implicitly carries such a bias and yet clearly has much room to be improved.

As transformers are largely successful due to the symmetries they encode, their stability, and crucially their hardward-sympathy, the problem of strong generalization equires innovation at the low-level or the objective-level.

### RNNs

- 7k parameter RNN with fourier-encodings
- Training generally grows eventually unstable, but resists stable overfitting wrt val loss
- PCN-based RNNs are difficult to fit at all because the very high effective suffers from the PCN "slow information dissemination" problem; it's also unclear how best to relax and perform weight decay while maintaining parameter tying.

[RNN at 100 epochs](artifacts/mandel/rnn_100.png)

Basic generalization arises immediately, partial structure of bulb present above the y axis.

There seems to be repeating structure vertically -- it's unclear if fourier encodings are encouraging this behavior, but it is consistent

[RNN at 3000 epochs](artifacts/mandel/rnn_3000.png)

By even 300 epochs, the general structure forming above the y-axis disappears quickly. Proto-general structures are actively antagonized by SGD.

#### Using Hyper Neural Networks

Generating the RNN using a HNN has profoundly different optimization dynamics. We found we can absolutely not grokk on this problem -- SGD strongly prefers to optimize error before weight norm and we cannot ever quite overfit on this problem so grokking never occurs. The HNN however can optimize the target network loss independently of the amount of parameters in the target network. As parameters for optimization are decoupled from representation, the loss can be descended in a less biased way. However to make this tractable, I used a coordinate network to decode the HNN latent into the target network. The coordinate network is under-parameterized wrt the target network, creating a spectral bias, hurting interpretation. But generalization does appear to be dramatically better and proto-general structures remain stable even at 150k epochs.

[HNN-made RNN at 150k epochs](artifacts/mandel/hnn_150k.png)

Bizarrely, the HNNs seem to under-go interesting phase transitions during training.

[HNN training dynamics](artifacts/mandel/hnn_train_curve.png)

Note that one of the axis is not logarithmic. The HNN appears to internally restructure as to massively shrink the low-loss network subspace. Speculatively, followed by a lag, overfitting occurs -- it's unclear if it's directly related. Intuitively, we expect high quality HNN-based optimization to require a large frontier of "candiate target networks". The HNN, when optimizing the average target loss is motivated to crush the subspace, reducing to standard SGD once target network parameter diversity hits zero.

Confirmed the minimum tanh-based FFN that can fit (z^2 + c) at high depth.

[Learning $(z^2+c)$ directly)](artifacts/mandel/ffn_tight.png])

Architecture tanh-FFN, $4 \to 8 \to 2$, depth=1, hidden=8
Input and outputs are minimal to represent complex numbers with real activations.
