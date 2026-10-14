# WaFormer

Accurate and efficient waveform classification is critical for integrated radar-communication systems, especially on resource-constrained platforms facing spectrum congestion. This paper introduces WaFormer, a lightweight Transformer architecture tailored for time-frequency spectrogram analysis. WaFormer integrates an Anisotropic Depthwise Aggregation (ADA) module for axis-aware local feature extraction with Warped Rotary (WaRo) axial attention for global context modeling. Its core innovation, Warped Positional Encoding (WPE), employs learnable monotonic splines to adaptively rescale coordinates for rotary positional encoding, optionally augmented by a warped-space distance bias. This allows WaFormer to focus positional sensitivity on salient signal structures while preserving relative order with minimal overhead. Experiments on a synthetic signal dataset comprising $12$ radar and communication waveform classes under diverse channel impairments confirm that WaFormer achieves $91.18\%$ accuracy with only $58$K parameters and a model-level inference latency of $0.738$ ms, demonstrating high performance under the considered channel impairments and a favorable accuracy-efficiency trade-off over existing deep models.


![Architecture](https://github.com/DatChanThanh/WaFormer/blob/38cdd525727b710b6d2fed05073d2373881b161c/architecture.png)


The dataset can be download on [Google Drive](https://drive.google.com/drive/u/1/folders/15TJjTUcQEKmzlx7vJDgb7TK7XPZJZ5hF) (please report if not available).

 If there is any error or need to be discussed, please email to [Thanh-Dat Tran](https://github.com/DatChanThanh) via [trandatt21@gmail.com](mailto:trandatt21@gmail.com).
